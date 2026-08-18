"""
Statistics engine.

Pure, presentation-independent computation of ScanStatistics from a
ScanResult. No Rich, no CLI, no formatting concerns - this is business
logic the terminal summary and (in a later milestone) the JSON/Markdown/
HTML reports all consume the same way, so the numbers can never drift
between output formats.

Correlation: scanner_findings and ai_findings are joined by finding_id,
never by list position or matching lengths - see _match_ai_findings.
Computation is two linear passes (one over scanner_findings, one over
ai_findings) plus O(1) set lookups for the join, not an O(n*m) scan.
"""
from ..contracts import AIEnrichedFinding, ScanResult, ScannerFinding
from .models import AIEnrichmentStatus, ConfidenceStatistics, ScanStatistics


def calculate_statistics(result: ScanResult) -> ScanStatistics:
    scanner_findings = result.scanner_findings
    ai_findings = result.ai_findings

    severity_counts, scanner_counts, category_counts = _tally_scanner_findings(scanner_findings)
    matched_ai_findings, confidence, verification_counts = _tally_ai_findings(scanner_findings, ai_findings)

    total_findings = len(scanner_findings)
    ai_enrichment_status = _enrichment_status(
        _expected_enrichments(result, total_findings), matched_ai_findings
    )

    return ScanStatistics(
        total_findings=total_findings,
        critical_findings=severity_counts["critical"],
        high_findings=severity_counts["high"],
        medium_findings=severity_counts["medium"],
        low_findings=severity_counts["low"],
        scanner_counts=scanner_counts,
        category_counts=category_counts,
        ai_enrichment_status=ai_enrichment_status,
        correlated_findings=len(result.correlated_findings),
        multi_scanner_findings=sum(1 for c in result.correlated_findings if c.is_multi_scanner),
        matched_ai_findings=matched_ai_findings,
        confidence=confidence,
        verification_counts=verification_counts,
        repository_name=result.repository.name,
        scan_mode=result.metadata.mode.value,
        timestamp=result.metadata.timestamp,
        duration_seconds=result.metadata.duration_seconds,
        files_analyzed=None,
    )


def _tally_scanner_findings(
    scanner_findings: list[ScannerFinding],
) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    scanner_counts: dict[str, int] = {}
    category_counts: dict[str, int] = {}

    for f in scanner_findings:
        severity_counts[f.severity.value] += 1
        scanner_counts[f.scanner] = scanner_counts.get(f.scanner, 0) + 1
        category_counts[f.category] = category_counts.get(f.category, 0) + 1

    return severity_counts, scanner_counts, category_counts


def _tally_ai_findings(
    scanner_findings: list[ScannerFinding],
    ai_findings: list[AIEnrichedFinding],
) -> tuple[int, ConfidenceStatistics, dict[str, int]]:
    scanner_finding_ids = {f.finding_id for f in scanner_findings}

    matched_ai_findings = 0
    confidence_counts = {"high": 0, "medium": 0, "low": 0}
    verification_counts: dict[str, int] = {}
    scores: list[float] = []

    for a in ai_findings:
        if a.finding_id in scanner_finding_ids:
            matched_ai_findings += 1
        confidence_counts[a.confidence_label.value] += 1
        verification_counts[a.verification_status.value] = verification_counts.get(a.verification_status.value, 0) + 1
        scores.append(a.confidence_score)

    confidence = ConfidenceStatistics(
        enriched_count=len(ai_findings),
        high_count=confidence_counts["high"],
        medium_count=confidence_counts["medium"],
        low_count=confidence_counts["low"],
        average_score=(sum(scores) / len(scores)) if scores else None,
        min_score=min(scores) if scores else None,
        max_score=max(scores) if scores else None,
    )

    return matched_ai_findings, confidence, verification_counts


def _enrichment_status(expected_enrichments: int, matched_ai_findings: int) -> AIEnrichmentStatus:
    """Classify enrichment completeness against the number of enrichments actually expected.

    `expected_enrichments` is the *unit the AI pipeline enriches*, not the raw
    finding count - see _expected_enrichments below for why those differ.

    `>=` rather than `==` for the AVAILABLE case: matched can only exceed the
    expectation if the two were counted against different units, and reporting
    "partial" for more enrichment than expected would be the same category of
    false alarm this function exists to avoid.
    """
    if matched_ai_findings == 0:
        return AIEnrichmentStatus.UNAVAILABLE
    if expected_enrichments > 0 and matched_ai_findings >= expected_enrichments:
        return AIEnrichmentStatus.AVAILABLE
    return AIEnrichmentStatus.PARTIAL


def _expected_enrichments(result: ScanResult, total_findings: int) -> int:
    """How many AI enrichments a fully successful run should produce.

    ai/pipeline.py enriches once per *correlated issue*, not once per raw
    finding: main.py passes `correlated_findings=result.correlated_findings`,
    and the pipeline emits one AIEnrichedFinding per group, keyed on the group's
    canonical raw finding_id. On the benchmark repository that is 13 enrichments
    for 17 raw findings.

    Comparing against the raw count therefore reported `partial` on a completely
    successful run - 13 != 17 - and made AVAILABLE unreachable for any repository
    where correlation grouped anything. Worse, the documented failure count
    (expected minus matched) implied four findings had failed when none had.

    Falls back to `total_findings` when there are no correlation groups at all,
    which keeps every pre-correlation result - and any caller that builds a
    ScanResult without them - classified exactly as before.
    """
    return len(result.correlated_findings) or total_findings
