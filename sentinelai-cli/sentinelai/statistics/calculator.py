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
    ai_enrichment_status = _enrichment_status(total_findings, matched_ai_findings)

    return ScanStatistics(
        total_findings=total_findings,
        critical_findings=severity_counts["critical"],
        high_findings=severity_counts["high"],
        medium_findings=severity_counts["medium"],
        low_findings=severity_counts["low"],
        scanner_counts=scanner_counts,
        category_counts=category_counts,
        ai_enrichment_status=ai_enrichment_status,
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


def _enrichment_status(total_findings: int, matched_ai_findings: int) -> AIEnrichmentStatus:
    if matched_ai_findings == 0:
        return AIEnrichmentStatus.UNAVAILABLE
    if total_findings > 0 and matched_ai_findings == total_findings:
        return AIEnrichmentStatus.AVAILABLE
    return AIEnrichmentStatus.PARTIAL
