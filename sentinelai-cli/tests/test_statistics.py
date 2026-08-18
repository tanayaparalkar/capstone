"""
Tests for the statistics engine (sentinelai/statistics/).

Covers finding counts, open-ended scanner/category distributions,
scanner<->AI correlation by finding_id (never by list position or
matching lengths), confidence/verification aggregation, and metadata
passthrough - plus the edge cases the milestone calls out explicitly
(empty lists, missing optional fields, orphaned AI findings, partial
enrichment).
"""
import json

import pytest
from pydantic import ValidationError

from sentinelai.contracts import (
    AIEnrichedFinding,
    ConfidenceLabel,
    CorrelatedFinding,
    CorrelationRule,
    RepositoryInfo,
    ScanMetadata,
    ScanMode,
    ScanResult,
    ScannerFinding,
    Severity,
    VerificationStatus,
)
from sentinelai.statistics import AIEnrichmentStatus, ScanStatistics, calculate_statistics


def _scanner_finding(**overrides) -> ScannerFinding:
    defaults = dict(
        finding_id="SENT-001",
        scanner="semgrep",
        category="sql-injection",
        severity=Severity.CRITICAL,
        file="app/db/queries.py",
        line_start=47,
        rule_id="semgrep.python.sql-injection.string-concat",
        message="User-supplied input concatenated directly into a SQL query string.",
    )
    defaults.update(overrides)
    return ScannerFinding(**defaults)


def _ai_finding(**overrides) -> AIEnrichedFinding:
    defaults = dict(
        finding_id="SENT-001",
        title="SQL injection via unparameterized query",
        severity=Severity.CRITICAL,
        explanation="The query is built via string concatenation.",
        remediation="Use parameterized queries.",
        confidence_score=0.9,
        confidence_label=ConfidenceLabel.HIGH,
        verification_status=VerificationStatus.VERIFIED,
    )
    defaults.update(overrides)
    return AIEnrichedFinding(**defaults)


def _scan_result(
    scanner_findings=None,
    ai_findings=None,
    mode=ScanMode.STANDARD,
    duration_seconds=None,
    correlated_findings=None,
) -> ScanResult:
    return ScanResult(
        repository=RepositoryInfo(name="demo-app", path="/tmp/demo-app"),
        metadata=ScanMetadata(timestamp="2026-08-08T12:00:00Z", mode=mode, duration_seconds=duration_seconds),
        scanner_findings=scanner_findings or [],
        ai_findings=ai_findings or [],
        correlated_findings=correlated_findings or [],
    )


def _correlated(correlation_id, source_ids, canonical=None):
    """One correlation group over the given raw finding ids."""
    ids = sorted(source_ids)
    return CorrelatedFinding(
        correlation_id=correlation_id,
        canonical_finding_id=canonical or ids[0],
        source_finding_ids=ids,
        scanners=["bandit"],
        file="app.py",
        line_start=1,
        line_end=1,
        cwe="CWE-89",
        severity=Severity.HIGH,
        category="sql-injection",
        rule=CorrelationRule.SINGLETON if len(ids) == 1 else CorrelationRule.SAME_CWE_SAME_LINE,
        correlation_reason="test fixture",
    )


# --- finding counts -----------------------------------------------------------------


def test_zero_findings():
    stats = calculate_statistics(_scan_result())
    assert stats.total_findings == 0
    assert stats.critical_findings == 0
    assert stats.high_findings == 0
    assert stats.medium_findings == 0
    assert stats.low_findings == 0
    assert stats.scanner_counts == {}
    assert stats.category_counts == {}


def test_one_finding():
    stats = calculate_statistics(_scan_result(scanner_findings=[_scanner_finding()]))
    assert stats.total_findings == 1
    assert stats.critical_findings == 1


def test_multiple_findings_counted_correctly():
    findings = [_scanner_finding(finding_id=f"SENT-{i:03d}") for i in range(1, 6)]
    stats = calculate_statistics(_scan_result(scanner_findings=findings))
    assert stats.total_findings == 5


def test_all_severity_levels():
    findings = [
        _scanner_finding(finding_id="SENT-001", severity=Severity.CRITICAL),
        _scanner_finding(finding_id="SENT-002", severity=Severity.CRITICAL),
        _scanner_finding(finding_id="SENT-003", severity=Severity.HIGH),
        _scanner_finding(finding_id="SENT-004", severity=Severity.MEDIUM),
        _scanner_finding(finding_id="SENT-005", severity=Severity.LOW),
    ]
    stats = calculate_statistics(_scan_result(scanner_findings=findings))
    assert stats.critical_findings == 2
    assert stats.high_findings == 1
    assert stats.medium_findings == 1
    assert stats.low_findings == 1
    assert stats.total_findings == 5


# --- scanner / category distribution -----------------------------------------------------------------


def test_multiple_scanners_distribution():
    findings = [
        _scanner_finding(finding_id="SENT-001", scanner="semgrep"),
        _scanner_finding(finding_id="SENT-002", scanner="semgrep"),
        _scanner_finding(finding_id="SENT-003", scanner="bandit"),
        _scanner_finding(finding_id="SENT-004", scanner="gitleaks"),
    ]
    stats = calculate_statistics(_scan_result(scanner_findings=findings))
    assert stats.scanner_counts["semgrep"] == 2
    assert stats.scanner_counts["bandit"] == 1
    assert stats.scanner_counts["gitleaks"] == 1
    assert sum(stats.scanner_counts.values()) == 4


def test_multiple_categories_distribution():
    findings = [
        _scanner_finding(finding_id="SENT-001", category="sql-injection"),
        _scanner_finding(finding_id="SENT-002", category="sql-injection"),
        _scanner_finding(finding_id="SENT-003", category="secret-exposure"),
    ]
    stats = calculate_statistics(_scan_result(scanner_findings=findings))
    assert stats.category_counts["sql-injection"] == 2
    assert stats.category_counts["secret-exposure"] == 1


def test_arbitrary_scanner_names_are_not_hardcoded():
    findings = [
        _scanner_finding(finding_id="SENT-001", scanner="my-custom-scanner-2000"),
        _scanner_finding(finding_id="SENT-002", scanner="another-tool-nobody-heard-of"),
    ]
    stats = calculate_statistics(_scan_result(scanner_findings=findings))
    assert stats.scanner_counts["my-custom-scanner-2000"] == 1
    assert stats.scanner_counts["another-tool-nobody-heard-of"] == 1


def test_arbitrary_categories_are_not_hardcoded():
    findings = [_scanner_finding(finding_id="SENT-001", category="a-brand-new-vulnerability-class")]
    stats = calculate_statistics(_scan_result(scanner_findings=findings))
    assert stats.category_counts["a-brand-new-vulnerability-class"] == 1


# --- scanner <-> AI correlation -----------------------------------------------------------------


def test_scanner_and_ai_findings_different_lengths():
    scanner_findings = [
        _scanner_finding(finding_id="SENT-001"),
        _scanner_finding(finding_id="SENT-002"),
        _scanner_finding(finding_id="SENT-003"),
    ]
    ai_findings = [_ai_finding(finding_id="SENT-002"), _ai_finding(finding_id="SENT-003")]
    stats = calculate_statistics(_scan_result(scanner_findings=scanner_findings, ai_findings=ai_findings))
    assert stats.total_findings == 3
    assert stats.confidence.enriched_count == 2
    assert stats.matched_ai_findings == 2
    assert stats.ai_enrichment_status == AIEnrichmentStatus.PARTIAL


def test_ai_findings_matched_by_finding_id_not_position():
    # ai_findings intentionally in different order and a different subset
    # than scanner_findings - correlation must use finding_id, not index.
    scanner_findings = [
        _scanner_finding(finding_id="SENT-001"),
        _scanner_finding(finding_id="SENT-002"),
        _scanner_finding(finding_id="SENT-003"),
    ]
    ai_findings = [_ai_finding(finding_id="SENT-003"), _ai_finding(finding_id="SENT-001")]
    stats = calculate_statistics(_scan_result(scanner_findings=scanner_findings, ai_findings=ai_findings))
    assert stats.matched_ai_findings == 2


def test_no_ai_findings_reports_unavailable_not_low_confidence():
    stats = calculate_statistics(_scan_result(scanner_findings=[_scanner_finding()]))
    assert stats.ai_enrichment_status == AIEnrichmentStatus.UNAVAILABLE
    assert stats.confidence.enriched_count == 0
    assert stats.confidence.high_count == 0
    assert stats.confidence.medium_count == 0
    assert stats.confidence.low_count == 0
    assert stats.confidence.average_score is None
    assert stats.confidence.min_score is None
    assert stats.confidence.max_score is None


def test_partial_ai_enrichment():
    scanner_findings = [_scanner_finding(finding_id=f"SENT-{i:03d}") for i in range(1, 4)]
    ai_findings = [_ai_finding(finding_id="SENT-001")]
    stats = calculate_statistics(_scan_result(scanner_findings=scanner_findings, ai_findings=ai_findings))
    assert stats.ai_enrichment_status == AIEnrichmentStatus.PARTIAL
    assert stats.matched_ai_findings == 1


def test_full_ai_enrichment_is_available():
    scanner_findings = [_scanner_finding(finding_id="SENT-001"), _scanner_finding(finding_id="SENT-002")]
    ai_findings = [_ai_finding(finding_id="SENT-001"), _ai_finding(finding_id="SENT-002")]
    stats = calculate_statistics(_scan_result(scanner_findings=scanner_findings, ai_findings=ai_findings))
    assert stats.ai_enrichment_status == AIEnrichmentStatus.AVAILABLE
    assert stats.matched_ai_findings == 2


def test_ai_findings_without_corresponding_scanner_finding_not_discarded():
    # Orphaned AI output (no matching scanner finding in this result) still
    # counts toward confidence/verification stats - it isn't silently dropped -
    # but must not inflate matched_ai_findings or ai_enrichment_status.
    scanner_findings = [_scanner_finding(finding_id="SENT-001")]
    ai_findings = [_ai_finding(finding_id="SENT-001"), _ai_finding(finding_id="SENT-999")]
    stats = calculate_statistics(_scan_result(scanner_findings=scanner_findings, ai_findings=ai_findings))
    assert stats.confidence.enriched_count == 2  # both AI records counted
    assert stats.matched_ai_findings == 1  # only one actually corresponds
    assert stats.ai_enrichment_status == AIEnrichmentStatus.AVAILABLE  # 1/1 scanner findings matched


def test_scanner_findings_without_ai_enrichment():
    scanner_findings = [_scanner_finding(finding_id="SENT-001"), _scanner_finding(finding_id="SENT-002")]
    stats = calculate_statistics(_scan_result(scanner_findings=scanner_findings))
    assert stats.matched_ai_findings == 0
    assert stats.ai_enrichment_status == AIEnrichmentStatus.UNAVAILABLE


# --- confidence -----------------------------------------------------------------


def test_confidence_distribution_by_label():
    ai_findings = [
        _ai_finding(finding_id="SENT-001", confidence_label=ConfidenceLabel.HIGH),
        _ai_finding(finding_id="SENT-002", confidence_label=ConfidenceLabel.HIGH),
        _ai_finding(finding_id="SENT-003", confidence_label=ConfidenceLabel.MEDIUM),
        _ai_finding(finding_id="SENT-004", confidence_label=ConfidenceLabel.LOW),
    ]
    scanner_findings = [_scanner_finding(finding_id=f"SENT-{i:03d}") for i in range(1, 5)]
    stats = calculate_statistics(_scan_result(scanner_findings=scanner_findings, ai_findings=ai_findings))
    assert stats.confidence.high_count == 2
    assert stats.confidence.medium_count == 1
    assert stats.confidence.low_count == 1


def test_numeric_confidence_score_statistics():
    ai_findings = [
        _ai_finding(finding_id="SENT-001", confidence_score=0.9),
        _ai_finding(finding_id="SENT-002", confidence_score=0.5),
        _ai_finding(finding_id="SENT-003", confidence_score=0.7),
    ]
    scanner_findings = [_scanner_finding(finding_id=f"SENT-{i:03d}") for i in range(1, 4)]
    stats = calculate_statistics(_scan_result(scanner_findings=scanner_findings, ai_findings=ai_findings))
    assert stats.confidence.min_score == 0.5
    assert stats.confidence.max_score == 0.9
    assert abs(stats.confidence.average_score - 0.7) < 1e-9


# --- verification -----------------------------------------------------------------


def test_verification_status_distribution():
    ai_findings = [
        _ai_finding(finding_id="SENT-001", verification_status=VerificationStatus.VERIFIED),
        _ai_finding(finding_id="SENT-002", verification_status=VerificationStatus.VERIFIED),
        _ai_finding(finding_id="SENT-003", verification_status=VerificationStatus.REJECTED),
        _ai_finding(finding_id="SENT-004", verification_status=VerificationStatus.INSUFFICIENT_EVIDENCE),
    ]
    scanner_findings = [_scanner_finding(finding_id=f"SENT-{i:03d}") for i in range(1, 5)]
    stats = calculate_statistics(_scan_result(scanner_findings=scanner_findings, ai_findings=ai_findings))
    assert stats.verification_counts["verified"] == 2
    assert stats.verification_counts["rejected"] == 1
    assert stats.verification_counts["insufficient_evidence"] == 1
    assert "unverified" not in stats.verification_counts  # never occurred - not assumed present


def test_verification_counts_empty_without_ai_findings():
    stats = calculate_statistics(_scan_result(scanner_findings=[_scanner_finding()]))
    assert stats.verification_counts == {}


# --- repository / scan metadata -----------------------------------------------------------------


def test_metadata_passthrough():
    result = _scan_result(mode=ScanMode.FULL, duration_seconds=12.5)
    stats = calculate_statistics(result)
    assert stats.repository_name == "demo-app"
    assert stats.scan_mode == "full"
    assert stats.duration_seconds == 12.5
    assert stats.timestamp == result.metadata.timestamp


def test_missing_optional_metadata_stays_none_not_invented():
    result = _scan_result(duration_seconds=None)
    stats = calculate_statistics(result)
    assert stats.duration_seconds is None
    assert stats.files_analyzed is None


# --- edge cases -----------------------------------------------------------------


def test_scanner_finding_with_no_file_or_line():
    finding = _scanner_finding(file=None, line_start=None, line_end=None)
    stats = calculate_statistics(_scan_result(scanner_findings=[finding]))
    assert stats.total_findings == 1


def test_ai_finding_without_optional_fields():
    finding = _ai_finding(
        evidence=None, repository_context=None, exploit_path=None, impact=None, patch_suggestion=None
    )
    stats = calculate_statistics(
        _scan_result(scanner_findings=[_scanner_finding()], ai_findings=[finding])
    )
    assert stats.confidence.enriched_count == 1
    assert stats.confidence.high_count == 1


def test_invalid_severity_rejected_at_the_contract_boundary():
    # The statistics engine never has to defend against an invalid severity
    # reaching it - Pydantic validation on ScannerFinding already prevents
    # one from being constructed in the first place.
    with pytest.raises(ValidationError):
        _scanner_finding(severity="not-a-real-severity")


# --- determinism -----------------------------------------------------------------


def test_deterministic_results():
    findings = [_scanner_finding(finding_id=f"SENT-{i:03d}", scanner="semgrep") for i in range(1, 4)]
    result = _scan_result(scanner_findings=findings)
    first = calculate_statistics(result)
    second = calculate_statistics(result)
    assert first == second
    assert first.model_dump() == second.model_dump()


# --- serialization -----------------------------------------------------------------


def test_statistics_are_json_serializable_and_reconstructable():
    result = _scan_result(
        scanner_findings=[_scanner_finding()],
        ai_findings=[_ai_finding()],
    )
    stats = calculate_statistics(result)
    restored = ScanStatistics.model_validate(json.loads(stats.model_dump_json()))
    assert restored == stats


# --- ai_enrichment_status counts against the unit the pipeline actually enriches -------
#
# ai/pipeline.py enriches once per *correlated issue*, not once per raw finding.
# Measuring completeness against the raw count reported `partial` on a run where
# nothing failed - on the benchmark repository, 13 enrichments for 17 raw
# findings - and made AVAILABLE unreachable whenever correlation grouped
# anything. It also made the documented failure count (expected minus matched)
# claim four failures that never happened.


def _grouped_result(raw_ids, groups, enriched_ids):
    """A result whose AI findings are keyed on each group's canonical raw id."""
    return _scan_result(
        scanner_findings=[_scanner_finding(finding_id=i) for i in raw_ids],
        ai_findings=[_ai_finding(finding_id=i) for i in enriched_ids],
        correlated_findings=[_correlated(f"CORR-{n:03d}", ids) for n, ids in enumerate(groups, 1)],
    )


def test_every_correlated_issue_enriched_is_available_despite_more_raw_findings():
    """The measured regression: 3 raw findings, 2 correlated issues, both enriched."""
    result = _grouped_result(
        raw_ids=["SENT-001", "SENT-002", "SENT-003"],
        groups=[["SENT-001", "SENT-002"], ["SENT-003"]],
        enriched_ids=["SENT-001", "SENT-003"],
    )

    stats = calculate_statistics(result)

    assert stats.ai_enrichment_status == AIEnrichmentStatus.AVAILABLE
    assert stats.total_findings == 3
    assert stats.correlated_findings == 2
    assert stats.matched_ai_findings == 2


def test_more_raw_findings_than_correlated_issues_does_not_produce_partial():
    """Raw > correlated is the normal case, not a failure signal."""
    result = _grouped_result(
        raw_ids=[f"SENT-{i:03d}" for i in range(1, 6)],
        groups=[["SENT-001", "SENT-002", "SENT-003"], ["SENT-004", "SENT-005"]],
        enriched_ids=["SENT-001", "SENT-004"],
    )

    stats = calculate_statistics(result)

    assert stats.ai_enrichment_status == AIEnrichmentStatus.AVAILABLE
    assert stats.total_findings > stats.correlated_findings


def test_some_correlated_issues_enriched_is_still_partial():
    """A genuine partial failure must still be reported as one."""
    result = _grouped_result(
        raw_ids=["SENT-001", "SENT-002", "SENT-003"],
        groups=[["SENT-001"], ["SENT-002"], ["SENT-003"]],
        enriched_ids=["SENT-001"],
    )

    stats = calculate_statistics(result)

    assert stats.ai_enrichment_status == AIEnrichmentStatus.PARTIAL
    assert stats.correlated_findings == 3
    assert stats.matched_ai_findings == 1


def test_the_failure_count_is_expected_minus_matched_not_raw_minus_matched():
    """The number reported as failed must be real.

    Under the old denominator this result showed 3 - 1 = 2 'failures' while only
    one issue had actually failed.
    """
    result = _grouped_result(
        raw_ids=["SENT-001", "SENT-002", "SENT-003"],
        groups=[["SENT-001", "SENT-002"], ["SENT-003"]],
        enriched_ids=["SENT-001"],
    )

    stats = calculate_statistics(result)

    assert stats.ai_enrichment_status == AIEnrichmentStatus.PARTIAL
    assert stats.correlated_findings - stats.matched_ai_findings == 1  # the true failure count
    assert stats.total_findings - stats.matched_ai_findings == 2  # the misleading one


def test_correlated_issues_with_no_ai_findings_is_unavailable():
    """Scanner-only mode: correlation runs offline, so groups exist without AI."""
    result = _grouped_result(
        raw_ids=["SENT-001", "SENT-002"],
        groups=[["SENT-001", "SENT-002"]],
        enriched_ids=[],
    )

    stats = calculate_statistics(result)

    assert stats.ai_enrichment_status == AIEnrichmentStatus.UNAVAILABLE
    assert stats.correlated_findings == 1
    assert stats.matched_ai_findings == 0


def test_zero_findings_with_no_correlation_is_unchanged():
    stats = calculate_statistics(_scan_result())

    assert stats.total_findings == 0
    assert stats.correlated_findings == 0
    assert stats.ai_enrichment_status == AIEnrichmentStatus.UNAVAILABLE


def test_results_without_correlation_groups_fall_back_to_the_raw_count():
    """Pre-correlation results must classify exactly as they always did."""
    scanner_findings = [_scanner_finding(finding_id="SENT-001"), _scanner_finding(finding_id="SENT-002")]

    full = calculate_statistics(
        _scan_result(scanner_findings=scanner_findings,
                     ai_findings=[_ai_finding(finding_id="SENT-001"), _ai_finding(finding_id="SENT-002")])
    )
    partial = calculate_statistics(
        _scan_result(scanner_findings=scanner_findings, ai_findings=[_ai_finding(finding_id="SENT-001")])
    )

    assert full.ai_enrichment_status == AIEnrichmentStatus.AVAILABLE
    assert partial.ai_enrichment_status == AIEnrichmentStatus.PARTIAL


def test_matched_exceeding_expected_is_not_reported_as_partial():
    """Defensive: more enrichment than expected is not a partial failure."""
    result = _grouped_result(
        raw_ids=["SENT-001", "SENT-002"],
        groups=[["SENT-001", "SENT-002"]],
        enriched_ids=["SENT-001", "SENT-002"],
    )

    stats = calculate_statistics(result)

    assert stats.matched_ai_findings > stats.correlated_findings
    assert stats.ai_enrichment_status == AIEnrichmentStatus.AVAILABLE
