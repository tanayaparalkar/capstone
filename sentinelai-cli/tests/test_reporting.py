"""
Tests for the JSON report generator (sentinelai/reporting/).

build_json_report()/to_json() are pure functions of an existing
ScanResult + a pre-computed ScanStatistics - no scanner/AI logic, no
statistics recalculation. These tests exercise that composition directly
rather than going through the full CLI (test_cli.py covers the
CLI-integration angle: stdout purity, file writing, error handling).
"""
import json

from sentinelai.contracts import (
    AIEnrichedFinding,
    ConfidenceLabel,
    RepositoryInfo,
    ScanMetadata,
    ScanMode,
    ScanResult,
    ScannerFinding,
    Severity,
    VerificationStatus,
)
from sentinelai.reporting import REPORT_SCHEMA_VERSION, JSONReport, build_json_report, to_json
from sentinelai.statistics import calculate_statistics


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
        raw_evidence="query = f\"SELECT * FROM users WHERE username = '{username}'\"",
        cwe="CWE-89",
    )
    defaults.update(overrides)
    return ScannerFinding(**defaults)


def _ai_finding(**overrides) -> AIEnrichedFinding:
    defaults = dict(
        finding_id="SENT-001",
        title="SQL injection via unparameterized query",
        severity=Severity.CRITICAL,
        scanner_sources=["semgrep"],
        explanation="The query is built via string concatenation, allowing arbitrary SQL to be injected.",
        remediation="Use parameterized queries.",
        confidence_score=0.92,
        confidence_label=ConfidenceLabel.HIGH,
        verification_status=VerificationStatus.VERIFIED,
    )
    defaults.update(overrides)
    return AIEnrichedFinding(**defaults)


def _scan_result(scanner_findings=None, ai_findings=None, **metadata_overrides) -> ScanResult:
    metadata_defaults = dict(timestamp="2026-08-09T12:00:00Z", mode=ScanMode.STANDARD, duration_seconds=1.5)
    metadata_defaults.update(metadata_overrides)
    return ScanResult(
        repository=RepositoryInfo(name="demo-app", path="/tmp/demo-app", commit_hash="abc123", branch="main"),
        metadata=ScanMetadata(**metadata_defaults),
        scanner_findings=scanner_findings or [],
        ai_findings=ai_findings or [],
    )


def _report_json(result: ScanResult) -> dict:
    stats = calculate_statistics(result)
    return json.loads(to_json(result, stats))


# --- basic structure -----------------------------------------------------------------


def test_valid_json_generation():
    data = _report_json(_scan_result(scanner_findings=[_scanner_finding()]))
    assert isinstance(data, dict)


def test_schema_version_present():
    data = _report_json(_scan_result())
    assert data["schema_version"] == REPORT_SCHEMA_VERSION


def test_empty_findings():
    data = _report_json(_scan_result())
    assert data["findings"]["scanner"] == []
    assert data["findings"]["ai_enriched"] == []
    assert data["statistics"]["total_findings"] == 0


def test_scanner_findings_present_and_complete():
    data = _report_json(_scan_result(scanner_findings=[_scanner_finding()]))
    finding = data["findings"]["scanner"][0]
    assert finding["finding_id"] == "SENT-001"
    assert finding["scanner"] == "semgrep"
    assert finding["category"] == "sql-injection"
    assert finding["severity"] == "critical"
    assert finding["file"] == "app/db/queries.py"
    assert finding["line_start"] == 47
    assert finding["rule_id"] == "semgrep.python.sql-injection.string-concat"


def test_ai_enriched_findings_present():
    result = _scan_result(scanner_findings=[_scanner_finding()], ai_findings=[_ai_finding()])
    data = _report_json(result)
    ai = data["findings"]["ai_enriched"][0]
    assert ai["finding_id"] == "SENT-001"
    assert ai["confidence_label"] == "high"


def test_mixed_scanner_and_ai_findings():
    result = _scan_result(
        scanner_findings=[_scanner_finding(finding_id="SENT-001"), _scanner_finding(finding_id="SENT-002")],
        ai_findings=[_ai_finding(finding_id="SENT-001")],
    )
    data = _report_json(result)
    assert len(data["findings"]["scanner"]) == 2
    assert len(data["findings"]["ai_enriched"]) == 1


# --- statistics integration -----------------------------------------------------------------


def test_statistics_included_and_matches_calculator():
    result = _scan_result(scanner_findings=[_scanner_finding()])
    expected_stats = calculate_statistics(result).model_dump(mode="json")
    data = _report_json(result)
    assert data["statistics"] == expected_stats


def test_statistics_not_recalculated_independently():
    # Passing statistics that don't match the result (e.g. from a filtered
    # subset) proves the generator trusts what it's given rather than
    # deriving its own numbers from result.scanner_findings.
    result = _scan_result(scanner_findings=[_scanner_finding()])
    stats_for_empty_result = calculate_statistics(_scan_result())
    report = build_json_report(result, stats_for_empty_result)
    assert report.statistics.total_findings == 0
    assert len(report.findings.scanner) == 1


# --- repository / scan metadata -----------------------------------------------------------------


def test_repository_metadata_included():
    data = _report_json(_scan_result())
    assert data["repository"]["name"] == "demo-app"
    assert data["repository"]["path"] == "/tmp/demo-app"
    assert data["repository"]["commit_hash"] == "abc123"
    assert data["repository"]["branch"] == "main"


def test_scan_metadata_included():
    data = _report_json(_scan_result(mode=ScanMode.FULL, duration_seconds=3.25))
    assert data["scan"]["mode"] == "full"
    assert data["scan"]["duration_seconds"] == 3.25
    assert "timestamp" in data["scan"]


def test_timestamp_is_iso8601():
    data = _report_json(_scan_result())
    # datetime.fromisoformat handles the 'Z'-normalized offset Pydantic emits
    from datetime import datetime

    datetime.fromisoformat(data["scan"]["timestamp"].replace("Z", "+00:00"))


# --- optional fields / preservation -----------------------------------------------------------------


def test_optional_fields_serialize_as_null_not_dropped():
    finding = _scanner_finding(file=None, line_start=None, line_end=None, cwe=None, raw_evidence=None)
    data = _report_json(_scan_result(scanner_findings=[finding]))
    f = data["findings"]["scanner"][0]
    assert f["file"] is None
    assert f["line_start"] is None
    assert f["cwe"] is None
    assert f["raw_evidence"] is None


def test_confidence_score_preserved_exactly():
    result = _scan_result(scanner_findings=[_scanner_finding()], ai_findings=[_ai_finding(confidence_score=0.837)])
    data = _report_json(result)
    assert data["findings"]["ai_enriched"][0]["confidence_score"] == 0.837


def test_verification_status_preserved():
    result = _scan_result(
        scanner_findings=[_scanner_finding()],
        ai_findings=[_ai_finding(verification_status=VerificationStatus.INSUFFICIENT_EVIDENCE)],
    )
    data = _report_json(result)
    assert data["findings"]["ai_enriched"][0]["verification_status"] == "insufficient_evidence"


def test_cwe_preserved():
    data = _report_json(_scan_result(scanner_findings=[_scanner_finding(cwe="CWE-798")]))
    assert data["findings"]["scanner"][0]["cwe"] == "CWE-798"


def test_raw_evidence_preserved():
    evidence = "AWS_ACCESS_KEY_ID = \"AKIAXXXXXXXXXXXXXXXX\""
    data = _report_json(_scan_result(scanner_findings=[_scanner_finding(raw_evidence=evidence)]))
    assert data["findings"]["scanner"][0]["raw_evidence"] == evidence


# --- unicode -----------------------------------------------------------------


def test_unicode_content_preserved():
    finding = _scanner_finding(message="Ce message contient des caractères non-ASCII: — 中文 ✓")
    data = _report_json(_scan_result(scanner_findings=[finding]))
    assert data["findings"]["scanner"][0]["message"] == finding.message


# --- determinism -----------------------------------------------------------------


def test_deterministic_output_for_same_input():
    result = _scan_result(
        scanner_findings=[
            _scanner_finding(finding_id="SENT-003"),
            _scanner_finding(finding_id="SENT-001"),
            _scanner_finding(finding_id="SENT-002"),
        ]
    )
    first = to_json(result, calculate_statistics(result))
    second = to_json(result, calculate_statistics(result))
    assert first == second


def test_findings_sorted_by_id_regardless_of_input_order():
    result = _scan_result(
        scanner_findings=[
            _scanner_finding(finding_id="SENT-003"),
            _scanner_finding(finding_id="SENT-001"),
            _scanner_finding(finding_id="SENT-002"),
        ]
    )
    data = _report_json(result)
    ids = [f["finding_id"] for f in data["findings"]["scanner"]]
    assert ids == ["SENT-001", "SENT-002", "SENT-003"]


# --- round-trip -----------------------------------------------------------------


def test_report_round_trips_through_json():
    result = _scan_result(scanner_findings=[_scanner_finding()], ai_findings=[_ai_finding()])
    report = build_json_report(result, calculate_statistics(result))
    restored = JSONReport.model_validate(json.loads(report.model_dump_json()))
    assert restored == report
