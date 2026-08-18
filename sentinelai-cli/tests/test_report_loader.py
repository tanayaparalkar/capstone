"""
Tests for loading a previously-saved SentinelAI JSON report back into a
ScanResult + ScanStatistics (sentinelai/reporting/loader.py), which
powers `sentinelai report`.

These round-trip real output from the JSON report generator itself
(to_json), so they exercise the true save -> load path rather than a
hand-built fixture that might drift from what the generator actually
produces.
"""
import json

import pytest

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
from sentinelai.reporting import to_json
from sentinelai.reporting.loader import ReportLoadError, load_scan_result
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
        exploit_path="Submitting \"' OR '1'='1' --\" bypasses authentication.",
        impact="Full authentication bypass.",
        remediation="Use parameterized queries.",
        patch_suggestion="-old\n+new",
        confidence_score=0.917,
        confidence_label=ConfidenceLabel.HIGH,
        verification_status=VerificationStatus.VERIFIED,
        related_findings=["SENT-002"],
        references=["CWE-89"],
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


def _write_report(tmp_path, result: ScanResult, filename="scan-result.json"):
    content = to_json(result, calculate_statistics(result))
    path = tmp_path / filename
    path.write_text(content, encoding="utf-8")
    return path


# --- round trip -----------------------------------------------------------------


def test_round_trip_preserves_scanner_findings(tmp_path):
    result = _scan_result(scanner_findings=[_scanner_finding()])
    path = _write_report(tmp_path, result)
    loaded_result, _ = load_scan_result(path)
    assert loaded_result.scanner_findings == result.scanner_findings


def test_round_trip_preserves_ai_findings(tmp_path):
    result = _scan_result(scanner_findings=[_scanner_finding()], ai_findings=[_ai_finding()])
    path = _write_report(tmp_path, result)
    loaded_result, _ = load_scan_result(path)
    assert loaded_result.ai_findings == result.ai_findings


def test_round_trip_preserves_repository_and_scan_metadata(tmp_path):
    result = _scan_result(mode=ScanMode.FULL, duration_seconds=3.5)
    path = _write_report(tmp_path, result)
    loaded_result, _ = load_scan_result(path)
    assert loaded_result.repository == result.repository
    assert loaded_result.metadata == result.metadata


def test_round_trip_preserves_every_ai_field(tmp_path):
    ai = _ai_finding()
    result = _scan_result(scanner_findings=[_scanner_finding()], ai_findings=[ai])
    path = _write_report(tmp_path, result)
    loaded_result, _ = load_scan_result(path)
    loaded_ai = loaded_result.ai_findings[0]
    assert loaded_ai.explanation == ai.explanation
    assert loaded_ai.exploit_path == ai.exploit_path
    assert loaded_ai.impact == ai.impact
    assert loaded_ai.remediation == ai.remediation
    assert loaded_ai.patch_suggestion == ai.patch_suggestion
    assert loaded_ai.confidence_score == ai.confidence_score
    assert loaded_ai.confidence_label == ai.confidence_label
    assert loaded_ai.verification_status == ai.verification_status
    assert loaded_ai.related_findings == ai.related_findings
    assert loaded_ai.references == ai.references


def test_round_trip_statistics_match_original(tmp_path):
    result = _scan_result(scanner_findings=[_scanner_finding()], ai_findings=[_ai_finding()])
    original_stats = calculate_statistics(result)
    path = _write_report(tmp_path, result)
    _, loaded_stats = load_scan_result(path)
    assert loaded_stats == original_stats


def test_tampered_statistics_in_file_are_ignored_and_recomputed(tmp_path):
    result = _scan_result(scanner_findings=[_scanner_finding()])
    path = _write_report(tmp_path, result)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["statistics"]["total_findings"] = 9999
    data["statistics"]["critical_findings"] = 9999
    path.write_text(json.dumps(data), encoding="utf-8")

    _, loaded_stats = load_scan_result(path)
    assert loaded_stats.total_findings == 1
    assert loaded_stats.critical_findings == 1


# --- error handling -----------------------------------------------------------------


def test_nonexistent_file_raises_clean_error(tmp_path):
    with pytest.raises(ReportLoadError, match="could not read"):
        load_scan_result(tmp_path / "does-not-exist.json")


def test_invalid_json_raises_clean_error(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(ReportLoadError, match="not valid JSON"):
        load_scan_result(path)


def test_non_dict_json_raises_clean_error(tmp_path):
    path = tmp_path / "list.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ReportLoadError):
        load_scan_result(path)


def test_missing_required_field_raises_clean_error(tmp_path):
    result = _scan_result(scanner_findings=[_scanner_finding()])
    path = _write_report(tmp_path, result)
    data = json.loads(path.read_text(encoding="utf-8"))
    del data["repository"]
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ReportLoadError, match="schema"):
        load_scan_result(path)


def test_invalid_enum_value_raises_clean_error(tmp_path):
    result = _scan_result(scanner_findings=[_scanner_finding()])
    path = _write_report(tmp_path, result)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["findings"]["scanner"][0]["severity"] = "catastrophic"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ReportLoadError, match="schema"):
        load_scan_result(path)


def test_wrong_data_type_raises_clean_error(tmp_path):
    result = _scan_result(scanner_findings=[_scanner_finding()])
    path = _write_report(tmp_path, result)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["findings"]["scanner"][0]["line_start"] = "not-a-number"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ReportLoadError, match="schema"):
        load_scan_result(path)


def test_unsupported_schema_version_raises_clean_error(tmp_path):
    result = _scan_result(scanner_findings=[_scanner_finding()])
    path = _write_report(tmp_path, result)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["schema_version"] = "99.0"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ReportLoadError, match="schema version"):
        load_scan_result(path)


def test_missing_schema_version_raises_clean_error(tmp_path):
    result = _scan_result(scanner_findings=[_scanner_finding()])
    path = _write_report(tmp_path, result)
    data = json.loads(path.read_text(encoding="utf-8"))
    del data["schema_version"]
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ReportLoadError, match="schema version"):
        load_scan_result(path)


# --- unicode / untrusted content -----------------------------------------------------------------


def test_unicode_content_round_trips(tmp_path):
    finding = _scanner_finding(message="Ce message contient des caractères non-ASCII: 中文 ✓")
    result = _scan_result(scanner_findings=[finding])
    path = _write_report(tmp_path, result)
    loaded_result, _ = load_scan_result(path)
    assert loaded_result.scanner_findings[0].message == finding.message


def test_malicious_looking_content_round_trips_as_inert_data(tmp_path):
    payload = '<script>alert(1)</script>"; DROP TABLE findings; --'
    finding = _scanner_finding(message=payload, raw_evidence=payload)
    result = _scan_result(scanner_findings=[finding])
    path = _write_report(tmp_path, result)
    loaded_result, _ = load_scan_result(path)
    assert loaded_result.scanner_findings[0].message == payload
    assert loaded_result.scanner_findings[0].raw_evidence == payload


# --- determinism -----------------------------------------------------------------


def test_deterministic_loading(tmp_path):
    result = _scan_result(scanner_findings=[_scanner_finding()], ai_findings=[_ai_finding()])
    path = _write_report(tmp_path, result)
    first_result, first_stats = load_scan_result(path)
    second_result, second_stats = load_scan_result(path)
    assert first_result == second_result
    assert first_stats == second_stats
