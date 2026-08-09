"""
Tests for the SARIF report generator (sentinelai/reporting/sarif_report.py).

build_sarif_report()/to_sarif() are pure functions of an existing
ScanResult + a pre-computed ScanStatistics - no scanner/AI logic, no
statistics recalculation, no second finding model. These tests check
valid JSON output and the SARIF 2.1.0 structure this generator commits
to (schema/version, tool.driver, rules, results, locations, levels) via
targeted structural assertions rather than a full JSON Schema validator
(see the module docstring for why).
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
from sentinelai.reporting import SARIF_VERSION, build_sarif_report, to_sarif
from sentinelai.statistics import calculate_statistics

VALID_SARIF_LEVELS = {"error", "warning", "note", "none"}


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
        confidence_score=0.917,
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


def _report(result: ScanResult) -> dict:
    return build_sarif_report(result, calculate_statistics(result))


# --- valid JSON / top-level structure -----------------------------------------------------------------


def test_valid_sarif_json_via_to_sarif():
    result = _scan_result(scanner_findings=[_scanner_finding()])
    parsed = json.loads(to_sarif(result, calculate_statistics(result)))
    assert isinstance(parsed, dict)


def test_correct_sarif_version():
    report = _report(_scan_result())
    assert report["version"] == "2.1.0"
    assert SARIF_VERSION == "2.1.0"


def test_correct_top_level_structure():
    report = _report(_scan_result(scanner_findings=[_scanner_finding()]))
    assert "$schema" in report
    assert report["$schema"].startswith("https://")
    assert "version" in report
    assert isinstance(report["runs"], list)
    assert len(report["runs"]) == 1
    run = report["runs"][0]
    assert "tool" in run
    assert "results" in run


# --- tool/driver metadata -----------------------------------------------------------------


def test_tool_driver_metadata():
    report = _report(_scan_result())
    driver = report["runs"][0]["tool"]["driver"]
    assert driver["name"] == "SentinelAI"
    assert "version" in driver
    assert isinstance(driver["version"], str)
    assert "rules" in driver


# --- rule mapping -----------------------------------------------------------------


def test_rule_mapping_uses_existing_rule_id():
    finding = _scanner_finding(rule_id="semgrep.python.sql-injection.string-concat")
    report = _report(_scan_result(scanner_findings=[finding]))
    rules = report["runs"][0]["tool"]["driver"]["rules"]
    assert len(rules) == 1
    assert rules[0]["id"] == "semgrep.python.sql-injection.string-concat"
    assert rules[0]["name"] == finding.category
    assert rules[0]["shortDescription"]["text"] == finding.message


def test_rules_deduplicated_by_rule_id():
    findings = [
        _scanner_finding(finding_id="SENT-001", rule_id="same.rule"),
        _scanner_finding(finding_id="SENT-002", rule_id="same.rule"),
    ]
    report = _report(_scan_result(scanner_findings=findings))
    rules = report["runs"][0]["tool"]["driver"]["rules"]
    assert len(rules) == 1


def test_multiple_distinct_rules():
    findings = [
        _scanner_finding(finding_id="SENT-001", rule_id="rule.one"),
        _scanner_finding(finding_id="SENT-002", rule_id="rule.two"),
    ]
    report = _report(_scan_result(scanner_findings=findings))
    rule_ids = {r["id"] for r in report["runs"][0]["tool"]["driver"]["rules"]}
    assert rule_ids == {"rule.one", "rule.two"}


# --- result mapping -----------------------------------------------------------------


def test_result_mapping_basic_fields():
    finding = _scanner_finding()
    report = _report(_scan_result(scanner_findings=[finding]))
    result = report["runs"][0]["results"][0]
    assert result["ruleId"] == finding.rule_id
    assert result["message"]["text"] == finding.message
    assert result["properties"]["sentinelai"]["findingId"] == finding.finding_id
    assert result["properties"]["sentinelai"]["scanner"] == finding.scanner
    assert result["properties"]["sentinelai"]["category"] == finding.category


def test_result_rule_index_matches_rules_array():
    findings = [
        _scanner_finding(finding_id="SENT-001", rule_id="rule.one"),
        _scanner_finding(finding_id="SENT-002", rule_id="rule.two"),
    ]
    report = _report(_scan_result(scanner_findings=findings))
    rules = report["runs"][0]["tool"]["driver"]["rules"]
    for result in report["runs"][0]["results"]:
        expected_index = next(i for i, r in enumerate(rules) if r["id"] == result["ruleId"])
        assert result["ruleIndex"] == expected_index


# --- severity mapping -----------------------------------------------------------------


def test_severity_mapping_all_levels():
    findings = [
        _scanner_finding(finding_id="SENT-001", severity=Severity.CRITICAL),
        _scanner_finding(finding_id="SENT-002", severity=Severity.HIGH),
        _scanner_finding(finding_id="SENT-003", severity=Severity.MEDIUM),
        _scanner_finding(finding_id="SENT-004", severity=Severity.LOW),
    ]
    report = _report(_scan_result(scanner_findings=findings))
    levels_by_id = {r["properties"]["sentinelai"]["findingId"]: r["level"] for r in report["runs"][0]["results"]}
    assert levels_by_id["SENT-001"] == "error"
    assert levels_by_id["SENT-002"] == "error"
    assert levels_by_id["SENT-003"] == "warning"
    assert levels_by_id["SENT-004"] == "note"


def test_all_result_levels_are_valid_sarif_levels():
    findings = [_scanner_finding(finding_id=f"SENT-{i:03d}", severity=sev) for i, sev in enumerate(Severity, start=1)]
    report = _report(_scan_result(scanner_findings=findings))
    for result in report["runs"][0]["results"]:
        assert result["level"] in VALID_SARIF_LEVELS


def test_ai_confidence_does_not_alter_severity_level():
    finding = _scanner_finding(severity=Severity.CRITICAL)
    ai = _ai_finding(confidence_score=0.05, confidence_label=ConfidenceLabel.LOW)
    report = _report(_scan_result(scanner_findings=[finding], ai_findings=[ai]))
    assert report["runs"][0]["results"][0]["level"] == "error"


# --- file / line locations -----------------------------------------------------------------


def test_file_location():
    finding = _scanner_finding(file="config/settings.py", line_start=14, line_end=14)
    report = _report(_scan_result(scanner_findings=[finding]))
    location = report["runs"][0]["results"][0]["locations"][0]
    assert location["physicalLocation"]["artifactLocation"]["uri"] == "config/settings.py"


def test_file_location_normalizes_backslashes_to_forward_slashes():
    # A cross-platform defensive fix found during the Milestone 12 audit:
    # SARIF requires artifactLocation.uri to use forward slashes. The mock
    # provider never produces backslash paths, but a future Windows-hosted
    # provider plausibly could - normalize regardless of what it supplies.
    finding = _scanner_finding(file="config\\settings.py", line_start=14)
    report = _report(_scan_result(scanner_findings=[finding]))
    location = report["runs"][0]["results"][0]["locations"][0]
    assert location["physicalLocation"]["artifactLocation"]["uri"] == "config/settings.py"


def test_line_location_is_1_based_and_matches_input():
    finding = _scanner_finding(file="a.py", line_start=47, line_end=47)
    report = _report(_scan_result(scanner_findings=[finding]))
    region = report["runs"][0]["results"][0]["locations"][0]["physicalLocation"]["region"]
    assert region["startLine"] == 47
    assert "endLine" not in region  # equal to startLine, omitted rather than redundant


def test_line_range_includes_end_line_when_different():
    finding = _scanner_finding(file="a.py", line_start=10, line_end=15)
    report = _report(_scan_result(scanner_findings=[finding]))
    region = report["runs"][0]["results"][0]["locations"][0]["physicalLocation"]["region"]
    assert region["startLine"] == 10
    assert region["endLine"] == 15


def test_missing_line_handling_file_only():
    finding = _scanner_finding(file="requirements.txt", line_start=None, line_end=None)
    report = _report(_scan_result(scanner_findings=[finding]))
    physical_location = report["runs"][0]["results"][0]["locations"][0]["physicalLocation"]
    assert physical_location["artifactLocation"]["uri"] == "requirements.txt"
    assert "region" not in physical_location


def test_missing_file_produces_no_locations():
    finding = _scanner_finding(file=None, line_start=None, line_end=None)
    report = _report(_scan_result(scanner_findings=[finding]))
    result = report["runs"][0]["results"][0]
    assert "locations" not in result


# --- CWE -----------------------------------------------------------------


def test_cwe_mapping_in_rule_and_result():
    finding = _scanner_finding(cwe="CWE-798")
    report = _report(_scan_result(scanner_findings=[finding]))
    rule = report["runs"][0]["tool"]["driver"]["rules"][0]
    result = report["runs"][0]["results"][0]
    assert rule["properties"]["cwe"] == "CWE-798"
    assert result["properties"]["sentinelai"]["cwe"] == "CWE-798"


def test_missing_cwe_omitted_not_invented():
    finding = _scanner_finding(cwe=None)
    report = _report(_scan_result(scanner_findings=[finding]))
    rule = report["runs"][0]["tool"]["driver"]["rules"][0]
    result = report["runs"][0]["results"][0]
    assert "properties" not in rule
    assert "cwe" not in result["properties"]["sentinelai"]


# --- evidence -----------------------------------------------------------------


def test_raw_evidence_preserved_in_properties():
    evidence = "obj = pickle.loads(cache.get(key))"
    finding = _scanner_finding(raw_evidence=evidence)
    report = _report(_scan_result(scanner_findings=[finding]))
    assert report["runs"][0]["results"][0]["properties"]["sentinelai"]["rawEvidence"] == evidence


# --- scanner distribution -----------------------------------------------------------------


def test_multiple_scanners_represented():
    findings = [
        _scanner_finding(finding_id="SENT-001", scanner="semgrep"),
        _scanner_finding(finding_id="SENT-002", scanner="bandit"),
        _scanner_finding(finding_id="SENT-003", scanner="gitleaks"),
    ]
    report = _report(_scan_result(scanner_findings=findings))
    scanners = {r["properties"]["sentinelai"]["scanner"] for r in report["runs"][0]["results"]}
    assert scanners == {"semgrep", "bandit", "gitleaks"}


def test_arbitrary_scanner_names_not_hardcoded():
    finding = _scanner_finding(scanner="my-custom-scanner-2000")
    report = _report(_scan_result(scanner_findings=[finding]))
    assert report["runs"][0]["results"][0]["properties"]["sentinelai"]["scanner"] == "my-custom-scanner-2000"


# --- AI enrichment / correlation -----------------------------------------------------------------


def test_ai_enriched_finding_has_ai_properties():
    result = _scan_result(scanner_findings=[_scanner_finding()], ai_findings=[_ai_finding()])
    report = _report(result)
    ai_props = report["runs"][0]["results"][0]["properties"]["sentinelai"]["ai"]
    assert ai_props["confidenceScore"] == 0.917
    assert ai_props["confidenceLabel"] == "high"
    assert ai_props["verificationStatus"] == "verified"
    assert ai_props["explanation"]
    assert ai_props["remediation"]


def test_scanner_only_finding_has_no_ai_properties():
    report = _report(_scan_result(scanner_findings=[_scanner_finding()]))
    assert "ai" not in report["runs"][0]["results"][0]["properties"]["sentinelai"]


def test_mismatched_scanner_and_ai_lists_correlate_by_finding_id():
    # spec example: scanner SENT-001/002/003, AI SENT-002/003
    scanner_findings = [
        _scanner_finding(finding_id="SENT-001"),
        _scanner_finding(finding_id="SENT-002"),
        _scanner_finding(finding_id="SENT-003"),
    ]
    ai_findings = [_ai_finding(finding_id="SENT-002"), _ai_finding(finding_id="SENT-003")]
    report = _report(_scan_result(scanner_findings=scanner_findings, ai_findings=ai_findings))
    results = report["runs"][0]["results"]
    assert len(results) == 3
    by_id = {r["properties"]["sentinelai"]["findingId"]: r for r in results}
    assert "ai" not in by_id["SENT-001"]["properties"]["sentinelai"]
    assert "ai" in by_id["SENT-002"]["properties"]["sentinelai"]
    assert "ai" in by_id["SENT-003"]["properties"]["sentinelai"]


def test_orphan_ai_finding_does_not_create_extra_result():
    scanner_findings = [_scanner_finding(finding_id="SENT-001")]
    ai_findings = [_ai_finding(finding_id="SENT-001"), _ai_finding(finding_id="SENT-999")]
    report = _report(_scan_result(scanner_findings=scanner_findings, ai_findings=ai_findings))
    assert len(report["runs"][0]["results"]) == 1  # one scanner finding -> exactly one result


# --- unicode -----------------------------------------------------------------


def test_unicode_content_preserved():
    finding = _scanner_finding(message="Ce message contient des caractères non-ASCII: 中文 ✓")
    report = _report(_scan_result(scanner_findings=[finding]))
    assert report["runs"][0]["results"][0]["message"]["text"] == finding.message


# --- security: malicious-looking content -----------------------------------------------------------------


def test_script_tag_content_serializes_as_inert_json_string():
    # SARIF/JSON consumers parse this as data, never render or execute it -
    # so <script> legitimately appears verbatim as a JSON string *value*.
    # What matters is that it stays a well-formed, quoted string field and
    # never breaks the surrounding JSON structure.
    payload = "<script>alert(1)</script>"
    finding = _scanner_finding(message=payload, raw_evidence=payload)
    result = _scan_result(scanner_findings=[finding])
    output = to_sarif(result, calculate_statistics(result))
    parsed = json.loads(output)  # would raise if the payload corrupted JSON structure
    sarif_result = parsed["runs"][0]["results"][0]
    assert sarif_result["message"]["text"] == payload
    assert sarif_result["properties"]["sentinelai"]["rawEvidence"] == payload


def test_json_breaking_characters_stay_valid_json():
    payload = '"; DROP TABLE findings; -- \\ \n\t{"injected": true}'
    finding = _scanner_finding(message=payload)
    result = _scan_result(scanner_findings=[finding])
    output = to_sarif(result, calculate_statistics(result))
    parsed = json.loads(output)  # would raise if serialization were broken
    assert parsed["runs"][0]["results"][0]["message"]["text"] == payload


# --- determinism -----------------------------------------------------------------


def test_deterministic_result_ordering():
    findings = [
        _scanner_finding(finding_id="SENT-003"),
        _scanner_finding(finding_id="SENT-001"),
        _scanner_finding(finding_id="SENT-002"),
    ]
    report = _report(_scan_result(scanner_findings=findings))
    ids = [r["properties"]["sentinelai"]["findingId"] for r in report["runs"][0]["results"]]
    assert ids == ["SENT-001", "SENT-002", "SENT-003"]


def test_deterministic_output_for_same_input():
    result = _scan_result(scanner_findings=[_scanner_finding()])
    stats = calculate_statistics(result)
    assert to_sarif(result, stats) == to_sarif(result, stats)


# --- empty findings -----------------------------------------------------------------


def test_empty_findings_produces_valid_empty_sarif():
    report = _report(_scan_result())
    assert report["runs"][0]["results"] == []
    assert report["runs"][0]["tool"]["driver"]["rules"] == []
