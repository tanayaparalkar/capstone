"""Tests for sentinelai.scanners.gitleaks - the Gitleaks scanner wrapper. subprocess.run is always mocked; no real Gitleaks installation is ever invoked.

Fixture payload shapes below mirror gitleaks 8.30.1's real JSON output,
captured by running `brew install gitleaks` and inspecting actual results
during development of the wrapper - see gitleaks.py's module docstring.
"""
import json
import subprocess
from unittest.mock import patch

import pytest

from sentinelai.backend.context_builder import build_repository_context
from sentinelai.backend.loader import load_repository
from sentinelai.contracts import ScannerFinding, Severity
from sentinelai.scanners.exceptions import ScannerExecutionError
from sentinelai.scanners.gitleaks import GitleaksScanner


def _context(tmp_path):
    return build_repository_context(load_repository(str(tmp_path)))


def _completed(stdout="", stderr="", returncode=0):
    return subprocess.CompletedProcess(args=["gitleaks"], returncode=returncode, stdout=stdout, stderr=stderr)


def _gitleaks_result(**overrides):
    result = {
        "RuleID": "generic-api-key",
        "Description": "Detected a Generic API Key, potentially exposing access to various services and sensitive operations.",
        "StartLine": 2,
        "EndLine": 2,
        "StartColumn": 2,
        "EndColumn": 67,
        "Match": 'aws_secret_access_key = "REDACTED"',
        "Secret": "REDACTED",
        "File": "/repo/config.py",
        "SymlinkFile": "",
        "Commit": "",
        "Entropy": 4.8341837,
        "Author": "",
        "Email": "",
        "Date": "",
        "Message": "",
        "Tags": [],
        "Fingerprint": "/repo/config.py:generic-api-key:2",
    }
    result.update(overrides)
    return result


def test_successful_json_parsing(tmp_path):
    payload = [_gitleaks_result()]
    completed = _completed(stdout=json.dumps(payload), returncode=1)

    with patch("sentinelai.scanners.gitleaks.subprocess.run", return_value=completed):
        findings = GitleaksScanner().scan(_context(tmp_path))

    assert len(findings) == 1
    finding = findings[0]
    assert finding == ScannerFinding(
        finding_id="/repo/config.py:generic-api-key:2",
        scanner="gitleaks",
        category="generic-api-key",
        severity=Severity.HIGH,
        file="/repo/config.py",
        line_start=2,
        line_end=2,
        rule_id="generic-api-key",
        message="Detected a Generic API Key, potentially exposing access to various services and sensitive operations.",
        raw_evidence='aws_secret_access_key = "REDACTED"',
        cwe=None,
    )


def test_empty_findings_exit_code_zero(tmp_path):
    completed = _completed(stdout="[]\n", returncode=0)

    with patch("sentinelai.scanners.gitleaks.subprocess.run", return_value=completed):
        findings = GitleaksScanner().scan(_context(tmp_path))

    assert findings == []


def test_top_level_json_not_a_list_defaults_to_empty(tmp_path):
    # Gitleaks' schema is a bare top-level array, not {"results": [...]}; anything
    # else that still parses as valid JSON is treated leniently as no findings.
    completed = _completed(stdout="null", returncode=0)

    with patch("sentinelai.scanners.gitleaks.subprocess.run", return_value=completed):
        findings = GitleaksScanner().scan(_context(tmp_path))

    assert findings == []


def test_missing_executable_raises_scanner_execution_error(tmp_path):
    with patch("sentinelai.scanners.gitleaks.subprocess.run", side_effect=FileNotFoundError("no such file")):
        with pytest.raises(ScannerExecutionError, match="gitleaks"):
            GitleaksScanner().scan(_context(tmp_path))


def test_exit_code_one_with_leaks_found_is_not_an_error(tmp_path):
    payload = [_gitleaks_result()]
    completed = _completed(stdout=json.dumps(payload), returncode=1)

    with patch("sentinelai.scanners.gitleaks.subprocess.run", return_value=completed):
        findings = GitleaksScanner().scan(_context(tmp_path))

    assert len(findings) == 1


def test_exit_code_one_with_empty_stdout_is_a_genuine_failure(tmp_path):
    # Verified empirically: gitleaks also returns exit 1 for a genuine error (e.g. an
    # unreadable target path), with empty (non-JSON) stdout - the ambiguous case this
    # wrapper's docstring documents. Invalid JSON must still raise, even at exit 1.
    completed = _completed(stdout="", stderr="FTL stat /no/such/path: no such file or directory", returncode=1)

    with patch("sentinelai.scanners.gitleaks.subprocess.run", return_value=completed):
        with pytest.raises(ScannerExecutionError):
            GitleaksScanner().scan(_context(tmp_path))


def test_non_success_exit_status_outside_zero_or_one_raises_scanner_execution_error(tmp_path):
    completed = _completed(stdout="", stderr="Error: unknown flag: --nonexistent-flag", returncode=126)

    with patch("sentinelai.scanners.gitleaks.subprocess.run", return_value=completed):
        with pytest.raises(ScannerExecutionError, match="126"):
            GitleaksScanner().scan(_context(tmp_path))


def test_invalid_json_raises_scanner_execution_error(tmp_path):
    completed = _completed(stdout="not json{{{", returncode=0)

    with patch("sentinelai.scanners.gitleaks.subprocess.run", return_value=completed):
        with pytest.raises(ScannerExecutionError):
            GitleaksScanner().scan(_context(tmp_path))


def test_unexpected_subprocess_failure_propagates_as_scanner_execution_error_with_cause(tmp_path):
    original = OSError("permission denied")

    with patch("sentinelai.scanners.gitleaks.subprocess.run", side_effect=original):
        with pytest.raises(ScannerExecutionError) as exc_info:
            GitleaksScanner().scan(_context(tmp_path))

    assert exc_info.value.__cause__ is original


def test_missing_optional_json_fields_use_sensible_defaults(tmp_path):
    minimal_result = {"RuleID": "aws-access-token", "File": "main.py"}
    payload = [minimal_result]
    completed = _completed(stdout=json.dumps(payload), returncode=1)

    with patch("sentinelai.scanners.gitleaks.subprocess.run", return_value=completed):
        findings = GitleaksScanner().scan(_context(tmp_path))

    assert len(findings) == 1
    finding = findings[0]
    assert finding.finding_id == "gitleaks-0"
    assert finding.scanner == "gitleaks"
    assert finding.category == "aws-access-token"
    assert finding.severity == Severity.HIGH
    assert finding.file == "main.py"
    assert finding.line_start is None
    assert finding.line_end is None
    assert finding.rule_id == "aws-access-token"
    assert finding.message == ""
    assert finding.raw_evidence is None
    assert finding.cwe is None


def test_end_line_before_start_line_is_clamped_to_start(tmp_path):
    result = _gitleaks_result(StartLine=10, EndLine=5)
    payload = [result]
    completed = _completed(stdout=json.dumps(payload), returncode=1)

    with patch("sentinelai.scanners.gitleaks.subprocess.run", return_value=completed):
        findings = GitleaksScanner().scan(_context(tmp_path))

    assert findings[0].line_start == 10
    assert findings[0].line_end == 10


def test_positional_finding_id_used_when_fingerprint_absent(tmp_path):
    first = _gitleaks_result(File="a.py")
    del first["Fingerprint"]
    second = _gitleaks_result(File="b.py")
    del second["Fingerprint"]
    payload = [first, second]
    completed = _completed(stdout=json.dumps(payload), returncode=1)

    with patch("sentinelai.scanners.gitleaks.subprocess.run", return_value=completed):
        findings = GitleaksScanner().scan(_context(tmp_path))

    assert [f.finding_id for f in findings] == ["gitleaks-0", "gitleaks-1"]


def test_all_findings_are_severity_high(tmp_path):
    payload = [_gitleaks_result(RuleID="generic-api-key"), _gitleaks_result(RuleID="aws-access-token")]
    completed = _completed(stdout=json.dumps(payload), returncode=1)

    with patch("sentinelai.scanners.gitleaks.subprocess.run", return_value=completed):
        findings = GitleaksScanner().scan(_context(tmp_path))

    assert all(finding.severity == Severity.HIGH for finding in findings)


def test_deterministic_output_across_calls(tmp_path):
    payload = [_gitleaks_result(File="a.py"), _gitleaks_result(File="b.py")]
    completed = _completed(stdout=json.dumps(payload), returncode=1)

    with patch("sentinelai.scanners.gitleaks.subprocess.run", return_value=completed):
        first = GitleaksScanner().scan(_context(tmp_path))
        second = GitleaksScanner().scan(_context(tmp_path))

    assert first == second


def test_scanner_returns_only_scanner_finding_objects(tmp_path):
    payload = [_gitleaks_result(File="a.py"), _gitleaks_result(File="b.py")]
    completed = _completed(stdout=json.dumps(payload), returncode=1)

    with patch("sentinelai.scanners.gitleaks.subprocess.run", return_value=completed):
        findings = GitleaksScanner().scan(_context(tmp_path))

    assert isinstance(findings, list)
    assert all(isinstance(finding, ScannerFinding) for finding in findings)


def test_command_construction_and_repository_path(tmp_path):
    completed = _completed(stdout="[]", returncode=0)

    with patch("sentinelai.scanners.gitleaks.subprocess.run", return_value=completed) as mock_run:
        context = _context(tmp_path)
        GitleaksScanner(executable="my-gitleaks").scan(context)
        command = mock_run.call_args[0][0]

    assert command[0] == "my-gitleaks"
    assert command[1] == "dir"
    assert str(context.repository.absolute_path) in command
    assert "-f" in command
    assert "json" in command
    assert "-r" in command
    assert "-" in command


def test_redact_flag_is_always_passed(tmp_path):
    # Verified empirically that --redact replaces the raw secret value with the
    # literal string "REDACTED" in gitleaks' own JSON output before this wrapper
    # ever parses it - this is a security property, not a formatting choice, so it
    # is not made optional. See module docstring.
    completed = _completed(stdout="[]", returncode=0)

    with patch("sentinelai.scanners.gitleaks.subprocess.run", return_value=completed) as mock_run:
        GitleaksScanner().scan(_context(tmp_path))
        command = mock_run.call_args[0][0]

    assert "--redact" in command
