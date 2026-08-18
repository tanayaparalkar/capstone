"""Tests for sentinelai.scanners.semgrep - the Semgrep scanner wrapper. subprocess.run is always mocked; no real Semgrep installation is ever invoked."""
import json
import subprocess
from unittest.mock import patch

import pytest

from sentinelai.backend.context_builder import build_repository_context
from sentinelai.backend.loader import load_repository
from sentinelai.contracts import ScannerFinding, Severity
from sentinelai.scanners.exceptions import ScannerExecutionError
from sentinelai.scanners.semgrep import SemgrepScanner


def _context(tmp_path):
    return build_repository_context(load_repository(str(tmp_path)))


def _completed(stdout="", stderr="", returncode=0):
    return subprocess.CompletedProcess(args=["semgrep"], returncode=returncode, stdout=stdout, stderr=stderr)


def _semgrep_result(**overrides):
    result = {
        "check_id": "python.lang.security.audit.dangerous-system-call",
        "path": "app.py",
        "start": {"line": 10, "col": 1},
        "end": {"line": 10, "col": 20},
        "extra": {
            "message": "Found dangerous system call.",
            "severity": "ERROR",
            "lines": "os.system(user_input)",
            "fingerprint": "abc123",
            "metadata": {"category": "security", "cwe": ["CWE-78: OS Command Injection"]},
        },
    }
    result.update(overrides)
    return result


def test_successful_json_parsing(tmp_path):
    payload = {"results": [_semgrep_result()]}

    with patch("sentinelai.scanners.semgrep.subprocess.run", return_value=_completed(stdout=json.dumps(payload))):
        findings = SemgrepScanner().scan(_context(tmp_path))

    assert len(findings) == 1
    finding = findings[0]
    assert finding == ScannerFinding(
        finding_id="abc123",
        scanner="semgrep",
        category="security",
        severity=Severity.HIGH,
        file="app.py",
        line_start=10,
        line_end=10,
        rule_id="python.lang.security.audit.dangerous-system-call",
        message="Found dangerous system call.",
        raw_evidence="os.system(user_input)",
        cwe="CWE-78: OS Command Injection",
    )


def test_empty_findings(tmp_path):
    payload = {"results": []}

    with patch("sentinelai.scanners.semgrep.subprocess.run", return_value=_completed(stdout=json.dumps(payload))):
        findings = SemgrepScanner().scan(_context(tmp_path))

    assert findings == []


def test_missing_results_key_defaults_to_empty(tmp_path):
    payload = {"errors": []}

    with patch("sentinelai.scanners.semgrep.subprocess.run", return_value=_completed(stdout=json.dumps(payload))):
        findings = SemgrepScanner().scan(_context(tmp_path))

    assert findings == []


@pytest.mark.parametrize(
    "semgrep_severity,expected",
    [
        ("ERROR", Severity.HIGH),
        ("WARNING", Severity.MEDIUM),
        ("INFO", Severity.LOW),
        ("error", Severity.HIGH),
        ("nonsense", Severity.LOW),
        (None, Severity.LOW),
    ],
)
def test_severity_mapping(tmp_path, semgrep_severity, expected):
    result = _semgrep_result()
    if semgrep_severity is None:
        del result["extra"]["severity"]
    else:
        result["extra"]["severity"] = semgrep_severity
    payload = {"results": [result]}

    with patch("sentinelai.scanners.semgrep.subprocess.run", return_value=_completed(stdout=json.dumps(payload))):
        findings = SemgrepScanner().scan(_context(tmp_path))

    assert findings[0].severity == expected


def test_missing_executable_raises_scanner_execution_error(tmp_path):
    with patch("sentinelai.scanners.semgrep.subprocess.run", side_effect=FileNotFoundError("no such file")):
        with pytest.raises(ScannerExecutionError, match="semgrep"):
            SemgrepScanner().scan(_context(tmp_path))


def test_non_zero_exit_status_raises_scanner_execution_error(tmp_path):
    completed = _completed(stdout="", stderr="invalid config", returncode=2)

    with patch("sentinelai.scanners.semgrep.subprocess.run", return_value=completed):
        with pytest.raises(ScannerExecutionError, match="2"):
            SemgrepScanner().scan(_context(tmp_path))


def test_invalid_json_raises_scanner_execution_error(tmp_path):
    with patch("sentinelai.scanners.semgrep.subprocess.run", return_value=_completed(stdout="not json{{{")):
        with pytest.raises(ScannerExecutionError):
            SemgrepScanner().scan(_context(tmp_path))


def test_unexpected_subprocess_failure_propagates_as_scanner_execution_error_with_cause(tmp_path):
    original = OSError("permission denied")

    with patch("sentinelai.scanners.semgrep.subprocess.run", side_effect=original):
        with pytest.raises(ScannerExecutionError) as exc_info:
            SemgrepScanner().scan(_context(tmp_path))

    assert exc_info.value.__cause__ is original


def test_missing_optional_json_fields_use_sensible_defaults(tmp_path):
    minimal_result = {"check_id": "some.rule.id", "path": "main.py"}
    payload = {"results": [minimal_result]}

    with patch("sentinelai.scanners.semgrep.subprocess.run", return_value=_completed(stdout=json.dumps(payload))):
        findings = SemgrepScanner().scan(_context(tmp_path))

    assert len(findings) == 1
    finding = findings[0]
    assert finding.finding_id == "semgrep-0"
    assert finding.scanner == "semgrep"
    assert finding.category == "some.rule.id"
    assert finding.severity == Severity.LOW
    assert finding.file == "main.py"
    assert finding.line_start is None
    assert finding.line_end is None
    assert finding.rule_id == "some.rule.id"
    assert finding.message == ""
    assert finding.raw_evidence is None
    assert finding.cwe is None


def test_end_line_before_start_line_is_clamped_to_start(tmp_path):
    result = _semgrep_result(start={"line": 10, "col": 1}, end={"line": 5, "col": 1})
    payload = {"results": [result]}

    with patch("sentinelai.scanners.semgrep.subprocess.run", return_value=_completed(stdout=json.dumps(payload))):
        findings = SemgrepScanner().scan(_context(tmp_path))

    assert findings[0].line_start == 10
    assert findings[0].line_end == 10


def test_cwe_as_plain_string_is_preserved(tmp_path):
    result = _semgrep_result()
    result["extra"]["metadata"]["cwe"] = "CWE-89: SQL Injection"
    payload = {"results": [result]}

    with patch("sentinelai.scanners.semgrep.subprocess.run", return_value=_completed(stdout=json.dumps(payload))):
        findings = SemgrepScanner().scan(_context(tmp_path))

    assert findings[0].cwe == "CWE-89: SQL Injection"


def test_positional_finding_id_is_used_when_fingerprint_is_absent(tmp_path):
    first = _semgrep_result(path="a.py")
    del first["extra"]["fingerprint"]
    second = _semgrep_result(path="b.py")
    del second["extra"]["fingerprint"]
    payload = {"results": [first, second]}

    with patch("sentinelai.scanners.semgrep.subprocess.run", return_value=_completed(stdout=json.dumps(payload))):
        findings = SemgrepScanner().scan(_context(tmp_path))

    assert [f.finding_id for f in findings] == ["semgrep-0", "semgrep-1"]


def test_deterministic_output_across_calls(tmp_path):
    payload = {"results": [_semgrep_result(path="a.py"), _semgrep_result(path="b.py")]}

    with patch("sentinelai.scanners.semgrep.subprocess.run", return_value=_completed(stdout=json.dumps(payload))):
        first = SemgrepScanner().scan(_context(tmp_path))
        second = SemgrepScanner().scan(_context(tmp_path))

    assert first == second


def test_scanner_returns_only_scanner_finding_objects(tmp_path):
    payload = {"results": [_semgrep_result(path="a.py"), _semgrep_result(path="b.py")]}

    with patch("sentinelai.scanners.semgrep.subprocess.run", return_value=_completed(stdout=json.dumps(payload))):
        findings = SemgrepScanner().scan(_context(tmp_path))

    assert isinstance(findings, list)
    assert all(isinstance(finding, ScannerFinding) for finding in findings)


def test_config_flag_is_only_passed_when_provided(tmp_path):
    payload = {"results": []}

    with patch(
        "sentinelai.scanners.semgrep.subprocess.run", return_value=_completed(stdout=json.dumps(payload))
    ) as mock_run:
        SemgrepScanner().scan(_context(tmp_path))
        command_without_config = mock_run.call_args[0][0]

    with patch(
        "sentinelai.scanners.semgrep.subprocess.run", return_value=_completed(stdout=json.dumps(payload))
    ) as mock_run:
        SemgrepScanner(config="auto").scan(_context(tmp_path))
        command_with_config = mock_run.call_args[0][0]

    assert "--config" not in command_without_config
    assert "--config" in command_with_config
    assert "auto" in command_with_config


def test_command_requests_json_output_and_targets_repository_path(tmp_path):
    payload = {"results": []}

    with patch(
        "sentinelai.scanners.semgrep.subprocess.run", return_value=_completed(stdout=json.dumps(payload))
    ) as mock_run:
        context = _context(tmp_path)
        SemgrepScanner(executable="my-semgrep").scan(context)
        command = mock_run.call_args[0][0]

    assert command[0] == "my-semgrep"
    assert "--json" in command
    assert str(context.repository.absolute_path) in command
