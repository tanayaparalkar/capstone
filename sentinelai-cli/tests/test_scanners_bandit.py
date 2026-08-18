"""Tests for sentinelai.scanners.bandit - the Bandit scanner wrapper. subprocess.run is always mocked; no real Bandit installation is ever invoked."""
import json
import subprocess
from unittest.mock import patch

import pytest

from sentinelai.backend.context_builder import build_repository_context
from sentinelai.backend.loader import load_repository
from sentinelai.contracts import ScannerFinding, Severity
from sentinelai.scanners.bandit import BanditScanner
from sentinelai.scanners.exceptions import ScannerExecutionError


def _context(tmp_path):
    return build_repository_context(load_repository(str(tmp_path)))


def _completed(stdout="", stderr="", returncode=0):
    return subprocess.CompletedProcess(args=["bandit"], returncode=returncode, stdout=stdout, stderr=stderr)


def _bandit_result(**overrides):
    result = {
        "code": "9  import subprocess\n10  subprocess.call(cmd, shell=True)\n",
        "filename": "app.py",
        "issue_confidence": "HIGH",
        "issue_cwe": {"id": 78, "link": "https://cwe.mitre.org/data/definitions/78.html"},
        "issue_severity": "MEDIUM",
        "issue_text": "subprocess call with shell=True identified.",
        "line_number": 10,
        "line_range": [9, 10],
        "test_id": "B602",
        "test_name": "subprocess_popen_with_shell_equals_true",
    }
    result.update(overrides)
    return result


def test_successful_json_parsing(tmp_path):
    payload = {"results": [_bandit_result()]}

    with patch("sentinelai.scanners.bandit.subprocess.run", return_value=_completed(stdout=json.dumps(payload))):
        findings = BanditScanner().scan(_context(tmp_path))

    assert len(findings) == 1
    finding = findings[0]
    assert finding == ScannerFinding(
        finding_id="bandit-0",
        scanner="bandit",
        category="subprocess_popen_with_shell_equals_true",
        severity=Severity.MEDIUM,
        file="app.py",
        line_start=9,
        line_end=10,
        rule_id="B602",
        message="subprocess call with shell=True identified. (confidence: HIGH)",
        raw_evidence="9  import subprocess\n10  subprocess.call(cmd, shell=True)\n",
        cwe="CWE-78",
    )


def test_empty_findings(tmp_path):
    payload = {"results": []}

    with patch("sentinelai.scanners.bandit.subprocess.run", return_value=_completed(stdout=json.dumps(payload))):
        findings = BanditScanner().scan(_context(tmp_path))

    assert findings == []


def test_missing_results_key_defaults_to_empty(tmp_path):
    payload = {"errors": []}

    with patch("sentinelai.scanners.bandit.subprocess.run", return_value=_completed(stdout=json.dumps(payload))):
        findings = BanditScanner().scan(_context(tmp_path))

    assert findings == []


@pytest.mark.parametrize(
    "bandit_severity,expected",
    [
        ("LOW", Severity.LOW),
        ("MEDIUM", Severity.MEDIUM),
        ("HIGH", Severity.HIGH),
        ("medium", Severity.MEDIUM),
        ("UNDEFINED", Severity.LOW),
        ("nonsense", Severity.LOW),
        (None, Severity.LOW),
    ],
)
def test_severity_mapping(tmp_path, bandit_severity, expected):
    result = _bandit_result()
    if bandit_severity is None:
        del result["issue_severity"]
    else:
        result["issue_severity"] = bandit_severity
    payload = {"results": [result]}

    with patch("sentinelai.scanners.bandit.subprocess.run", return_value=_completed(stdout=json.dumps(payload))):
        findings = BanditScanner().scan(_context(tmp_path))

    assert findings[0].severity == expected


def test_missing_executable_raises_scanner_execution_error(tmp_path):
    with patch("sentinelai.scanners.bandit.subprocess.run", side_effect=FileNotFoundError("no such file")):
        with pytest.raises(ScannerExecutionError, match="bandit"):
            BanditScanner().scan(_context(tmp_path))


def test_exit_code_one_with_findings_is_not_an_error(tmp_path):
    # Bandit's own convention: exit 1 means issues were found, not that bandit failed.
    payload = {"results": [_bandit_result()]}
    completed = _completed(stdout=json.dumps(payload), returncode=1)

    with patch("sentinelai.scanners.bandit.subprocess.run", return_value=completed):
        findings = BanditScanner().scan(_context(tmp_path))

    assert len(findings) == 1


def test_non_zero_exit_status_outside_zero_or_one_raises_scanner_execution_error(tmp_path):
    completed = _completed(stdout="", stderr="bandit could not process file", returncode=2)

    with patch("sentinelai.scanners.bandit.subprocess.run", return_value=completed):
        with pytest.raises(ScannerExecutionError, match="2"):
            BanditScanner().scan(_context(tmp_path))


def test_invalid_json_raises_scanner_execution_error(tmp_path):
    with patch("sentinelai.scanners.bandit.subprocess.run", return_value=_completed(stdout="not json{{{")):
        with pytest.raises(ScannerExecutionError):
            BanditScanner().scan(_context(tmp_path))


def test_unexpected_subprocess_failure_propagates_as_scanner_execution_error_with_cause(tmp_path):
    original = OSError("permission denied")

    with patch("sentinelai.scanners.bandit.subprocess.run", side_effect=original):
        with pytest.raises(ScannerExecutionError) as exc_info:
            BanditScanner().scan(_context(tmp_path))

    assert exc_info.value.__cause__ is original


def test_missing_optional_json_fields_use_sensible_defaults(tmp_path):
    minimal_result = {"test_id": "B101", "filename": "main.py"}
    payload = {"results": [minimal_result]}

    with patch("sentinelai.scanners.bandit.subprocess.run", return_value=_completed(stdout=json.dumps(payload))):
        findings = BanditScanner().scan(_context(tmp_path))

    assert len(findings) == 1
    finding = findings[0]
    assert finding.finding_id == "bandit-0"
    assert finding.scanner == "bandit"
    assert finding.category == "B101"
    assert finding.severity == Severity.LOW
    assert finding.file == "main.py"
    assert finding.line_start is None
    assert finding.line_end is None
    assert finding.rule_id == "B101"
    assert finding.message == ""
    assert finding.raw_evidence is None
    assert finding.cwe is None


def test_line_range_preferred_over_single_line_number(tmp_path):
    result = _bandit_result(line_number=10, line_range=[8, 9, 10])
    payload = {"results": [result]}

    with patch("sentinelai.scanners.bandit.subprocess.run", return_value=_completed(stdout=json.dumps(payload))):
        findings = BanditScanner().scan(_context(tmp_path))

    assert findings[0].line_start == 8
    assert findings[0].line_end == 10


def test_line_number_used_when_line_range_absent(tmp_path):
    result = _bandit_result()
    del result["line_range"]
    payload = {"results": [result]}

    with patch("sentinelai.scanners.bandit.subprocess.run", return_value=_completed(stdout=json.dumps(payload))):
        findings = BanditScanner().scan(_context(tmp_path))

    assert findings[0].line_start == 10
    assert findings[0].line_end == 10


def test_cwe_as_plain_int_is_preserved(tmp_path):
    result = _bandit_result(issue_cwe=89)
    payload = {"results": [result]}

    with patch("sentinelai.scanners.bandit.subprocess.run", return_value=_completed(stdout=json.dumps(payload))):
        findings = BanditScanner().scan(_context(tmp_path))

    assert findings[0].cwe == "CWE-89"


def test_confidence_is_omitted_from_message_when_absent(tmp_path):
    result = _bandit_result()
    del result["issue_confidence"]
    payload = {"results": [result]}

    with patch("sentinelai.scanners.bandit.subprocess.run", return_value=_completed(stdout=json.dumps(payload))):
        findings = BanditScanner().scan(_context(tmp_path))

    assert findings[0].message == "subprocess call with shell=True identified."


def test_deterministic_output_across_calls(tmp_path):
    payload = {"results": [_bandit_result(filename="a.py"), _bandit_result(filename="b.py")]}

    with patch("sentinelai.scanners.bandit.subprocess.run", return_value=_completed(stdout=json.dumps(payload))):
        first = BanditScanner().scan(_context(tmp_path))
        second = BanditScanner().scan(_context(tmp_path))

    assert first == second


def test_finding_ids_are_positional(tmp_path):
    payload = {"results": [_bandit_result(filename="a.py"), _bandit_result(filename="b.py")]}

    with patch("sentinelai.scanners.bandit.subprocess.run", return_value=_completed(stdout=json.dumps(payload))):
        findings = BanditScanner().scan(_context(tmp_path))

    assert [f.finding_id for f in findings] == ["bandit-0", "bandit-1"]


def test_scanner_returns_only_scanner_finding_objects(tmp_path):
    payload = {"results": [_bandit_result(filename="a.py"), _bandit_result(filename="b.py")]}

    with patch("sentinelai.scanners.bandit.subprocess.run", return_value=_completed(stdout=json.dumps(payload))):
        findings = BanditScanner().scan(_context(tmp_path))

    assert isinstance(findings, list)
    assert all(isinstance(finding, ScannerFinding) for finding in findings)


def test_command_construction_and_repository_path(tmp_path):
    payload = {"results": []}

    with patch(
        "sentinelai.scanners.bandit.subprocess.run", return_value=_completed(stdout=json.dumps(payload))
    ) as mock_run:
        context = _context(tmp_path)
        BanditScanner(executable="my-bandit").scan(context)
        command = mock_run.call_args[0][0]

    assert command[0] == "my-bandit"
    assert "-r" in command
    assert "-f" in command
    assert "json" in command
    assert str(context.repository.absolute_path) in command
