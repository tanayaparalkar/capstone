"""Tests for sentinelai.scanners.orchestrator - the scanner orchestrator."""
import pytest

from sentinelai.backend.context_builder import build_repository_context
from sentinelai.backend.loader import load_repository
from sentinelai.contracts import ScannerFinding, Severity
from sentinelai.scanners.base import Scanner
from sentinelai.scanners.exceptions import ScannerExecutionError
from sentinelai.scanners.orchestrator import ScannerOrchestrator


class _RecordingScanner(Scanner):
    """A dummy scanner that logs its own invocation and returns fixed findings, or raises."""

    def __init__(self, name, findings=(), call_log=None, error=None):
        self._name = name
        self._findings = list(findings)
        self._call_log = call_log if call_log is not None else []
        self._error = error
        self.received_context = None

    def scan(self, context):
        self._call_log.append(self._name)
        self.received_context = context
        if self._error is not None:
            raise self._error
        return list(self._findings)


def _finding(finding_id: str, scanner: str) -> ScannerFinding:
    return ScannerFinding(
        finding_id=finding_id,
        scanner=scanner,
        category="test-category",
        severity=Severity.LOW,
        rule_id="dummy.rule",
        message="test finding",
    )


def _context(tmp_path):
    return build_repository_context(load_repository(str(tmp_path)))


def test_empty_collection_returns_empty_list(tmp_path):
    orchestrator = ScannerOrchestrator()

    result = orchestrator.run(_context(tmp_path), [])

    assert result == []


def test_single_scanner_returns_its_findings(tmp_path):
    finding = _finding("SENT-001", "alpha")
    scanner = _RecordingScanner("alpha", findings=[finding])
    orchestrator = ScannerOrchestrator()

    result = orchestrator.run(_context(tmp_path), [scanner])

    assert result == [finding]


def test_multiple_scanners_are_concatenated(tmp_path):
    finding_a = _finding("SENT-001", "alpha")
    finding_b = _finding("SENT-002", "beta")
    scanners = [
        _RecordingScanner("alpha", findings=[finding_a]),
        _RecordingScanner("beta", findings=[finding_b]),
    ]
    orchestrator = ScannerOrchestrator()

    result = orchestrator.run(_context(tmp_path), scanners)

    assert result == [finding_a, finding_b]


def test_each_scanner_is_invoked_exactly_once(tmp_path):
    call_log = []
    scanners = [
        _RecordingScanner("alpha", call_log=call_log),
        _RecordingScanner("beta", call_log=call_log),
        _RecordingScanner("gamma", call_log=call_log),
    ]
    orchestrator = ScannerOrchestrator()

    orchestrator.run(_context(tmp_path), scanners)

    assert call_log == ["alpha", "beta", "gamma"]


def test_scanners_run_in_the_order_given_not_reordered(tmp_path):
    # The orchestrator does not sort or otherwise reorder its input - ordering is
    # entirely the caller's decision. Passing a deliberately non-alphabetical order
    # here proves that, rather than coincidentally matching an alphabetical default.
    call_log = []
    scanners = [
        _RecordingScanner("zeta", call_log=call_log),
        _RecordingScanner("alpha", call_log=call_log),
        _RecordingScanner("mu", call_log=call_log),
    ]
    orchestrator = ScannerOrchestrator()

    orchestrator.run(_context(tmp_path), scanners)

    assert call_log == ["zeta", "alpha", "mu"]


def test_finding_order_matches_given_scanner_order(tmp_path):
    finding_zeta = _finding("SENT-Z", "zeta")
    finding_alpha = _finding("SENT-A", "alpha")
    scanners = [
        _RecordingScanner("zeta", findings=[finding_zeta]),
        _RecordingScanner("alpha", findings=[finding_alpha]),
    ]
    orchestrator = ScannerOrchestrator()

    result = orchestrator.run(_context(tmp_path), scanners)

    assert result == [finding_zeta, finding_alpha]


def test_context_is_passed_to_each_scanner(tmp_path):
    context = _context(tmp_path)
    scanner = _RecordingScanner("alpha")
    orchestrator = ScannerOrchestrator()

    orchestrator.run(context, [scanner])

    assert scanner.received_context is context


def test_scanner_execution_error_propagates_unchanged(tmp_path):
    original = ScannerExecutionError("semgrep exited with status 2")
    scanner = _RecordingScanner("alpha", error=original)
    orchestrator = ScannerOrchestrator()

    with pytest.raises(ScannerExecutionError) as exc_info:
        orchestrator.run(_context(tmp_path), [scanner])

    assert exc_info.value is original


def test_unexpected_exception_propagates_unchanged():
    class _SentinelBug(Exception):
        pass

    original = _SentinelBug("something unrelated to scanning broke")
    scanner = _RecordingScanner("alpha", error=original)
    orchestrator = ScannerOrchestrator()

    with pytest.raises(_SentinelBug) as exc_info:
        orchestrator.run(None, [scanner])

    assert exc_info.value is original


def test_failure_stops_scanners_later_in_given_order(tmp_path):
    call_log = []
    failing = _RecordingScanner("alpha", call_log=call_log, error=ScannerExecutionError("boom"))
    never_runs = _RecordingScanner("beta", call_log=call_log)
    scanners = [failing, never_runs]
    orchestrator = ScannerOrchestrator()

    with pytest.raises(ScannerExecutionError):
        orchestrator.run(_context(tmp_path), scanners)

    assert call_log == ["alpha"]


def test_returns_list_of_scanner_finding_only(tmp_path):
    scanners = [
        _RecordingScanner("alpha", findings=[_finding("SENT-001", "alpha")]),
        _RecordingScanner("beta", findings=[_finding("SENT-002", "beta")]),
    ]
    orchestrator = ScannerOrchestrator()

    result = orchestrator.run(_context(tmp_path), scanners)

    assert isinstance(result, list)
    assert all(isinstance(finding, ScannerFinding) for finding in result)


def test_repeated_runs_are_independent(tmp_path):
    finding = _finding("SENT-001", "alpha")
    orchestrator = ScannerOrchestrator()
    context = _context(tmp_path)

    first = orchestrator.run(context, [_RecordingScanner("alpha", findings=[finding])])
    second = orchestrator.run(context, [_RecordingScanner("alpha", findings=[finding])])

    assert first == second == [finding]
