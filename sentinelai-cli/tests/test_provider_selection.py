"""
Verifies the real, unpatched _get_provider()/LiveFindingsProvider path -
including through the actual CLI - as opposed to test_cli.py, whose
autouse fixture deliberately pins every test in that file to
MockFindingsProvider so CLI rendering/filtering/formatting behavior stays
testable without real scanner tools.

Real scanner executables are never invoked here either: the CLI-level
tests below inject a ScannerRegistry of dummy Scanner subclasses into a
real LiveFindingsProvider, then drive it through the real Typer app -
composing the actual production seam (_get_provider -> LiveFindingsProvider
-> backend -> ScannerOrchestrator -> rendering) that no other test
currently exercises together.
"""
import json
from pathlib import Path

import git as gitpython
from typer.testing import CliRunner

from sentinelai.contracts import ScannerFinding, Severity
from sentinelai.core import ExitCode
from sentinelai.main import _get_provider, app
from sentinelai.providers import LiveFindingsProvider
from sentinelai.scanners.base import Scanner
from sentinelai.scanners.exceptions import ScannerExecutionError
from sentinelai.scanners.registry import ScannerRegistry

runner = CliRunner()


class _RecordingScanner(Scanner):
    """A dummy scanner that returns fixed findings, or raises - no subprocess involved."""

    def __init__(self, findings=(), error=None):
        self._findings = list(findings)
        self._error = error

    def scan(self, context):
        if self._error is not None:
            raise self._error
        return list(self._findings)


def _registry(*scanners) -> ScannerRegistry:
    registry = ScannerRegistry()
    for index, scanner in enumerate(scanners):
        registry.register(f"dummy-{index}", scanner)
    return registry


def _init_git_repo_with_content(path: Path) -> str:
    repo = gitpython.Repo.init(path)
    with repo.config_writer() as cw:
        cw.set_value("user", "email", "test@example.com")
        cw.set_value("user", "name", "Test")
    (path / "main.py").write_text("print(1)\n", encoding="utf-8")
    (path / "app.js").write_text("console.log(1);\n", encoding="utf-8")
    repo.index.add(["main.py", "app.js"])
    return repo.index.commit("initial commit").hexsha


def test_get_provider_returns_live_findings_provider():
    assert isinstance(_get_provider(), LiveFindingsProvider)


def test_cli_scan_end_to_end_through_live_provider(monkeypatch, tmp_path):
    """The full production seam, composed for the first time: CLI -> _get_provider()
    -> LiveFindingsProvider -> real backend (real git repo, real language detection)
    -> ScannerOrchestrator -> injected fake scanner -> rendered JSON output."""
    commit_sha = _init_git_repo_with_content(tmp_path)
    finding = ScannerFinding(
        finding_id="SENT-LIVE-001",
        scanner="dummy",
        category="test-category",
        severity=Severity.HIGH,
        rule_id="dummy.rule",
        message="live provider integration finding",
    )
    registry = _registry(_RecordingScanner(findings=[finding]))
    monkeypatch.setattr("sentinelai.main._get_provider", lambda: LiveFindingsProvider(registry=registry))

    result = runner.invoke(app, ["scan", str(tmp_path), "--format", "json", "--quick"])

    assert result.exit_code == 0
    data = json.loads(result.output)
    assert [f["finding_id"] for f in data["findings"]["scanner"]] == ["SENT-LIVE-001"]
    assert data["repository"]["path"] == str(tmp_path)
    assert data["repository"]["commit_hash"] == commit_sha
    assert set(data["repository"]["languages"]) == {"Python", "JavaScript"}
    assert data["scan"]["mode"] == "quick"


def test_cli_scan_reports_a_real_chained_provider_error_cleanly(monkeypatch, tmp_path):
    """A real ScannerExecutionError, chained into ProviderError by LiveFindingsProvider
    itself (not a bare RuntimeError stand-in), reaching the CLI's generic handler."""
    registry = _registry(_RecordingScanner(error=ScannerExecutionError("semgrep exited with status 2: boom")))
    monkeypatch.setattr("sentinelai.main._get_provider", lambda: LiveFindingsProvider(registry=registry))

    result = runner.invoke(app, ["scan", str(tmp_path)])

    assert result.exit_code == ExitCode.PROVIDER_ERROR
    # Whitespace-flattened for this phrase only: Rich wraps stderr to the
    # console width, and whether "semgrep exited with status 2" straddles a
    # line break depends on the length of tmp_path - which differs between a
    # macOS /private/var/folders/... path and a CI /tmp/pytest-of-runner/...
    # one. The message content is the contract; its wrap points are not.
    assert "semgrep exited with status 2" in " ".join(result.stderr.split())
    assert "Traceback" not in result.output


def test_cli_scan_debug_flag_shows_traceback_for_a_real_provider_error(monkeypatch, tmp_path):
    registry = _registry(_RecordingScanner(error=ScannerExecutionError("boom")))
    monkeypatch.setattr("sentinelai.main._get_provider", lambda: LiveFindingsProvider(registry=registry))

    result = runner.invoke(app, ["--debug", "scan", str(tmp_path)])

    assert result.exit_code == ExitCode.PROVIDER_ERROR
    assert "Traceback" in result.stderr
