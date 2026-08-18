"""Tests for sentinelai.providers.live_provider - the real FindingsProvider implementation.

Real scanner executables (semgrep/bandit/gitleaks) are never invoked here: tests inject a
ScannerRegistry populated with dummy Scanner subclasses instead of patching subprocess.run.
Backend functions (load_repository, build_repository_context) run for real against tmp_path -
they're already covered by their own test suites and involve no external tools.
"""
from pathlib import Path

import git as gitpython
import pytest

from sentinelai.contracts import ScanMode, ScanResult, ScannerFinding, Severity
from sentinelai.core.errors import ProviderError
from sentinelai.providers import FindingsProvider
from sentinelai.providers.live_provider import LiveFindingsProvider
from sentinelai.scanners.base import Scanner
from sentinelai.scanners.exceptions import ScannerExecutionError
from sentinelai.scanners.registry import ScannerRegistry


class _RecordingScanner(Scanner):
    """A dummy scanner that logs its own invocation and returns fixed findings, or raises."""

    def __init__(self, name, findings=(), call_log=None, error=None):
        self._name = name
        self._findings = list(findings)
        self._call_log = call_log if call_log is not None else []
        self._error = error

    def scan(self, context):
        self._call_log.append(self._name)
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


def _write(path: Path, content: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _registry(*scanners) -> ScannerRegistry:
    registry = ScannerRegistry()
    for index, scanner in enumerate(scanners):
        registry.register(f"dummy-{index}", scanner)
    return registry


def _init_git_repo(path: Path) -> gitpython.Repo:
    repo = gitpython.Repo.init(path)
    with repo.config_writer() as cw:
        cw.set_value("user", "email", "test@example.com")
        cw.set_value("user", "name", "Test")
    return repo


def _commit(repo: gitpython.Repo, filename: str = "file.txt") -> str:
    (Path(repo.working_tree_dir) / filename).write_text("content")
    repo.index.add([filename])
    return repo.index.commit("commit").hexsha


def test_is_a_findings_provider(tmp_path):
    provider = LiveFindingsProvider(registry=_registry())

    assert isinstance(provider, FindingsProvider)


def test_returns_scan_result(tmp_path):
    provider = LiveFindingsProvider(registry=_registry())

    result = provider.get_scan_result(str(tmp_path))

    assert isinstance(result, ScanResult)


def test_populates_scanner_findings_from_all_registered_scanners(tmp_path):
    finding_a = _finding("SENT-001", "alpha")
    finding_b = _finding("SENT-002", "beta")
    registry = _registry(
        _RecordingScanner("alpha", findings=[finding_a]),
        _RecordingScanner("beta", findings=[finding_b]),
    )
    provider = LiveFindingsProvider(registry=registry)

    result = provider.get_scan_result(str(tmp_path))

    assert result.scanner_findings == [finding_a, finding_b]


def test_leaves_ai_findings_empty(tmp_path):
    provider = LiveFindingsProvider(registry=_registry())

    result = provider.get_scan_result(str(tmp_path))

    assert result.ai_findings == []


def test_repository_info_name_and_path_use_raw_argument(tmp_path):
    provider = LiveFindingsProvider(registry=_registry())

    result = provider.get_scan_result(str(tmp_path))

    assert result.repository.name == tmp_path.name
    assert result.repository.path == str(tmp_path)


def test_repository_info_commit_hash_and_branch_for_git_repository(tmp_path):
    repo = _init_git_repo(tmp_path)
    sha = _commit(repo)
    provider = LiveFindingsProvider(registry=_registry())

    result = provider.get_scan_result(str(tmp_path))

    assert result.repository.commit_hash == sha
    assert result.repository.branch == repo.active_branch.name


def test_repository_info_commit_hash_and_branch_are_none_for_non_git_repo(tmp_path):
    provider = LiveFindingsProvider(registry=_registry())

    result = provider.get_scan_result(str(tmp_path))

    assert result.repository.commit_hash is None
    assert result.repository.branch is None


def test_repository_info_languages_reflect_detected_languages(tmp_path):
    _write(tmp_path / "main.py")
    _write(tmp_path / "a.py")
    _write(tmp_path / "app.js")
    provider = LiveFindingsProvider(registry=_registry())

    result = provider.get_scan_result(str(tmp_path))

    assert set(result.repository.languages) == {"Python", "JavaScript"}


def test_scan_metadata_records_requested_mode(tmp_path):
    provider = LiveFindingsProvider(registry=_registry())

    result = provider.get_scan_result(str(tmp_path), mode=ScanMode.FULL)

    assert result.metadata.mode == ScanMode.FULL


def test_scan_metadata_timestamp_is_set(tmp_path):
    provider = LiveFindingsProvider(registry=_registry())

    result = provider.get_scan_result(str(tmp_path))

    assert result.metadata.timestamp is not None


def test_duration_seconds_is_left_unset(tmp_path):
    # main.py fills this in after get_scan_result() returns - matches mock_provider.py.
    provider = LiveFindingsProvider(registry=_registry())

    result = provider.get_scan_result(str(tmp_path))

    assert result.metadata.duration_seconds is None


def test_all_modes_currently_select_all_registered_scanners(tmp_path):
    call_log = []
    registry = _registry(
        _RecordingScanner("alpha", call_log=call_log),
        _RecordingScanner("beta", call_log=call_log),
    )

    for mode in (ScanMode.QUICK, ScanMode.STANDARD, ScanMode.FULL):
        call_log.clear()
        provider = LiveFindingsProvider(registry=registry)
        provider.get_scan_result(str(tmp_path), mode=mode)
        assert call_log == ["alpha", "beta"]


def test_nonexistent_repository_path_raises_provider_error(tmp_path):
    provider = LiveFindingsProvider(registry=_registry())

    with pytest.raises(ProviderError):
        provider.get_scan_result(str(tmp_path / "does-not-exist"))


def test_repository_path_that_is_a_file_raises_provider_error(tmp_path):
    file_path = tmp_path / "not_a_directory.txt"
    _write(file_path)
    provider = LiveFindingsProvider(registry=_registry())

    with pytest.raises(ProviderError):
        provider.get_scan_result(str(file_path))


def test_scanner_failure_raises_provider_error_chained_from_original(tmp_path):
    original = ScannerExecutionError("semgrep exited with status 2")
    registry = _registry(_RecordingScanner("alpha", error=original))
    provider = LiveFindingsProvider(registry=registry)

    with pytest.raises(ProviderError) as exc_info:
        provider.get_scan_result(str(tmp_path))

    assert exc_info.value.__cause__ is original


def test_scanner_failure_does_not_silently_skip_remaining_scanners(tmp_path):
    call_log = []
    failing = _RecordingScanner("alpha", call_log=call_log, error=ScannerExecutionError("boom"))
    never_runs = _RecordingScanner("beta", call_log=call_log)
    registry = _registry(failing, never_runs)
    provider = LiveFindingsProvider(registry=registry)

    with pytest.raises(ProviderError):
        provider.get_scan_result(str(tmp_path))

    assert call_log == ["alpha"]


def test_unexpected_backend_exception_is_translated_to_provider_error(tmp_path):
    _write(tmp_path / "pyproject.toml", "[project\ndependencies = [")  # malformed TOML
    provider = LiveFindingsProvider(registry=_registry())

    with pytest.raises(ProviderError) as exc_info:
        provider.get_scan_result(str(tmp_path))

    assert exc_info.value.__cause__ is not None


def test_scan_result_is_pydantic_round_trip_safe(tmp_path):
    finding = _finding("SENT-001", "alpha")
    registry = _registry(_RecordingScanner("alpha", findings=[finding]))
    provider = LiveFindingsProvider(registry=registry)

    result = provider.get_scan_result(str(tmp_path))
    restored = ScanResult.model_validate_json(result.model_dump_json())

    assert restored == result


def test_default_constructor_wires_up_real_scanners_without_invoking_them(monkeypatch, tmp_path):
    import subprocess

    calls = []

    def _fake_run(command, **kwargs):
        executable = command[0]
        calls.append(executable)
        # semgrep/bandit expect {"results": [...]}; gitleaks expects a bare [].
        stdout = "[]" if executable == "gitleaks" else '{"results": []}'
        return subprocess.CompletedProcess(args=command, returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr("sentinelai.scanners.semgrep.subprocess.run", _fake_run)
    monkeypatch.setattr("sentinelai.scanners.bandit.subprocess.run", _fake_run)
    monkeypatch.setattr("sentinelai.scanners.gitleaks.subprocess.run", _fake_run)

    provider = LiveFindingsProvider()
    result = provider.get_scan_result(str(tmp_path))

    assert result.scanner_findings == []
    assert set(calls) == {"semgrep", "bandit", "gitleaks"}
