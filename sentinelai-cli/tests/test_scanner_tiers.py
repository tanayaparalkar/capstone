"""
Scanner tiers: which scanners run, and the guarantee that adding the
extended set changed nothing about the default one.

The backward-compatibility tests here carry more weight than the
feature tests. Phase 1's frozen baseline - 17 raw findings, 13 correlated
issues, 4 multi-scanner groups - is measured against the core three
scanners, and it stays meaningful only if a default `sentinelai scan`
keeps running exactly those three. So every way a default scan could
silently acquire a dependency scanner (a new default, --full implying it,
a registry listing everything) has an explicit test proving it does not.
"""
import subprocess
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from sentinelai.contracts import ScanMode, ScannerFinding, ScannerTier, Severity
from sentinelai.main import app
from sentinelai.providers.live_provider import LiveFindingsProvider, _default_registry
from sentinelai.providers.mock_provider import MockFindingsProvider
from sentinelai.scanners.base import Scanner
from sentinelai.scanners.exceptions import ScannerExecutionError
from sentinelai.scanners.registry import ScannerRegistry

runner = CliRunner()

CORE_SCANNERS = ["bandit", "gitleaks", "semgrep"]
EXTENDED_ONLY = ["osv-scanner", "trivy"]


@pytest.fixture(autouse=True)
def _use_mock_provider(monkeypatch):
    """Pin the CLI to MockFindingsProvider, matching tests/test_cli.py's fixture.

    These tests assert which *tier* the CLI requests, which is decided before
    any provider runs - so they must not invoke real scanner executables. The
    provider-level tests below construct LiveFindingsProvider directly and are
    unaffected by this.
    """
    import sentinelai.main as main_module

    monkeypatch.setattr(main_module, "_get_provider", lambda: MockFindingsProvider())


class _RecordingScanner(Scanner):
    def __init__(self, name):
        self.name = name
        self.calls = 0

    def scan(self, context):
        self.calls += 1
        return []


class _FailingScanner(Scanner):
    def scan(self, context):
        raise ScannerExecutionError("simulated trivy failure")


def _registry(**scanners):
    registry = ScannerRegistry()
    for name, (scanner, tier) in scanners.items():
        registry.register(name, scanner, tier)
    return registry


# --- the default registry -------------------------------------------------------------------------------------------


def test_default_registry_holds_all_five_scanners():
    names = [registration.name for registration in _default_registry().list_scanners()]

    assert names == sorted(CORE_SCANNERS + EXTENDED_ONLY)


def test_core_tier_is_exactly_the_three_code_scanners():
    names = [r.name for r in _default_registry().list_scanners(ScannerTier.CORE)]

    assert names == CORE_SCANNERS


def test_extended_tier_is_a_superset_never_a_replacement():
    """Adding dependency scanning must not take code scanning away."""
    core = {r.name for r in _default_registry().list_scanners(ScannerTier.CORE)}
    extended = {r.name for r in _default_registry().list_scanners(ScannerTier.EXTENDED)}

    assert core < extended
    assert extended - core == set(EXTENDED_ONLY)


def test_dependency_scanners_are_registered_under_their_wrapper_names():
    registry = _default_registry()

    assert type(registry.get("trivy")).__name__ == "TrivyScanner"
    assert type(registry.get("osv-scanner")).__name__ == "OSVScanner"


# --- registry behavior ----------------------------------------------------------------------------------------------


def test_register_defaults_to_core_so_existing_call_sites_are_unchanged():
    registry = ScannerRegistry()
    registry.register("legacy", _RecordingScanner("legacy"))  # two-argument form, as before

    assert [r.name for r in registry.list_scanners(ScannerTier.CORE)] == ["legacy"]
    assert registry.list_scanners()[0].tier is ScannerTier.CORE


def test_list_scanners_without_a_tier_still_returns_everything():
    registry = _registry(
        a=(_RecordingScanner("a"), ScannerTier.CORE), z=(_RecordingScanner("z"), ScannerTier.EXTENDED)
    )

    assert [r.name for r in registry.list_scanners()] == ["a", "z"]


def test_listing_is_sorted_by_name_not_registration_order():
    registry = _registry(
        z=(_RecordingScanner("z"), ScannerTier.CORE), a=(_RecordingScanner("a"), ScannerTier.CORE)
    )

    assert [r.name for r in registry.list_scanners(ScannerTier.CORE)] == ["a", "z"]


# --- provider selection ---------------------------------------------------------------------------------------------


def _provider_with_recorders():
    core, extended = _RecordingScanner("core"), _RecordingScanner("extended")
    registry = _registry(core=(core, ScannerTier.CORE), extended=(extended, ScannerTier.EXTENDED))
    return LiveFindingsProvider(registry=registry), core, extended


def test_default_scan_runs_core_scanners_only(tmp_path):
    provider, core, extended = _provider_with_recorders()

    provider.get_scan_result(str(tmp_path))

    assert (core.calls, extended.calls) == (1, 0)


def test_extended_tier_runs_both(tmp_path):
    provider, core, extended = _provider_with_recorders()

    provider.get_scan_result(str(tmp_path), tier=ScannerTier.EXTENDED)

    assert (core.calls, extended.calls) == (1, 1)


@pytest.mark.parametrize("mode", list(ScanMode))
def test_no_scan_mode_implies_the_extended_tier(tmp_path, mode):
    """--full is a depth control; it must not silently acquire dependency scanning."""
    provider, core, extended = _provider_with_recorders()

    provider.get_scan_result(str(tmp_path), mode=mode)

    assert (core.calls, extended.calls) == (1, 0)


def test_tier_is_recorded_in_scan_metadata(tmp_path):
    provider, _, _ = _provider_with_recorders()

    default = provider.get_scan_result(str(tmp_path))
    extended = provider.get_scan_result(str(tmp_path), tier=ScannerTier.EXTENDED)

    assert default.metadata.scanner_tier is ScannerTier.CORE
    assert extended.metadata.scanner_tier is ScannerTier.EXTENDED


def test_extended_scanner_failure_still_fails_closed(tmp_path):
    """A dependency scanner is not best-effort: its failure aborts the scan like any other."""
    registry = _registry(
        core=(_RecordingScanner("core"), ScannerTier.CORE), trivy=(_FailingScanner(), ScannerTier.EXTENDED)
    )
    provider = LiveFindingsProvider(registry=registry)

    from sentinelai.core.errors import ProviderError

    with pytest.raises(ProviderError, match="simulated trivy failure"):
        provider.get_scan_result(str(tmp_path), tier=ScannerTier.EXTENDED)


def test_a_broken_extended_scanner_cannot_affect_a_default_scan(tmp_path):
    registry = _registry(
        core=(_RecordingScanner("core"), ScannerTier.CORE), trivy=(_FailingScanner(), ScannerTier.EXTENDED)
    )

    result = LiveFindingsProvider(registry=registry).get_scan_result(str(tmp_path))

    assert result.metadata.scanner_tier is ScannerTier.CORE


# --- the CLI contract -----------------------------------------------------------------------------------------------


def _spy(monkeypatch):
    calls = []
    original = MockFindingsProvider.get_scan_result

    def wrapper(self, repo_path, mode=ScanMode.STANDARD, tier=ScannerTier.CORE):
        calls.append((mode, tier))
        return original(self, repo_path, mode=mode, tier=tier)

    monkeypatch.setattr(MockFindingsProvider, "get_scan_result", wrapper)
    return calls


def test_scan_without_the_flag_requests_the_core_tier(monkeypatch):
    calls = _spy(monkeypatch)

    assert runner.invoke(app, ["scan", "."]).exit_code == 0
    assert calls[0] == (ScanMode.STANDARD, ScannerTier.CORE)


def test_extended_flag_requests_the_extended_tier(monkeypatch):
    calls = _spy(monkeypatch)

    assert runner.invoke(app, ["scan", ".", "--extended"]).exit_code == 0
    assert calls[0][1] is ScannerTier.EXTENDED


def test_full_alone_does_not_request_the_extended_tier(monkeypatch):
    calls = _spy(monkeypatch)

    assert runner.invoke(app, ["scan", ".", "--full"]).exit_code == 0
    assert calls[0] == (ScanMode.FULL, ScannerTier.CORE)


def test_extended_composes_with_mode_rather_than_replacing_it(monkeypatch):
    calls = _spy(monkeypatch)

    assert runner.invoke(app, ["scan", ".", "--quick", "--extended"]).exit_code == 0
    assert calls[0] == (ScanMode.QUICK, ScannerTier.EXTENDED)


def test_extended_is_documented_in_help():
    output = runner.invoke(app, ["scan", "--help"]).output

    assert "--extended" in output
    assert "Trivy" in output


def test_scanner_tier_appears_in_the_json_report(tmp_path):
    output = tmp_path / "report.json"
    result = runner.invoke(app, ["scan", ".", "--format", "json", "--output", str(output)])

    assert result.exit_code == 0
    import json

    assert json.loads(output.read_text())["scan"]["scanner_tier"] == "core"


# --- the tier is visible in every rendered format ---------------------------------------------------------------------


def _rendered(tier):
    """Render one scan result in every format, at the given tier."""
    from sentinelai.contracts import ScanMetadata, ScanResult
    from sentinelai.reporting import to_html, to_json, to_markdown, to_sarif
    from sentinelai.statistics import calculate_statistics
    import datetime

    result = ScanResult(
        repository={"name": "demo", "path": "/tmp/demo"},
        metadata=ScanMetadata(
            timestamp=datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc),
            mode=ScanMode.STANDARD,
            scanner_tier=tier,
        ),
    )
    stats = calculate_statistics(result)
    return {
        "json": to_json(result, stats),
        "markdown": to_markdown(result, stats),
        "html": to_html(result, stats),
        "sarif": to_sarif(result, stats),
    }


@pytest.mark.parametrize("tier", list(ScannerTier))
@pytest.mark.parametrize("fmt", ["json", "markdown", "html", "sarif"])
def test_every_format_states_which_scanner_set_produced_it(tier, fmt):
    """A reader of any report must be able to tell whether dependency scanners ran.

    Without this, an extended report and a core report of the same repository are
    indistinguishable to a human except by finding count - which is exactly the
    ambiguity the tier exists to remove.
    """
    assert tier.value in _rendered(tier)[fmt]


def test_the_tier_survives_a_json_round_trip_into_every_other_format(tmp_path):
    """scan -> JSON -> `sentinelai report` reload -> markdown/html/sarif."""
    from sentinelai.reporting import to_html, to_markdown, to_sarif
    from sentinelai.reporting.loader import load_scan_result
    from sentinelai.statistics import calculate_statistics

    saved = tmp_path / "report.json"
    saved.write_text(_rendered(ScannerTier.EXTENDED)["json"], encoding="utf-8")

    reloaded, _ = load_scan_result(saved)
    stats = calculate_statistics(reloaded)

    assert reloaded.metadata.scanner_tier is ScannerTier.EXTENDED
    for rendered in (to_markdown(reloaded, stats), to_html(reloaded, stats), to_sarif(reloaded, stats)):
        assert "extended" in rendered


def test_sarif_records_the_tier_in_run_properties():
    import json as json_module

    sarif = json_module.loads(_rendered(ScannerTier.EXTENDED)["sarif"])

    assert sarif["runs"][0]["properties"]["sentinelai"]["scannerTier"] == "extended"


# --- reports written before tiers existed -----------------------------------------------------------------------------


def test_a_report_without_scanner_tier_still_loads():
    """ScanMetadata defaults to CORE, so schema_version stays 1.0 and old reports remain readable."""
    from sentinelai.contracts import ScanMetadata

    metadata = ScanMetadata.model_validate({"timestamp": "2026-01-01T00:00:00Z", "mode": "standard"})

    assert metadata.scanner_tier is ScannerTier.CORE


# --- both wrappers reachable through the real registry ------------------------------------------------------------------


def _dispatching_run(commands):
    """One subprocess.run stand-in that answers each tool in its own output shape.

    A single patch rather than one per scanner module: every wrapper does a
    plain `import subprocess`, so `sentinelai.scanners.<tool>.subprocess` is
    the one shared module object and five separate patches would all target
    the same attribute, leaving whichever was applied last to answer for
    every tool.
    """
    empty_output = {
        "bandit": '{"results": []}',
        "semgrep": '{"results": []}',
        "gitleaks": "[]",
        "trivy": '{"Results": null}',
        "osv-scanner": '{"results": []}',
    }

    def run(command, **kwargs):
        commands.append(command)
        tool = command[0]
        if tool in _uninstalled:
            # Exactly what subprocess.run raises for a binary that is not on
            # PATH - the real missing-tool path, not a stand-in exception.
            raise FileNotFoundError(2, "No such file or directory", tool)
        return subprocess.CompletedProcess(
            args=command, returncode=0, stdout=empty_output[tool], stderr=""
        )

    return run


# Tools treated as absent from PATH by _dispatching_run, for the two
# missing-binary tests below. Empty by default so every other test sees a
# fully-installed machine.
_uninstalled: set = set()


@pytest.fixture
def without_dependency_scanners():
    """Simulate a machine where Trivy and OSV-Scanner are not installed."""
    _uninstalled.update(EXTENDED_ONLY)
    yield
    _uninstalled.clear()


def test_extended_scan_fails_clearly_when_a_dependency_scanner_is_not_installed(
    tmp_path, without_dependency_scanners
):
    """A missing binary must fail the scan, not silently return fewer findings.

    This exercises the real missing-tool path - subprocess.run raising
    FileNotFoundError, caught by each wrapper's OSError handler - rather than a
    stand-in scanner that raises ScannerExecutionError directly, because it is
    the wrapper's own error translation that has to produce an actionable
    message here.
    """
    from sentinelai.core.errors import ProviderError

    with patch("subprocess.run", side_effect=_dispatching_run([])):
        with pytest.raises(ProviderError) as excinfo:
            LiveFindingsProvider().get_scan_result(str(tmp_path), tier=ScannerTier.EXTENDED)

    message = str(excinfo.value)
    assert "osv-scanner" in message  # names the tool that is actually missing
    assert "Failed to run" in message


def test_core_scan_still_works_when_the_dependency_scanners_are_not_installed(
    tmp_path, without_dependency_scanners
):
    """The default scan must not acquire a new installation requirement."""
    commands = []

    with patch("subprocess.run", side_effect=_dispatching_run(commands)):
        result = LiveFindingsProvider().get_scan_result(str(tmp_path))

    assert [command[0] for command in commands] == CORE_SCANNERS
    assert result.metadata.scanner_tier is ScannerTier.CORE
    assert result.scanner_findings == []


def test_a_default_scan_never_shells_out_to_a_dependency_scanner(tmp_path):
    commands = []

    with patch("subprocess.run", side_effect=_dispatching_run(commands)):
        LiveFindingsProvider().get_scan_result(str(tmp_path))

    assert [command[0] for command in commands] == CORE_SCANNERS


def test_the_registered_dependency_scanners_invoke_their_real_commands(tmp_path):
    """End-to-end through the registry: the tier wiring reaches the actual wrappers."""
    commands = []

    with patch("subprocess.run", side_effect=_dispatching_run(commands)):
        result = LiveFindingsProvider().get_scan_result(str(tmp_path), tier=ScannerTier.EXTENDED)

    assert [command[0] for command in commands] == sorted(CORE_SCANNERS + EXTENDED_ONLY)
    by_tool = {command[0]: command for command in commands}
    assert by_tool["trivy"][:6] == ["trivy", "fs", "--scanners", "vuln", "--format", "json"]
    assert by_tool["osv-scanner"][:5] == ["osv-scanner", "scan", "source", "--format", "json"]
    assert result.scanner_findings == []


def test_dependency_findings_never_correlate_with_code_findings():
    """A dependency finding has no line, so it can never be pulled into a code finding's group."""
    from sentinelai.correlation import correlate_findings

    code = ScannerFinding(
        finding_id="bandit-0", scanner="bandit", category="blacklist", severity=Severity.HIGH,
        file="requirements.txt", line_start=1, line_end=1, rule_id="B001", message="m", cwe="CWE-20",
    )
    dependency = ScannerFinding(
        finding_id="trivy-requirements.txt:flask@0.12.2:CVE-2018-1000656", scanner="trivy",
        category="vulnerable-dependency", severity=Severity.HIGH, file="requirements.txt",
        rule_id="CVE-2018-1000656", message="m", cwe="CWE-20",
    )

    groups = correlate_findings([code, dependency])

    assert len(groups) == 2
    assert all(len(g.source_finding_ids) == 1 for g in groups)
