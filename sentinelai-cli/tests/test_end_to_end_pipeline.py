"""
Whole-pipeline integration: the stages joined together, not in isolation.

Every module in this project has unit tests, and several pairs of adjacent
stages are tested together. What these tests add is the joins that nothing
currently crosses - most importantly patch application, which is covered
thoroughly on the patching side and thoroughly on the reporting side by two
suites that never meet.

The composition under test is the production one:

    repository -> LiveFindingsProvider -> ScannerOrchestrator -> correlation
      -> statistics -> reporting -> reload -> re-render

Only two things are ever substituted. The scanner is a fixed-output stand-in,
matching the convention tests/test_provider_selection.py already established,
so the default suite needs no Semgrep/Bandit/GitLeaks on PATH; the real-scanner
equivalent is at the end of this file, benchmark-marked. And `enrich_findings`
is stubbed where a structured patch is needed, because producing one requires a
live model - everything downstream of it (validation, backup, application,
verification, rollback, reporting) runs for real against real files.

These assert invariants, not implementation details: what must remain true of
the data as it crosses each boundary. Determinism and scanner-failure
propagation are deliberately absent - both are already covered thoroughly
elsewhere, and repeating them here would add lines without adding confidence.
"""
import json
import shutil
from pathlib import Path

import git as gitpython
import pytest
from typer.testing import CliRunner

from sentinelai.contracts import (
    AIEnrichedFinding,
    ConfidenceLabel,
    ScanMode,
    ScannerFinding,
    Severity,
    StructuredPatch,
    VerificationStatus,
)
from sentinelai.core.exit_codes import ExitCode
from sentinelai.main import app
from sentinelai.providers.live_provider import LiveFindingsProvider
from sentinelai.reporting import to_html, to_json, to_markdown, to_sarif
from sentinelai.reporting.loader import load_scan_result
from sentinelai.scanners.base import Scanner
from sentinelai.scanners.registry import ScannerRegistry
from sentinelai.statistics import calculate_statistics

runner = CliRunner()

_SOURCE = "import yaml\n\ndef load_config(raw):\n    return yaml.load(raw)\n\ndef main():\n"
_PATCHED = "import yaml\n\ndef load_config(raw):\n    return yaml.safe_load(raw)\n\ndef main():\n"
_DIFF = (
    "--- a/parser.py\n+++ b/parser.py\n@@ -1,6 +1,6 @@\n"
    " import yaml\n \n def load_config(raw):\n"
    "-    return yaml.load(raw)\n+    return yaml.safe_load(raw)\n \n def main():\n"
)


class _FixedScanner(Scanner):
    """Fixed output, no subprocess - same convention as test_provider_selection.py."""

    def __init__(self, findings):
        self._findings = list(findings)
        self.calls = 0

    def scan(self, context):
        self.calls += 1
        return list(self._findings)


def _registry(scanner) -> ScannerRegistry:
    registry = ScannerRegistry()
    registry.register("dummy", scanner)
    return registry


def _finding(finding_id, scanner="semgrep", cwe="CWE-89", line=7, severity=Severity.HIGH):
    return ScannerFinding(
        finding_id=finding_id,
        scanner=scanner,
        category="sql-injection",
        severity=severity,
        file="app/db.py",
        line_start=line,
        rule_id=f"{scanner}.rule",
        message="User input concatenated into a query.",
        raw_evidence="query = f\"SELECT {x}\"",
        cwe=cwe,
    )


def _repo(tmp_path: Path) -> Path:
    repo = gitpython.Repo.init(tmp_path)
    with repo.config_writer() as cw:
        cw.set_value("user", "email", "t@e.st")
        cw.set_value("user", "name", "Test")
    (tmp_path / "main.py").write_text("print(1)\n", encoding="utf-8")
    repo.index.add(["main.py"])
    repo.index.commit("initial")
    return tmp_path


def _scan(tmp_path, findings):
    """Run the real provider over a real repository and correlate, as production does."""
    provider = LiveFindingsProvider(registry=_registry(_FixedScanner(findings)))
    return provider.get_scan_result(str(tmp_path), mode=ScanMode.STANDARD)


# --- repository -> ... -> reload -> re-render -----------------------------------------------------------------


def test_no_raw_finding_is_lost_between_the_scanner_and_a_reloaded_report(tmp_path):
    """The project's central correlation guarantee, asserted across the whole chain.

    Correlation is additive, so every raw finding a scanner produced must still
    be present, unmodified, after serialization and reload. Unit tests assert
    this at the correlation boundary; this asserts it survives four more stages.
    """
    _repo(tmp_path)
    findings = [_finding("SENT-001"), _finding("SENT-002", scanner="bandit"), _finding("SENT-003", line=90)]
    result = _scan(tmp_path, findings)

    saved = tmp_path / "report.json"
    saved.write_text(to_json(result, calculate_statistics(result)), encoding="utf-8")
    reloaded, _ = load_scan_result(saved)

    assert [f.finding_id for f in reloaded.scanner_findings] == ["SENT-001", "SENT-002", "SENT-003"]
    assert reloaded.scanner_findings == sorted(result.scanner_findings, key=lambda f: f.finding_id)


def test_statistics_are_identical_before_and_after_a_round_trip(tmp_path):
    """Statistics are recomputed on reload rather than trusted; they must still agree.

    If they could diverge, a saved report re-rendered later would contradict the
    scan that produced it.
    """
    _repo(tmp_path)
    result = _scan(tmp_path, [_finding("SENT-001"), _finding("SENT-002", scanner="bandit")])
    original = calculate_statistics(result)

    saved = tmp_path / "report.json"
    saved.write_text(to_json(result, original), encoding="utf-8")
    reloaded, reloaded_stats = load_scan_result(saved)

    assert calculate_statistics(reloaded) == reloaded_stats
    for field in ("total_findings", "correlated_findings", "multi_scanner_findings", "scanner_counts"):
        assert getattr(reloaded_stats, field) == getattr(original, field), field


def test_correlated_groups_only_ever_reference_findings_that_exist(tmp_path):
    """Provenance invariant: a group is an index over raw findings, never a source of new ones."""
    _repo(tmp_path)
    result = _scan(
        tmp_path, [_finding("SENT-001"), _finding("SENT-002", scanner="bandit"), _finding("SENT-003", line=90)]
    )

    known = {f.finding_id for f in result.scanner_findings}
    # Non-vacuity: two of the fixture findings share CWE, file and line, so they
    # must group. Without this the loop below would pass over an empty list and
    # assert nothing at all.
    assert result.correlated_findings, "the fixture must produce at least one group"
    assert any(g.is_multi_scanner for g in result.correlated_findings)

    for group in result.correlated_findings:
        assert set(group.source_finding_ids) <= known
        assert group.canonical_finding_id in group.source_finding_ids


def test_every_format_reports_the_same_finding_counts(tmp_path):
    """Four renderers, one record - a disagreement here would make the reports untrustworthy."""
    _repo(tmp_path)
    result = _scan(tmp_path, [_finding("SENT-001"), _finding("SENT-002", scanner="bandit")])
    stats = calculate_statistics(result)

    rendered = {
        "json": json.loads(to_json(result, stats)),
        "sarif": json.loads(to_sarif(result, stats)),
        "markdown": to_markdown(result, stats),
        "html": to_html(result, stats),
    }

    assert len(rendered["json"]["findings"]["scanner"]) == stats.total_findings
    assert len(rendered["sarif"]["runs"][0]["results"]) == stats.total_findings
    for finding in result.scanner_findings:
        assert finding.finding_id in rendered["markdown"]
        assert finding.finding_id in rendered["html"]


def test_reloading_a_report_never_invokes_a_scanner(tmp_path):
    """`sentinelai report` re-renders a saved record; re-scanning would be a different scan.

    A documented guarantee with no test until now. Asserted by counting calls on
    the scanner the report was produced with.
    """
    _repo(tmp_path)
    scanner = _FixedScanner([_finding("SENT-001")])
    provider = LiveFindingsProvider(registry=_registry(scanner))
    result = provider.get_scan_result(str(tmp_path), mode=ScanMode.STANDARD)
    assert scanner.calls == 1

    saved = tmp_path / "report.json"
    saved.write_text(to_json(result, calculate_statistics(result)), encoding="utf-8")

    for fmt in ("json", "markdown", "html", "sarif"):
        assert runner.invoke(app, ["report", str(saved), "--format", fmt]).exit_code == 0

    assert scanner.calls == 1, "re-rendering must not re-scan"


# --- patch workflows, end to end through the CLI ----------------------------------------------------------------


def _patched_repo(tmp_path: Path) -> Path:
    _repo(tmp_path)
    target = tmp_path / "parser.py"
    target.write_text(_SOURCE, encoding="utf-8")
    return target


def _enriched(target: Path, **overrides):
    defaults = dict(diff=_DIFF, file=str(target))
    defaults.update(overrides)
    return AIEnrichedFinding(
        finding_id="SENT-001",
        title="unsafe yaml load",
        severity=Severity.HIGH,
        explanation="e",
        remediation="r",
        confidence_score=0.8,
        confidence_label=ConfidenceLabel.HIGH,
        verification_status=VerificationStatus.VERIFIED,
        structured_patch=StructuredPatch(**defaults),
    )


def _run_cli_with_patch(monkeypatch, tmp_path, target, extra_args):
    """Drive the real CLI with only the model call stubbed.

    Everything after enrich_findings - validation, backup, application,
    verification, rollback, statistics, reporting - executes for real.
    """
    # AI must look configured for --ai to pass its fail-fast guard. Same approach
    # as tests/test_cli_ai_flags.py: set the env vars and clear the cached
    # settings singleton so get_settings() rebuilds from them.
    from sentinelai.ai import config as ai_config

    monkeypatch.setenv("SENTINELAI_AI_LLM_MODEL", "test-model")
    monkeypatch.setenv("SENTINELAI_AI_EMBEDDING_MODEL", "test-embed")
    monkeypatch.setattr(ai_config, "_settings", None)

    scanner = _FixedScanner([_finding("SENT-001")])
    monkeypatch.setattr(
        "sentinelai.main._get_provider", lambda: LiveFindingsProvider(registry=_registry(scanner))
    )
    monkeypatch.setattr("sentinelai.main.enrich_findings", lambda *a, **k: [_enriched(target)])
    monkeypatch.setattr("sentinelai.main.create_default_retriever", lambda: object())
    monkeypatch.setattr("sentinelai.main.create_llm_generate_fn", lambda: (lambda *a, **k: ""))
    output = tmp_path / "report.json"
    result = runner.invoke(
        app, ["scan", str(tmp_path), "--ai", "--format", "json", "--output", str(output), *extra_args]
    )
    monkeypatch.setattr(ai_config, "_settings", None)
    return result, output


def test_apply_patches_writes_the_file_and_records_it_in_the_report(monkeypatch, tmp_path):
    """The join Phase 3.1 and 3.2 never crossed: patching and reporting in one run."""
    target = _patched_repo(tmp_path)

    result, output = _run_cli_with_patch(monkeypatch, tmp_path, target, ["--apply-patches"])

    assert result.exit_code in (ExitCode.SUCCESS, ExitCode.SECURITY_FINDINGS)
    assert target.read_text(encoding="utf-8") == _PATCHED, "the patch must reach the file"

    section = json.loads(output.read_text(encoding="utf-8"))["patch_application"]
    assert section["dry_run"] is False
    assert section["summary"]["applied"] == 1
    assert section["attempts"][0]["finding_id"] == "SENT-001"
    assert (tmp_path / ".sentinelai" / "backups" / "parser.py").read_text(encoding="utf-8") == _SOURCE


def test_dry_run_reports_the_patch_without_touching_the_repository(monkeypatch, tmp_path):
    """The dry-run promise, proven through the CLI rather than at the runner."""
    target = _patched_repo(tmp_path)
    before = target.read_text(encoding="utf-8")

    result, output = _run_cli_with_patch(monkeypatch, tmp_path, target, ["--apply-patches", "--dry-run"])

    assert result.exit_code in (ExitCode.SUCCESS, ExitCode.SECURITY_FINDINGS)
    assert target.read_text(encoding="utf-8") == before
    assert not (tmp_path / ".sentinelai").exists()

    section = json.loads(output.read_text(encoding="utf-8"))["patch_application"]
    assert section["dry_run"] is True
    assert section["summary"]["applied"] == 1


def test_a_scan_without_apply_patches_never_writes_or_reports_a_patch(monkeypatch, tmp_path):
    """Opt-in, asserted at the top of the pipeline rather than at the runner."""
    target = _patched_repo(tmp_path)
    before = target.read_text(encoding="utf-8")

    _, output = _run_cli_with_patch(monkeypatch, tmp_path, target, [])

    assert target.read_text(encoding="utf-8") == before
    assert not (tmp_path / ".sentinelai").exists()
    assert json.loads(output.read_text(encoding="utf-8"))["patch_application"] is None


def test_a_patch_that_does_not_fit_is_reported_and_leaves_the_file_alone(monkeypatch, tmp_path):
    """Best-effort patching: a failure is recorded, not raised, and the scan still succeeds."""
    target = _patched_repo(tmp_path)
    target.write_text("completely\ndifferent\ncontent\nhere\nnow\nok\n", encoding="utf-8")
    before = target.read_text(encoding="utf-8")

    result, output = _run_cli_with_patch(monkeypatch, tmp_path, target, ["--apply-patches"])

    assert result.exit_code in (ExitCode.SUCCESS, ExitCode.SECURITY_FINDINGS), "must not abort the scan"
    assert target.read_text(encoding="utf-8") == before

    section = json.loads(output.read_text(encoding="utf-8"))["patch_application"]
    assert section["summary"]["not_applicable"] == 1
    assert section["summary"]["applied"] == 0


def test_a_verification_failure_rolls_back_and_the_report_says_so(monkeypatch, tmp_path):
    """Rollback reaching the report - covered at unit level, never end to end."""
    target = _patched_repo(tmp_path)
    before = target.read_text(encoding="utf-8")

    from sentinelai.patching.applicator import PatchApplicator

    monkeypatch.setattr(
        PatchApplicator,
        "_atomic_write",
        lambda self, path, contents: Path(path).write_text("TRUNCATED", encoding="utf-8"),
    )

    result, output = _run_cli_with_patch(monkeypatch, tmp_path, target, ["--apply-patches"])

    assert result.exit_code in (ExitCode.SUCCESS, ExitCode.SECURITY_FINDINGS)
    assert target.read_text(encoding="utf-8") == before, "rollback must restore the original bytes"

    section = json.loads(output.read_text(encoding="utf-8"))["patch_application"]
    assert section["summary"]["rolled_back"] == 1
    assert section["attempts"][0]["rollback_executed"] is True
    assert section["attempts"][0]["rollback_failed"] is False


# --- the same chain, with real scanners -------------------------------------------------------------------------


_BENCHMARK_REPO = Path(__file__).resolve().parents[2] / "sentinelai-manual-test"
_REQUIRED = ("semgrep", "bandit", "gitleaks")
_missing = [tool for tool in _REQUIRED if shutil.which(tool) is None]


@pytest.mark.benchmark
@pytest.mark.skipif(not _BENCHMARK_REPO.is_dir(), reason=f"fixture not found at {_BENCHMARK_REPO}")
@pytest.mark.skipif(_missing, reason=f"scanner(s) not on PATH: {', '.join(_missing)}")
def test_real_scanners_survive_the_whole_chain_into_a_reloaded_report(tmp_path):
    """The same invariants, with nothing substituted.

    The tests above use a fixed-output scanner so the default suite needs no
    tools installed. This one runs the real three against the real fixture and
    asserts the chain end to end, so a defect that only appears with genuine
    scanner output cannot hide behind the stand-in.
    """
    result = LiveFindingsProvider().get_scan_result(str(_BENCHMARK_REPO), mode=ScanMode.STANDARD)
    stats = calculate_statistics(result)

    assert result.scanner_findings, "the fixture is intentionally vulnerable"
    assert stats.total_findings == len(result.scanner_findings)

    saved = tmp_path / "report.json"
    saved.write_text(to_json(result, stats), encoding="utf-8")
    reloaded, reloaded_stats = load_scan_result(saved)

    assert {f.finding_id for f in reloaded.scanner_findings} == {
        f.finding_id for f in result.scanner_findings
    }
    assert reloaded_stats.total_findings == stats.total_findings
    assert reloaded_stats.scanner_counts == stats.scanner_counts
    assert [c.correlation_id for c in reloaded.correlated_findings] == [
        c.correlation_id for c in result.correlated_findings
    ]

    for rendered in (to_markdown(reloaded, reloaded_stats), to_html(reloaded, reloaded_stats)):
        assert reloaded.scanner_findings[0].finding_id in rendered
