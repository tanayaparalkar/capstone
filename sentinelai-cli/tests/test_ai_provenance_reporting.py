"""AI provenance in every output format: present, absent ("AI: not used"), run-level in SARIF, and never a pattern name.

Fixtures only - ScanResults built here, and for the CLI a fake enrich_findings.
No provider is constructed or called.
"""
import io
import json
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

from sentinelai.ai.config import AISettings
from sentinelai.ai.redaction import PATTERN_NAMES
from sentinelai.contracts import (
    RedactionReport,
    RepositoryInfo,
    ScanMetadata,
    ScanMode,
    ScannerFinding,
    ScanResult,
    Severity,
)
from sentinelai.main import app
from sentinelai.presentation import render_summary
from sentinelai.providers import MockFindingsProvider
from sentinelai.reporting import load_scan_result, to_html, to_json, to_markdown, to_sarif
from sentinelai.reporting.provenance import provenance_line
from sentinelai.reporting.sarif_report import build_sarif_report
from sentinelai.statistics import calculate_statistics

runner = CliRunner()
# Every name the redactor can emit, plus GitLeaks-index names: none may reach any rendered output.
PATTERNS_HIT = sorted(PATTERN_NAMES) + ["gitleaks:generic-api-key", "gitleaks:aws-access-token"]
PROVENANCE = dict(
    ai_provider="groq",
    ai_model="llama-3.3-70b-versatile",
    egress_policy="allow",
    redaction=RedactionReport(secrets_masked=3, patterns_hit=PATTERNS_HIT, slices_scanned=7),
)
NEW_FIELDS = ("ai_provider", "ai_model", "egress_policy", "redaction")
_SARIF_KEYS = ("aiProvider", "aiModel", "egressPolicy", "redactionCount")


def _finding(finding_id: str = "SENT-001") -> ScannerFinding:
    return ScannerFinding(
        finding_id=finding_id,
        scanner="semgrep",
        category="sql-injection",
        severity=Severity.HIGH,
        file="app/db/queries.py",
        line_start=11,
        line_end=11,
        rule_id="python.lang.security.audit.formatted-sql-query",
        message="User input formatted into a SQL query.",
        raw_evidence="cursor.execute(f\"SELECT * FROM users WHERE name = '{username}'\")",
        cwe="CWE-89",
    )


def _result(**metadata) -> ScanResult:
    return ScanResult(
        repository=RepositoryInfo(name="demo-app", path="/tmp/demo-app"),
        metadata=ScanMetadata(timestamp="2026-10-29T12:00:00Z", mode=ScanMode.STANDARD, **metadata),
        scanner_findings=[_finding("SENT-001"), _finding("SENT-002")],
    )


@pytest.fixture
def enriched():
    return _result(**PROVENANCE)


@pytest.fixture
def scanner_only():
    return _result()


def _terminal(result: ScanResult) -> str:
    console = Console(file=io.StringIO(), width=200, record=True, color_system=None)
    render_summary(console, calculate_statistics(result), result.metadata)
    return console.export_text()


def _render_all(result: ScanResult) -> dict:
    stats = calculate_statistics(result)
    return {
        "terminal": _terminal(result),
        "markdown": to_markdown(result, stats),
        "html": to_html(result, stats),
        "json": to_json(result, stats),
        "sarif": to_sarif(result, stats),
    }


# ---- the shared line ----


def test_provenance_line_when_ai_ran(enriched):
    assert provenance_line(enriched.metadata) == "AI: groq/llama-3.3-70b-versatile · egress: allowed · 3 secrets masked"


def test_provenance_line_for_a_scanner_only_run(scanner_only):
    assert provenance_line(scanner_only.metadata) == "AI: not used"


def test_provenance_line_singular_and_zero():
    one = _result(**{**PROVENANCE, "redaction": RedactionReport(secrets_masked=1, slices_scanned=1)})
    zero = _result(**{**PROVENANCE, "egress_policy": "deny", "redaction": RedactionReport(secrets_masked=0, slices_scanned=4)})
    assert provenance_line(one.metadata).endswith("· 1 secret masked")
    assert provenance_line(zero.metadata) == "AI: groq/llama-3.3-70b-versatile · egress: denied · 0 secrets masked"


def test_unrecorded_redaction_is_not_shown_as_zero():
    result = _result(ai_provider="ollama", ai_model="llama3.1:8b", egress_policy="deny")
    assert provenance_line(result.metadata) == "AI: ollama/llama3.1:8b · egress: denied · redaction not applied"


def test_missing_model_and_egress_are_stated():
    result = _result(ai_provider="ollama")
    assert provenance_line(result.metadata) == "AI: ollama/unknown model · egress: not recorded · redaction not applied"


# ---- terminal ----


def test_terminal_summary_with_provenance(enriched):
    assert "AI: groq/llama-3.3-70b-versatile · egress: allowed · 3 secrets masked" in _terminal(enriched)


def test_terminal_summary_without_provenance(scanner_only):
    assert "AI: not used" in _terminal(scanner_only)


def test_terminal_does_not_interpret_a_model_name_as_markup():
    result = _result(ai_provider="ollama", ai_model="[bold red]evil[/bold red]")
    assert "ollama/[bold red]evil[/bold red]" in _terminal(result)


def test_terminal_summary_without_metadata_keeps_its_old_shape(enriched):
    console = Console(file=io.StringIO(), width=200, record=True, color_system=None)
    render_summary(console, calculate_statistics(enriched))
    assert "AI:" not in console.export_text()


# ---- markdown ----


def test_markdown_metadata_line_under_the_title(enriched):
    lines = to_markdown(enriched, calculate_statistics(enriched)).splitlines()
    assert lines[0] == "# SentinelAI Security Report"
    assert lines[2] == "*AI: groq/llama-3.3-70b-versatile · egress: allowed · 3 secrets masked*"


def test_markdown_metadata_line_without_provenance(scanner_only):
    lines = to_markdown(scanner_only, calculate_statistics(scanner_only)).splitlines()
    assert lines[2] == "*AI: not used*"


# ---- html ----


def _header(html: str) -> str:
    return html.split('<header class="report-header">', 1)[1].split("</header>", 1)[0]


def test_html_header_shows_provenance_beside_scan_metadata(enriched):
    header = _header(to_html(enriched, calculate_statistics(enriched)))
    assert "<strong>Repository:</strong> demo-app" in header
    assert "<strong>AI:</strong> groq/llama-3.3-70b-versatile" in header
    assert "<strong>Egress:</strong> allowed" in header
    assert "<strong>Redaction:</strong> 3 secrets masked" in header


def test_html_header_without_provenance(scanner_only):
    header = _header(to_html(scanner_only, calculate_statistics(scanner_only)))
    assert "<strong>AI:</strong> not used" in header
    assert "Egress:" not in header


def test_html_escapes_the_model_name():
    result = _result(ai_provider="ollama", ai_model="<script>alert(1)</script>")
    header = _header(to_html(result, calculate_statistics(result)))
    assert "<script>" not in header and "&lt;script&gt;" in header


# ---- json ----


def test_json_carries_the_fields_as_is_except_pattern_names(enriched):
    scan = json.loads(to_json(enriched, calculate_statistics(enriched)))["scan"]
    assert (scan["ai_provider"], scan["ai_model"], scan["egress_policy"]) == ("groq", "llama-3.3-70b-versatile", "allow")
    assert scan["redaction"] == {"secrets_masked": 3, "slices_scanned": 7}


def test_json_without_provenance_has_explicit_nulls(scanner_only):
    scan = json.loads(to_json(scanner_only, calculate_statistics(scanner_only)))["scan"]
    assert all(field in scan and scan[field] is None for field in NEW_FIELDS)


def test_json_report_round_trips_provenance(enriched, tmp_path):
    path = tmp_path / "report.json"
    path.write_text(to_json(enriched, calculate_statistics(enriched)), encoding="utf-8")
    loaded, _ = load_scan_result(path)
    assert (loaded.metadata.ai_provider, loaded.metadata.ai_model, loaded.metadata.egress_policy) == (
        "groq", "llama-3.3-70b-versatile", "allow",
    )
    assert loaded.metadata.redaction.secrets_masked == 3
    assert loaded.metadata.redaction.patterns_hit == []


def test_report_command_rerenders_provenance(enriched, tmp_path):
    path = tmp_path / "report.json"
    path.write_text(to_json(enriched, calculate_statistics(enriched)), encoding="utf-8")
    result = runner.invoke(app, ["report", str(path), "--format", "markdown"])
    assert result.exit_code == 0, result.stderr
    assert "*AI: groq/llama-3.3-70b-versatile · egress: allowed · 3 secrets masked*" in result.stdout


# ---- sarif ----


def test_sarif_provenance_is_in_the_run_level_property_bag(enriched):
    run = build_sarif_report(enriched, calculate_statistics(enriched))["runs"][0]
    bag = run["properties"]["sentinelai"]
    assert {key: bag[key] for key in _SARIF_KEYS} == {
        "aiProvider": "groq",
        "aiModel": "llama-3.3-70b-versatile",
        "egressPolicy": "allow",
        "redactionCount": 3,
    }


def test_sarif_without_provenance_records_nulls(scanner_only):
    bag = build_sarif_report(scanner_only, calculate_statistics(scanner_only))["runs"][0]["properties"]["sentinelai"]
    assert {key: bag[key] for key in _SARIF_KEYS} == dict.fromkeys(_SARIF_KEYS)


def _keys_anywhere(value) -> set:
    if isinstance(value, dict):
        return set(value) | set().union(*(_keys_anywhere(v) for v in value.values()))
    if isinstance(value, list):
        return set().union(*(_keys_anywhere(v) for v in value)) if value else set()
    return set()


def test_sarif_provenance_is_never_per_result(enriched):
    run = build_sarif_report(enriched, calculate_statistics(enriched))["runs"][0]
    assert len(run["results"]) == 2
    for result in run["results"]:
        assert not _keys_anywhere(result) & set(_SARIF_KEYS)
    assert not _keys_anywhere(run["tool"]) & set(_SARIF_KEYS)


def test_sarif_with_provenance_conforms_to_the_schema(enriched):
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads((Path(__file__).parent / "data" / "sarif-schema-2.1.0.json").read_text(encoding="utf-8"))
    jsonschema.validators.validator_for(schema)(schema).validate(build_sarif_report(enriched, calculate_statistics(enriched)))


# ---- pattern names never rendered ----


@pytest.mark.parametrize("fmt", ["terminal", "markdown", "html", "json", "sarif"])
def test_no_pattern_name_reaches_any_rendered_output(enriched, fmt):
    output = _render_all(enriched)[fmt]
    assert "3 secrets masked" in output or '"redactionCount": 3' in output or '"secrets_masked": 3' in output
    leaked = [name for name in PATTERNS_HIT if name in output]
    assert leaked == [], f"{fmt} output names redaction patterns: {leaked}"
    assert "patterns_hit" not in output


def test_pattern_name_check_is_not_vacuous(enriched):
    # The names are distinctive enough that they would be found if a renderer dumped the field.
    dumped = enriched.metadata.model_dump_json()
    assert all(name in dumped for name in PATTERNS_HIT)


def test_json_report_model_itself_carries_no_pattern_names(enriched):
    from sentinelai.reporting import build_json_report

    report = build_json_report(enriched, calculate_statistics(enriched))
    assert report.scan.redaction.patterns_hit == []
    assert enriched.metadata.redaction.patterns_hit == PATTERNS_HIT, "the caller's result must not be mutated"


# ---- the CLI records what actually ran ----


@pytest.fixture
def cli(monkeypatch):
    import sentinelai.ai.config as config
    import sentinelai.main as main_module

    for name in ("SENTINELAI_AI_PROVIDER", "SENTINELAI_AI_MODEL", "SENTINELAI_EGRESS", "SENTINELAI_AI_LLM_MODEL", "SENTINELAI_AI_EMBEDDING_MODEL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(config, "_settings", None)
    monkeypatch.setattr(main_module, "_get_provider", lambda: MockFindingsProvider())
    return main_module


def _enable_fake_ai(monkeypatch, main_module):
    monkeypatch.setenv("SENTINELAI_AI_LLM_MODEL", "fake-llm")
    monkeypatch.setattr(main_module, "get_settings", lambda: AISettings(llm_model="fake-llm", embedding_model="fake-embed"))
    monkeypatch.setattr(main_module, "create_retriever_for_scan", lambda: object())
    monkeypatch.setattr(main_module, "create_llm_provider", lambda cfg: __import__("sentinelai.ai.llm", fromlist=["MockLLMProvider"]).MockLLMProvider())
    monkeypatch.setattr(main_module, "enrich_findings", lambda *args, **kwargs: [])


def test_scanner_only_scan_records_no_provenance(cli, tmp_path):
    result = runner.invoke(app, ["scan", str(tmp_path), "--format", "json"])
    assert result.exit_code == 0, result.stderr
    scan = json.loads(result.stdout)["scan"]
    assert all(scan[field] is None for field in NEW_FIELDS)


def test_scanner_only_terminal_scan_says_ai_not_used(cli, tmp_path):
    result = runner.invoke(app, ["scan", str(tmp_path)])
    assert result.exit_code == 0, result.stderr
    assert "AI: not used" in result.stdout


def test_enriched_scan_records_the_provider_that_actually_ran(cli, monkeypatch, tmp_path):
    _enable_fake_ai(monkeypatch, cli)
    result = runner.invoke(app, ["scan", str(tmp_path), "--format", "json"])
    assert result.exit_code == 0, result.stderr
    scan = json.loads(result.stdout)["scan"]
    # The resolved provider; the legacy model variable is still the ollama default.
    assert (scan["ai_provider"], scan["ai_model"], scan["egress_policy"]) == ("ollama", "fake-llm", "deny")
    assert scan["redaction"] == {"secrets_masked": 0, "slices_scanned": 0}


def test_enriched_terminal_scan_shows_the_line(cli, monkeypatch, tmp_path):
    _enable_fake_ai(monkeypatch, cli)
    result = runner.invoke(app, ["scan", str(tmp_path)])
    assert result.exit_code == 0, result.stderr
    assert "AI: ollama/fake-llm · egress: denied · 0 secrets masked" in " ".join(result.stdout.split())


# ---- regression: `fix` calls scan() as a function ----


@pytest.mark.parametrize("args", [["--yes", "--dry-run"], ["--no-ai", "--yes", "--dry-run"]])
def test_fix_command_passes_every_scan_parameter(cli, monkeypatch, tmp_path, args):
    def must_not_run(*a, **k):
        raise AssertionError("fix must not run the --apply-patches writer")

    monkeypatch.setattr(cli, "run_patches", must_not_run)
    result = runner.invoke(app, ["fix", str(tmp_path), *args])
    assert result.exit_code == 0, (result.stderr, result.exception)
