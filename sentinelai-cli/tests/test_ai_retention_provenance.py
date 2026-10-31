"""
Regression tests for retention-status disclosure in scan provenance and reports.

Verifies:
1. retains_data=None (e.g. Groq, Gemini) maps to 'unverified' and renders as
   'retention: unverified' (never reinterpreted as False / 'not retained').
2. retains_data=False (e.g. Ollama, Mock) maps to 'not_retained' and renders as
   'retention: not retained'.
3. retains_data=True maps to 'retained' and renders as 'retention: retained'.
4. Retention status is preserved across JSON (including load_scan_result and
   `sentinelai report` round-trips), Markdown, HTML, SARIF, and terminal summary.
5. Existing pre-retention saved reports and provider configurations remain
   compatible without network calls.
"""
import io
import json
import urllib.request
from pathlib import Path
from typing import ClassVar, Optional

import pytest
from agent_fakes import make_agent_generate
from redaction_fixtures import FAKE_GROQ_KEY
from rich.console import Console
from typer.testing import CliRunner

from sentinelai.ai.config import AISettings, EGRESS_ENV, MODEL_ENV, PROVIDER_ENV, PROVIDERS, resolve_config
from sentinelai.ai.llm import LLMProvider, LLMResponse, MockLLMProvider
from sentinelai.ai.llm.gemini import GeminiProvider
from sentinelai.ai.llm.groq import GroqProvider
from sentinelai.ai.llm.ollama import OllamaProvider
from sentinelai.contracts import (
    RedactionReport,
    RepositoryInfo,
    ScanMetadata,
    ScanMode,
    ScannerFinding,
    ScanResult,
    Severity,
    retention_status_from_provider,
)
from sentinelai.main import app
from sentinelai.presentation import render_summary
from sentinelai.providers import MockFindingsProvider
from sentinelai.reporting import load_scan_result, to_html, to_json, to_markdown, to_sarif
from sentinelai.reporting.provenance import provenance_items, provenance_line, retention_label, sarif_properties
from sentinelai.statistics import calculate_statistics

runner = CliRunner()


def _finding(finding_id: str = "SENT-001") -> ScannerFinding:
    return ScannerFinding(
        finding_id=finding_id,
        scanner="semgrep",
        category="sql-injection",
        severity=Severity.HIGH,
        file="app/db.py",
        line_start=10,
        rule_id="python.sql.concat",
        message="User input formatted into a SQL query.",
        raw_evidence="cursor.execute('SELECT * FROM users WHERE name=' + name)",
        cwe="CWE-89",
    )


def _scan_result(**metadata_kwargs) -> ScanResult:
    return ScanResult(
        repository=RepositoryInfo(name="demo-repo", path="/tmp/demo-repo"),
        metadata=ScanMetadata(timestamp="2026-10-30T12:00:00Z", mode=ScanMode.STANDARD, **metadata_kwargs),
        scanner_findings=[_finding()],
    )


def _html_header(html: str) -> str:
    return html.split('<header class="report-header">', 1)[1].split("</header>", 1)[0]


def _terminal_summary(result: ScanResult) -> str:
    console = Console(file=io.StringIO(), width=220, record=True, color_system=None)
    render_summary(console, calculate_statistics(result), result.metadata)
    return console.export_text()


# --- 1, 2, 3: Tri-state mapping (None / False / True) ----------------------------


@pytest.mark.parametrize(
    "raw_retains,expected_status,expected_label",
    [
        (None, "unverified", "unverified"),
        (False, "not_retained", "not retained"),
        (True, "retained", "retained"),
    ],
)
def test_tri_state_retention_mapping_and_labels(raw_retains, expected_status, expected_label):
    status = retention_status_from_provider(raw_retains)
    assert status == expected_status

    result = _scan_result(
        ai_provider="groq" if raw_retains is None else "mock",
        ai_model="test-model",
        egress_policy="allow" if raw_retains is None else "deny",
        retention_status=status,
        redaction=RedactionReport(secrets_masked=0, slices_scanned=1),
    )

    assert retention_label(result.metadata) == expected_label
    assert f"retention: {expected_label}" in provenance_line(result.metadata)
    assert ("Retention", expected_label) in provenance_items(result.metadata)
    assert sarif_properties(result.metadata)["retentionStatus"] == expected_status


def test_configured_provider_classes_map_to_expected_retention_status():
    assert GroqProvider.retains_data is None
    assert GeminiProvider.retains_data is None
    assert OllamaProvider.retains_data is False
    assert MockLLMProvider.retains_data is False

    assert retention_status_from_provider(PROVIDERS["groq"].retains_data) == "unverified"
    assert retention_status_from_provider(PROVIDERS["gemini"].retains_data) == "unverified"
    assert retention_status_from_provider(PROVIDERS["ollama"].retains_data) == "not_retained"
    assert retention_status_from_provider(PROVIDERS["mock"].retains_data) == "not_retained"


# --- 4: Preserved in JSON, Markdown, HTML, SARIF, and terminal -------------------


@pytest.mark.parametrize(
    "raw_retains,expected_json_status,expected_display",
    [
        (None, "unverified", "unverified"),
        (False, "not_retained", "not retained"),
        (True, "retained", "retained"),
    ],
)
def test_all_report_formats_preserve_retention_status(raw_retains, expected_json_status, expected_display, tmp_path):
    result = _scan_result(
        ai_provider="groq",
        ai_model="llama-3.3-70b-versatile",
        egress_policy="allow",
        retention_status=retention_status_from_provider(raw_retains),
        redaction=RedactionReport(secrets_masked=2, slices_scanned=3),
    )
    stats = calculate_statistics(result)

    # JSON + round-trip via load_scan_result and `sentinelai report`
    raw_json = to_json(result, stats)
    parsed = json.loads(raw_json)
    assert parsed["scan"]["retention_status"] == expected_json_status

    json_path = tmp_path / "report.json"
    json_path.write_text(raw_json, encoding="utf-8")
    loaded, loaded_stats = load_scan_result(json_path)
    assert loaded.metadata.retention_status == expected_json_status

    # Markdown (both direct and re-rendered from saved JSON via `sentinelai report`)
    md = to_markdown(result, stats)
    assert f"retention: {expected_display}" in md.splitlines()[2]
    if raw_retains is None:
        assert "retention: not retained" not in md

    rerendered_md = runner.invoke(app, ["report", str(json_path), "--format", "markdown"])
    assert rerendered_md.exit_code == 0, rerendered_md.stderr
    assert f"retention: {expected_display}" in rerendered_md.stdout

    # HTML (both direct and re-rendered from saved JSON)
    html = to_html(result, stats)
    header = _html_header(html)
    assert f"<strong>Retention:</strong> {expected_display}" in header
    if raw_retains is None:
        assert "<strong>Retention:</strong> not retained" not in header

    rerendered_html = runner.invoke(app, ["report", str(json_path), "--format", "html"])
    assert rerendered_html.exit_code == 0, rerendered_html.stderr
    assert f"<strong>Retention:</strong> {expected_display}" in _html_header(rerendered_html.stdout)

    # Terminal summary
    term = _terminal_summary(loaded)
    assert f"retention: {expected_display}" in term

    # SARIF + schema conformance
    sarif_doc = json.loads(to_sarif(result, stats))
    assert sarif_doc["runs"][0]["properties"]["sentinelai"]["retentionStatus"] == expected_json_status
    jsonschema = pytest.importorskip("jsonschema", reason="dev-only dependency")
    schema = json.loads((Path(__file__).parent / "data" / "sarif-schema-2.1.0.json").read_text(encoding="utf-8"))
    jsonschema.validate(instance=sarif_doc, schema=schema)


# --- End-to-end CLI scan propagation for None, False, and True -------------------


@pytest.fixture
def isolated_cli(monkeypatch):
    import sentinelai.ai.config as config
    import sentinelai.main as main_module

    for env_var in (
        PROVIDER_ENV,
        MODEL_ENV,
        EGRESS_ENV,
        "GROQ_API_KEY",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "GOOGLE_CLOUD_PROJECT",
        "SENTINELAI_AI_LLM_MODEL",
        "SENTINELAI_AI_EMBEDDING_MODEL",
    ):
        monkeypatch.delenv(env_var, raising=False)
    monkeypatch.setattr(config, "_settings", None)
    monkeypatch.setattr(main_module, "_get_provider", lambda: MockFindingsProvider())

    def forbid_network(*args, **kwargs):
        raise AssertionError("network must not be called in retention provenance tests")

    monkeypatch.setattr(urllib.request, "urlopen", forbid_network)
    yield main_module
    monkeypatch.setattr(config, "_settings", None)


@pytest.mark.parametrize(
    "provider_name,egress_mode,provider_retains,expected_json_status,expected_display",
    [
        ("groq", "allow", None, "unverified", "unverified"),
        ("gemini", "allow", None, "unverified", "unverified"),
        ("mock", "deny", False, "not_retained", "not retained"),
        ("mock", "allow", True, "retained", "retained"),
    ],
)
def test_cli_scan_propagates_provider_retention_into_json_markdown_and_html(
    isolated_cli,
    monkeypatch,
    tmp_path,
    provider_name,
    egress_mode,
    provider_retains,
    expected_json_status,
    expected_display,
):
    if provider_name == "groq":
        monkeypatch.setenv("GROQ_API_KEY", FAKE_GROQ_KEY)
    elif provider_name == "gemini":
        sa_file = tmp_path / "fake-sa.json"
        sa_file.write_text("{}", encoding="utf-8")
        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(sa_file))
        monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "demo-project")

    generate = make_agent_generate()

    class _StubProvider(LLMProvider):
        name: ClassVar[str] = provider_name
        is_local: ClassVar[bool] = provider_name in ("ollama", "mock")
        retains_data: ClassVar[Optional[bool]] = provider_retains

        def complete(
            self,
            prompt: str,
            *,
            system: Optional[str] = None,
            max_tokens: int = 2048,
            temperature: float = 0.1,
        ) -> LLMResponse:
            return LLMResponse(text=generate(prompt), model="stub-model", latency_ms=1.0)

    monkeypatch.setattr(isolated_cli, "create_llm_provider", lambda cfg: _StubProvider())

    # JSON output
    res_json = runner.invoke(
        app,
        ["scan", str(tmp_path), "--ai-provider", provider_name, "--egress", egress_mode, "--format", "json"],
    )
    assert res_json.exit_code == 0, res_json.stderr
    assert json.loads(res_json.stdout)["scan"]["retention_status"] == expected_json_status

    # Markdown output
    res_md = runner.invoke(
        app,
        ["scan", str(tmp_path), "--ai-provider", provider_name, "--egress", egress_mode, "--format", "markdown"],
    )
    assert res_md.exit_code == 0, res_md.stderr
    assert f"retention: {expected_display}" in res_md.stdout

    # HTML output
    res_html = runner.invoke(
        app,
        ["scan", str(tmp_path), "--ai-provider", provider_name, "--egress", egress_mode, "--format", "html"],
    )
    assert res_html.exit_code == 0, res_html.stderr
    assert f"<strong>Retention:</strong> {expected_display}" in _html_header(res_html.stdout)


# --- 5: Legacy report & provider configuration compatibility ---------------------


def test_legacy_saved_enriched_report_without_retention_status_loads_as_unverified(tmp_path):
    legacy_enriched = _scan_result(
        ai_provider="groq",
        ai_model="llama-3.3-70b-versatile",
        egress_policy="allow",
        redaction=RedactionReport(secrets_masked=1, slices_scanned=2),
    )
    payload = json.loads(to_json(legacy_enriched, calculate_statistics(legacy_enriched)))
    payload["scan"].pop("retention_status")

    path = tmp_path / "legacy_enriched_report.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded, stats = load_scan_result(path)
    assert loaded.metadata.retention_status is None
    assert "retention: unverified" in provenance_line(loaded.metadata)
    assert "retention: not retained" not in provenance_line(loaded.metadata)
    assert "<strong>Retention:</strong> unverified" in _html_header(to_html(loaded, stats))


def test_existing_provider_resolution_is_unchanged(isolated_cli, monkeypatch):
    # Retention disclosure does not add a new consent gate or alter resolve_config.
    monkeypatch.setenv("GROQ_API_KEY", FAKE_GROQ_KEY)
    cfg = resolve_config({"provider": "groq", "egress": "allow"})
    assert (cfg.provider, cfg.model, cfg.egress) == ("groq", "llama-3.3-70b-versatile", "allow")


def test_remote_provider_class_with_retains_data_false_stays_unverified_without_runtime_verification():
    """A remote provider cannot be reported as 'not_retained' solely because a class attribute is False."""
    assert retention_status_from_provider(False, is_local=False) == "unverified"
    assert retention_status_from_provider(False, is_local=False, account_verified=True) == "not_retained"
    assert retention_status_from_provider(False, is_local=True) == "not_retained"


def test_provider_retention_basis_and_policy_notes_reflect_documented_semantics():
    """Each provider exposes documented retention_basis and retention_policy_note without overclaiming."""
    from sentinelai.ai.llm.nvidia import NvidiaNimProvider
    from sentinelai.ai.llm.xai import GrokProvider
    from sentinelai.contracts import aggregate_retention_status

    for remote_cls in (GrokProvider, GroqProvider, GeminiProvider, NvidiaNimProvider):
        assert remote_cls.is_local is False
        assert remote_cls.retains_data is None
        assert retention_status_from_provider(remote_cls.retains_data, is_local=remote_cls.is_local) == "unverified"
        assert "unverified" in remote_cls.retention_basis
        assert "cannot be verified at runtime" in remote_cls.retention_policy_note

    assert OllamaProvider.is_local is True
    assert OllamaProvider.retains_data is False
    assert OllamaProvider.retention_basis == "local_loopback_unverified_host_persistence"
    assert "does not verify the absence of local server logs" in OllamaProvider.retention_policy_note

    # Conservative aggregation across multiple providers
    assert aggregate_retention_status(["not_retained", "not_retained"]) == "not_retained"
    assert aggregate_retention_status(["not_retained", "unverified"]) == "unverified"
    assert aggregate_retention_status(["not_retained", "unverified", "retained"]) == "retained"
    assert aggregate_retention_status([]) == "unverified"

