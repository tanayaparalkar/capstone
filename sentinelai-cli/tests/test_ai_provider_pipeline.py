"""The composed provider path: slice -> redact -> prompt -> assert_safe -> complete -> RedactionReport.

Every test drives the real CLI (`sentinelai scan`) over a small fixture repository
whose code slices contain planted secrets. Scanner findings come from a fixed
FindingsProvider; the LLM is MockLLMProvider or a recording fake. urllib is
booby-trapped throughout, so any network call fails the test.
"""
import json
import urllib.request
from pathlib import Path
from typing import ClassVar, Optional

import pytest
from typer.testing import CliRunner

from agent_fakes import GenerateBackedProvider, make_agent_generate
from redaction_fixtures import FAKE_GROQ_KEY, OPAQUE
from redaction_fixtures import PLANTED as PLANTED_ENTRIES
from sentinelai.ai.llm import LLMResponse, MockLLMProvider
from sentinelai.ai.llm.errors import ProviderError, ProviderErrorKind
from sentinelai.contracts import AIConfig, RepositoryInfo, ScanMetadata, ScannerFinding, ScanResult, Severity
from sentinelai.core import ExitCode
from sentinelai.correlation import correlate_findings
from sentinelai.main import app
from sentinelai.providers import FindingsProvider

runner = CliRunner()
PLANTED = {e["pattern"]: e["secret"] for e in PLANTED_ENTRIES}
# Matched by no pattern: only the GitLeaks position in the SecretIndex can mask it.
SECRETS = [PLANTED["secret-assignment"], PLANTED["aws-access-key"], PLANTED["jwt"], PLANTED["stripe-key"], OPAQUE]


def _app_source() -> str:
    return "\n".join([
        "import psycopg2",                                             # 1
        f'DB_PASSWORD = "{PLANTED["secret-assignment"]}"',             # 2
        f'AWS_KEY_ID = "{PLANTED["aws-access-key"]}"',                 # 3
        f'GENERIC = "{OPAQUE}"',                                       # 4  <- GitLeaks finding
        "",                                                            # 5
        "def find_user(cursor, name):",                                # 6
        "    cursor.execute(f\"SELECT * FROM users WHERE name = '{name}'\")",  # 7 <- SENT-001
        f'    token = "{PLANTED["jwt"]}"',                             # 8
        "    return cursor.fetchone()",                                # 9
        "",
    ])


def _findings(repo: Path) -> list[ScannerFinding]:
    app_py = str(repo / "app.py")
    billing = str(repo / "billing.py")
    return [
        ScannerFinding(finding_id="SENT-001", scanner="semgrep", category="sql-injection", severity=Severity.HIGH,
                       file=app_py, line_start=7, line_end=7, rule_id="python.sqli", message="Formatted SQL query.",
                       raw_evidence="    cursor.execute(f\"SELECT * FROM users WHERE name = '{name}'\")", cwe="CWE-89"),
        ScannerFinding(finding_id="SENT-002", scanner="semgrep", category="hardcoded-secret", severity=Severity.MEDIUM,
                       file=billing, line_start=2, line_end=2, rule_id="python.hardcoded-key",
                       message="Hard-coded API key.", raw_evidence=f'stripe.api_key = "{PLANTED["stripe-key"]}"'),
        ScannerFinding(finding_id="SENT-003", scanner="gitleaks", category="generic-api-key", severity=Severity.HIGH,
                       file=app_py, line_start=4, line_end=4, rule_id="generic-api-key",
                       message="Detected a Generic API Key.", raw_evidence='GENERIC = "REDACTED"'),
    ]


class _FixedProvider(FindingsProvider):
    def __init__(self, findings):
        self.findings = findings

    def get_scan_result(self, repo_path, mode=None, tier=None):
        return ScanResult(
            repository=RepositoryInfo(name="fixture", path=repo_path),
            metadata=ScanMetadata(timestamp="2026-10-29T12:00:00Z", mode="standard"),
            scanner_findings=self.findings,
            correlated_findings=correlate_findings(self.findings),
        )


@pytest.fixture
def repo(tmp_path, monkeypatch):
    import sentinelai.ai.config as config
    import sentinelai.main as main_module

    (tmp_path / "app.py").write_text(_app_source(), encoding="utf-8")
    (tmp_path / "billing.py").write_text(f'import stripe\nstripe.api_key = "{PLANTED["stripe-key"]}"\n', encoding="utf-8")
    for name in ("SENTINELAI_AI_PROVIDER", "SENTINELAI_AI_MODEL", "SENTINELAI_EGRESS", "SENTINELAI_AI_LLM_MODEL",
                 "SENTINELAI_AI_EMBEDDING_MODEL", "GROQ_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(config, "_settings", None)
    monkeypatch.setattr(main_module, "_get_provider", lambda: _FixedProvider(_findings(tmp_path)))

    def no_network(*args, **kwargs):
        raise AssertionError("no network call may happen in these tests")

    monkeypatch.setattr(urllib.request, "urlopen", no_network)
    yield tmp_path
    monkeypatch.setattr(config, "_settings", None)


def _use(monkeypatch, provider):
    import sentinelai.main as main_module

    monkeypatch.setattr(main_module, "create_llm_provider", lambda cfg: provider)
    return provider


def _scan(repo, *args):
    result = runner.invoke(app, ["scan", str(repo), "--format", "json", *args])
    report = json.loads(result.stdout) if result.stdout.strip().startswith("{") else None
    return result, report


def _leaks(text: str) -> list[str]:
    return [s for s in SECRETS if any(s[i : i + 8] in text for i in range(len(s) - 7))]


# ---- end to end: no planted secret reaches the provider ----


def test_mock_provider_receives_no_planted_secret(repo, monkeypatch):
    provider = _use(monkeypatch, MockLLMProvider())
    result, report = _scan(repo, "--ai-provider", "mock")
    assert result.exit_code == 0, result.stderr
    assert provider.calls, "the provider must have been called"
    for prompt in provider.calls:
        assert _leaks(prompt) == []
    # The secrets were in the slices the prompts were built from - and were masked there.
    assert any("«REDACTED:secret-assignment»" in p and "cursor.execute" in p for p in provider.calls)
    assert any("«REDACTED:gitleaks:generic-api-key»" in p for p in provider.calls)


def test_every_agent_prompt_is_redacted(repo, monkeypatch):
    # A schema-aware fake gets all three agents (evidence, exploit/remediation, critic) to run.
    generate = make_agent_generate()
    _use(monkeypatch, GenerateBackedProvider(generate))
    result, report = _scan(repo, "--ai-provider", "ollama")
    assert result.exit_code == 0, result.stderr
    assert len(generate.calls) >= 3 * 2
    for prompt, _ in generate.calls:
        assert _leaks(prompt) == []
    assert {f["finding_id"] for f in report["findings"]["ai_enriched"]}


def test_redaction_report_is_recorded_in_scan_metadata(repo, monkeypatch):
    _use(monkeypatch, MockLLMProvider())
    _, report = _scan(repo, "--ai-provider", "mock")
    scan = report["scan"]
    assert (scan["ai_provider"], scan["ai_model"], scan["egress_policy"]) == ("mock", "mock-model", "deny")
    # SENT-001's slice: password, AWS key, GitLeaks-flagged line, JWT; SENT-002's evidence and slice: stripe key.
    assert scan["redaction"]["secrets_masked"] >= 5
    assert scan["redaction"]["slices_scanned"] >= 4
    assert "patterns_hit" not in scan["redaction"]


def test_secret_index_is_built_once_from_every_finding_before_any_prompt(repo, monkeypatch):
    import sentinelai.ai.prompt_builder as prompt_builder

    events = []

    class CountingIndex(prompt_builder.SecretIndex):
        def __init__(self):
            events.append("index")
            super().__init__()

        def add_from_findings(self, findings):
            events.append(("findings", sorted(f.finding_id for f in findings)))
            super().add_from_findings(findings)

    class RecordingMock(MockLLMProvider):
        def complete(self, prompt, *, max_tokens, temperature):
            events.append("complete")
            return super().complete(prompt, max_tokens=max_tokens, temperature=temperature)

    monkeypatch.setattr(prompt_builder, "SecretIndex", CountingIndex)
    _use(monkeypatch, RecordingMock())
    result, _ = _scan(repo, "--ai-provider", "mock")
    assert result.exit_code == 0, result.stderr
    assert events[:2] == ["index", ("findings", ["SENT-001", "SENT-002", "SENT-003"])]
    assert events.count("index") == 1
    assert "complete" in events[2:]


# ---- assert_safe runs before complete, on every path ----


class _Recorder:
    def __init__(self):
        self.events = []


def _instrument(monkeypatch, provider):
    import sentinelai.ai.prompt_builder as prompt_builder

    recorder = _Recorder()
    real_assert_safe = prompt_builder.assert_safe

    def recording_assert_safe(payload, cfg, prov, **kwargs):
        recorder.events.append(("assert_safe", payload))
        return real_assert_safe(payload, cfg, prov, **kwargs)

    real_complete = provider.complete

    def recording_complete(prompt, *, max_tokens, temperature):
        recorder.events.append(("complete", prompt))
        return real_complete(prompt, max_tokens=max_tokens, temperature=temperature)

    monkeypatch.setattr(prompt_builder, "assert_safe", recording_assert_safe)
    monkeypatch.setattr(provider, "complete", recording_complete)
    _use(monkeypatch, provider)
    return recorder


def _assert_guarded(events):
    completes = [i for i, (kind, _) in enumerate(events) if kind == "complete"]
    assert completes, "expected provider calls"
    for i in completes:
        assert i > 0 and events[i - 1] == ("assert_safe", events[i][1]), "complete() without assert_safe() on that prompt"


def test_assert_safe_precedes_every_complete_with_the_mock(repo, monkeypatch):
    recorder = _instrument(monkeypatch, MockLLMProvider())
    result, _ = _scan(repo, "--ai-provider", "mock")
    assert result.exit_code == 0, result.stderr
    _assert_guarded(recorder.events)


def test_assert_safe_precedes_every_complete_across_all_agents(repo, monkeypatch):
    recorder = _instrument(monkeypatch, GenerateBackedProvider(make_agent_generate()))
    result, _ = _scan(repo, "--ai-provider", "ollama")
    assert result.exit_code == 0, result.stderr
    _assert_guarded(recorder.events)
    assert sum(kind == "complete" for kind, _ in recorder.events) >= 6


def test_a_refused_prompt_never_reaches_complete(repo, monkeypatch):
    import sentinelai.ai.prompt_builder as prompt_builder
    from sentinelai.ai.egress import EgressViolation

    def refuse(payload, cfg, provider, **kwargs):
        raise EgressViolation("secrets", "payload matches secret pattern(s): jwt", ["jwt"])

    class MustNotBeCalled(MockLLMProvider):
        def complete(self, prompt, *, max_tokens, temperature):
            raise AssertionError("complete() reached without passing assert_safe()")

    monkeypatch.setattr(prompt_builder, "assert_safe", refuse)
    _use(monkeypatch, MustNotBeCalled())
    result, report = _scan(repo, "--ai-provider", "mock")
    assert result.exit_code == 0, result.stderr
    assert report["findings"]["ai_enriched"] == []
    assert {f["kind"] for f in report["scan"]["ai_faults"]} == {"EgressViolation"}


def test_the_pipeline_is_only_ever_given_the_guarded_generate(repo, monkeypatch):
    import sentinelai.main as main_module

    seen = []
    monkeypatch.setattr(main_module, "enrich_findings", lambda findings, retriever, generate, *a, **k: seen.append(generate) or [])
    _use(monkeypatch, MockLLMProvider())
    _scan(repo, "--ai-provider", "mock")
    assert len(seen) == 1 and seen[0].__qualname__ == "guarded_generate.<locals>.generate"


# ---- ProviderError mid-scan ----


class _FailsOn(GenerateBackedProvider):
    def __init__(self, generate, finding_id):
        super().__init__(generate)
        self.finding_id = finding_id

    def complete(self, prompt, *, max_tokens, temperature):
        if f"Finding: {self.finding_id}\n" in prompt:
            raise ProviderError("ollama", ProviderErrorKind.RATE_LIMIT, status=429)
        return super().complete(prompt, max_tokens=max_tokens, temperature=temperature)


def test_provider_error_mid_scan_leaves_that_finding_unenriched_and_records_the_fault(repo, monkeypatch):
    _use(monkeypatch, _FailsOn(make_agent_generate(), "SENT-002"))
    result, report = _scan(repo, "--ai-provider", "ollama")
    assert result.exit_code == ExitCode.SUCCESS, result.stderr
    enriched = {f["finding_id"] for f in report["findings"]["ai_enriched"]}
    assert "SENT-002" not in enriched and enriched
    assert report["scan"]["ai_faults"] == [
        {"finding_id": "SENT-002", "kind": "ProviderError", "message": "ollama: rate_limit (HTTP 429)"}
    ]
    # Never converted into a finding.
    assert {f["finding_id"] for f in report["findings"]["scanner"]} == {"SENT-001", "SENT-002", "SENT-003"}


def test_provider_failure_does_not_change_the_exit_code(repo, monkeypatch):
    # Highest scanner severity is HIGH: --fail-on critical must not trip because enrichment failed.
    _use(monkeypatch, _FailsOn(make_agent_generate(), "SENT-002"))
    result, report = _scan(repo, "--ai-provider", "ollama", "--fail-on", "critical")
    assert report["scan"]["ai_faults"], "enrichment must actually have run and failed"
    assert result.exit_code == ExitCode.SUCCESS
    assert result.exit_code != ExitCode.SECURITY_FINDINGS


def test_every_finding_failing_still_completes_the_scan(repo, monkeypatch):
    class AlwaysDown(MockLLMProvider):
        def complete(self, prompt, *, max_tokens, temperature):
            raise ProviderError("mock", ProviderErrorKind.TRANSPORT, "ConnectionRefusedError")

    _use(monkeypatch, AlwaysDown())
    result, report = _scan(repo, "--ai-provider", "mock", "--fail-on", "critical")
    assert result.exit_code == ExitCode.SUCCESS, result.stderr
    assert report["findings"]["ai_enriched"] == []
    faults = report["scan"]["ai_faults"]
    assert {f["finding_id"] for f in faults} == {"SENT-001", "SENT-002", "SENT-003"}
    assert all(f["kind"] == "ProviderError" for f in faults), "the all-failed summary must not be recorded twice"
    assert "AI enrichment failed; continuing without it" in " ".join(result.stderr.split())


def test_provider_construction_failure_is_a_scan_level_fault(repo, monkeypatch):
    import sentinelai.main as main_module

    def broken(cfg):
        raise ProviderError("ollama", ProviderErrorKind.CONFIGURATION, "host must be a loopback address")

    monkeypatch.setattr(main_module, "create_llm_provider", broken)
    result, report = _scan(repo, "--ai-provider", "ollama")
    assert result.exit_code == ExitCode.SUCCESS, result.stderr
    assert report["scan"]["ai_faults"] == [
        {"finding_id": None, "kind": "ProviderError", "message": "ollama: configuration: host must be a loopback address"}
    ]


def test_a_non_provider_exception_is_recorded_by_type_only(repo, monkeypatch):
    class Echoes(MockLLMProvider):
        def complete(self, prompt, *, max_tokens, temperature):
            raise RuntimeError(f"model echoed: {prompt}")

    _use(monkeypatch, Echoes())
    _, report = _scan(repo, "--ai-provider", "mock")
    assert {f["message"] for f in report["scan"]["ai_faults"]} == {"RuntimeError"}


# ---- egress=deny with a remote provider ----


def test_deny_with_a_remote_provider_refuses_before_any_scan_or_network_call(repo, monkeypatch):
    import sentinelai.main as main_module

    scanned = []
    monkeypatch.setattr(main_module, "_get_provider", lambda: scanned.append(True) or _FixedProvider([]))
    monkeypatch.setenv("GROQ_API_KEY", FAKE_GROQ_KEY)
    result = runner.invoke(app, ["scan", str(repo), "--format", "json", "--ai-provider", "groq", "--egress", "deny"])
    assert result.exit_code == ExitCode.INVALID_INPUT
    assert "egress is deny" in " ".join(result.stderr.split())
    assert scanned == []


def test_deny_is_enforced_by_the_guard_even_if_resolution_were_bypassed(repo, monkeypatch):
    # Defence in depth: hand main a remote config with egress deny directly. The real GroqProvider
    # is built, and assert_safe must refuse every prompt before urllib is touched.
    import sentinelai.main as main_module

    monkeypatch.setenv("GROQ_API_KEY", FAKE_GROQ_KEY)
    monkeypatch.setattr(
        main_module, "_resolve_ai_config", lambda overrides: AIConfig(provider="groq", model="llama-3.3-70b-versatile")
    )
    result, report = _scan(repo, "--ai-provider", "groq")
    assert result.exit_code == ExitCode.SUCCESS, result.stderr
    assert report["findings"]["ai_enriched"] == []
    faults = report["scan"]["ai_faults"]
    assert faults and all(f["kind"] == "EgressViolation" and "policy" in f["message"] for f in faults)
    assert FAKE_GROQ_KEY not in result.stdout + result.stderr
