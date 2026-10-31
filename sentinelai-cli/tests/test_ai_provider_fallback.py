"""Comprehensive regression suite for SentinelAI AI provider priority, automatic fallback,
and two-stage vulnerability verification architecture.

Covers:
1. Default priority order: Grok -> Gemini -> NVIDIA (skipped) -> Ollama.
2. Primary provider success avoids unnecessary fallback (subsequent providers never called).
3. Primary provider failure (e.g. 429) falls through to next authorized, configured provider.
4. Missing credentials trigger fallback to configured provider.
5. Authentication failure (401/403) triggers fallback.
6. Rate limit / quota exhaustion (429) triggers fallback.
7. Timeout / transport failure triggers fallback.
8. Invalid configuration (unknown provider, remote provider with egress deny) raises AIConfigError / exit code 2.
9. Egress denial (egress == "deny") does NOT fall back to another remote provider; only local Ollama is allowed.
10. Ollama attempted last and retains loopback endpoint validation.
11. Bounded sequence: no provider is called twice.
12. Scanner-only mode does not initialize providers or require API keys.
13. Two-stage analysis and verification separation: Stage A vs Stage B roles.
14. Verification failure (REJECTED) is not misreported as verified.
15. Provider operational errors do not become fabricated vulnerability findings.
16. Existing confidence, provenance, redaction, and reporting formats remain intact.
"""
import json
import urllib.request
from typing import ClassVar, Optional

import pytest
from typer.testing import CliRunner

from redaction_fixtures import FAKE_GROQ_KEY, OPAQUE
from sentinelai.ai import prompt_builder
from sentinelai.ai.agents.critic import critique
from sentinelai.ai.agents.evidence_analyst import assess_evidence
from sentinelai.ai.agents.exploit_remediation_analyst import assess_exploit_and_remediation
from sentinelai.ai.agents.schemas import (
    CriticAssessment,
    EvidenceAssessment,
    ExploitAssessment,
    ExploitRemediationAssessment,
    RemediationPlan,
)
from sentinelai.ai.agents.synthesizer import build_verifier_input, synthesize
from sentinelai.ai.config import (
    DEFAULT_PROVIDER_PRIORITY,
    PROVIDER_ALIASES,
    PROVIDERS,
    UNIMPLEMENTED_PROVIDERS,
    AIConfigError,
    resolve_config,
)
from sentinelai.ai.egress import EgressViolation
from sentinelai.ai.factory import create_llm_provider
from sentinelai.ai.llm.base import LLMProvider, LLMResponse, ProviderError
from sentinelai.ai.llm.errors import ProviderErrorKind
from sentinelai.ai.llm.fallback import FallbackLLMProvider, ProviderCandidate, create_fallback_provider
from sentinelai.ai.llm.mock import MockLLMProvider
from sentinelai.ai.llm.ollama import OllamaProvider as LocalOllamaProvider
from sentinelai.ai.pipeline import enrich_findings
from sentinelai.ai.repository_context import RepositoryContext
from sentinelai.ai.retrieval import RetrievedChunk
from sentinelai.ai.verifier import verify_finding
from sentinelai.contracts import (
    AIConfig,
    ConfidenceLabel,
    GroundingVerdict,
    ScannerFinding,
    Severity,
    VerificationStatus,
)
from sentinelai.core import ExitCode
from sentinelai.main import app

runner = CliRunner()


class FakeCallRecorderProvider(LLMProvider):
    """Test helper that records calls or raises configured operational errors."""

    name: ClassVar[str] = "fake"
    is_local: ClassVar[bool] = False
    retains_data: ClassVar[Optional[bool]] = None

    def __init__(
        self,
        name: str,
        *,
        is_local: bool = False,
        retains_data: Optional[bool] = None,
        error_to_raise: Optional[Exception] = None,
        response_text: str = '{"status": "ok"}',
    ):
        self.name = name
        self.is_local = is_local
        self.retains_data = retains_data
        self.error_to_raise = error_to_raise
        self.response_text = response_text
        self.call_count = 0
        self.prompts_received: list[str] = []

    def complete(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.1,
    ) -> LLMResponse:
        self.call_count += 1
        self.prompts_received.append(prompt)
        if self.error_to_raise is not None:
            raise self.error_to_raise
        return LLMResponse(
            text=self.response_text,
            prompt_tokens=10,
            completion_tokens=20,
            model=f"{self.name}-model",
        )


def _make_candidate(
    name: str,
    *,
    is_local: bool = False,
    retains_data: Optional[bool] = None,
    error_to_raise: Optional[Exception] = None,
    response_text: str = '{"status": "ok"}',
    egress: str = "allow",
) -> tuple[ProviderCandidate, FakeCallRecorderProvider]:
    fake_inst = FakeCallRecorderProvider(
        name=name,
        is_local=is_local,
        retains_data=retains_data,
        error_to_raise=error_to_raise,
        response_text=response_text,
    )
    cfg = AIConfig(provider=name, model=f"{name}-model", egress=egress)
    candidate = ProviderCandidate(
        name=name,
        model=f"{name}-model",
        is_local=is_local,
        retains_data=retains_data,
        config=cfg,
        factory=lambda: fake_inst,
    )
    return candidate, fake_inst


@pytest.fixture(autouse=True)
def _booby_trap_network(monkeypatch):
    """Ensure no test ever performs live network calls."""
    def no_network(*args, **kwargs):
        raise AssertionError("Network egress prohibited in test suite")

    monkeypatch.setattr(urllib.request, "urlopen", no_network)


# ==============================================================================
# Requirement 1: Default priority order: Grok -> Gemini -> NVIDIA -> Ollama
# ==============================================================================

def test_default_priority_order_and_skipping_unimplemented_nvidia():
    # New priority order: Gemini -> Grok (xAI) -> NVIDIA -> Ollama.
    # Groq remains a separate provider but is no longer in the default priority.
    assert DEFAULT_PROVIDER_PRIORITY == ("gemini", "grok", "nvidia", "ollama")
    # No alias mapping — "grok" and "groq" are distinct providers.
    assert PROVIDER_ALIASES == {}
    # All providers in the priority list are now implemented.
    assert UNIMPLEMENTED_PROVIDERS == set()

    # Build a fallback chain that includes all four default-priority providers.
    # NVIDIA is implemented now, so all four candidates are expected.
    cfg = AIConfig(
        provider="gemini",
        model="gemini-2.5-flash",
        egress="allow",
        fallback_providers=["grok", "nvidia", "ollama"],
    )
    fb_provider = create_fallback_provider(cfg)
    candidate_names = [c.name for c in fb_provider.candidates]
    assert candidate_names == ["gemini", "grok", "nvidia", "ollama"]


def test_grok_provider_is_native_xai_not_groq_alias(monkeypatch):
    """'grok' is a distinct provider (xAI native) — it no longer aliases 'groq'.
    resolve_config with provider='grok' returns provider='grok', not provider='groq'.
    The fallback candidate is named 'grok', served by GrokProvider.
    """
    monkeypatch.setenv("XAI_API_KEY", FAKE_GROQ_KEY)
    cfg = resolve_config({"provider": "grok", "egress": "allow"})
    # With the alias gone, 'grok' stays 'grok'
    assert cfg.provider == "grok"
    fb_provider = create_fallback_provider(cfg)
    assert fb_provider.candidates[0].name == "grok"
    # The first candidate is a GrokProvider (xAI), not a GroqProvider
    from sentinelai.ai.llm.xai import GrokProvider as XaiGrokProvider
    assert fb_provider.candidates[0].factory().__class__.__name__ == XaiGrokProvider.__name__


# ==============================================================================
# Requirement 2: Primary provider success avoids unnecessary fallback
# ==============================================================================

def test_primary_provider_success_avoids_unnecessary_fallback():
    cand1, fake1 = _make_candidate("groq", is_local=False, response_text='{"result": "from_groq"}')
    cand2, fake2 = _make_candidate("gemini", is_local=False, response_text='{"result": "from_gemini"}')
    cand3, fake3 = _make_candidate("ollama", is_local=True, response_text='{"result": "from_ollama"}')

    fb = FallbackLLMProvider([cand1, cand2, cand3], base_egress="allow")
    res = fb.complete("test prompt")

    assert res.text == '{"result": "from_groq"}'
    assert fake1.call_count == 1
    assert fake2.call_count == 0
    assert fake3.call_count == 0
    assert fb.active_provider_name == "groq"


# ==============================================================================
# Requirement 3: Primary provider failure falls through to next configured provider
# ==============================================================================

def test_primary_provider_failure_falls_through_to_next():
    cand1, fake1 = _make_candidate(
        "groq",
        is_local=False,
        error_to_raise=ProviderError("groq", ProviderErrorKind.HTTP, "503 Service Unavailable"),
    )
    cand2, fake2 = _make_candidate("gemini", is_local=False, response_text='{"result": "from_gemini"}')
    cand3, fake3 = _make_candidate("ollama", is_local=True, response_text='{"result": "from_ollama"}')

    fallback_events = []
    fb = FallbackLLMProvider(
        [cand1, cand2, cand3],
        base_egress="allow",
        on_fallback=lambda prev, nxt, exc: fallback_events.append((prev, nxt, type(exc).__name__)),
    )
    res = fb.complete("test prompt")

    assert res.text == '{"result": "from_gemini"}'
    assert fake1.call_count == 1
    assert fake2.call_count == 1
    assert fake3.call_count == 0
    assert fb.active_provider_name == "gemini"
    assert len(fb.fallback_faults) == 1
    assert fb.fallback_faults[0].kind == "ProviderError"
    assert fallback_events == [("groq", "gemini", "ProviderError")]


# ==============================================================================
# Requirement 4: Missing credentials trigger fallback to configured provider
# ==============================================================================

def test_missing_credentials_on_candidate_triggers_fallback():
    # Candidate 1 raises configuration error on init (simulating missing API key)
    cand1_cfg = AIConfig(provider="groq", model="m", egress="allow")
    def fail_init():
        raise ProviderError("groq", ProviderErrorKind.CONFIGURATION, "missing API key")

    cand1 = ProviderCandidate(
        name="groq",
        model="m",
        is_local=False,
        retains_data=None,
        config=cand1_cfg,
        factory=fail_init,
    )
    cand2, fake2 = _make_candidate("gemini", is_local=False, response_text='{"result": "from_gemini"}')

    fb = FallbackLLMProvider([cand1, cand2], base_egress="allow")
    res = fb.complete("test prompt")

    assert res.text == '{"result": "from_gemini"}'
    assert fake2.call_count == 1
    assert fb.active_provider_name == "gemini"
    assert len(fb.fallback_faults) == 1


# ==============================================================================
# Requirement 5: Authentication failure (401/403) triggers fallback
# ==============================================================================

def test_auth_failure_triggers_fallback():
    cand1, fake1 = _make_candidate(
        "groq",
        is_local=False,
        error_to_raise=ProviderError("groq", ProviderErrorKind.AUTH, "401 Unauthorized"),
    )
    cand2, fake2 = _make_candidate("ollama", is_local=True, response_text='{"result": "from_ollama"}')

    fb = FallbackLLMProvider([cand1, cand2], base_egress="allow")
    res = fb.complete("test prompt")

    assert res.text == '{"result": "from_ollama"}'
    assert fake1.call_count == 1
    assert fake2.call_count == 1
    assert fb.active_provider_name == "ollama"


# ==============================================================================
# Requirement 6: Rate limit / quota exhaustion (429) triggers fallback
# ==============================================================================

def test_rate_limit_429_triggers_fallback():
    cand1, fake1 = _make_candidate(
        "groq",
        is_local=False,
        error_to_raise=ProviderError("groq", ProviderErrorKind.RATE_LIMIT, "429 Too Many Requests"),
    )
    cand2, fake2 = _make_candidate("gemini", is_local=False, response_text='{"result": "from_gemini"}')

    fb = FallbackLLMProvider([cand1, cand2], base_egress="allow")
    res = fb.complete("test prompt")

    assert res.text == '{"result": "from_gemini"}'
    assert fake1.call_count == 1
    assert fake2.call_count == 1
    assert fb.active_provider_name == "gemini"


# ==============================================================================
# Requirement 7: Timeout and transport failure trigger fallback
# ==============================================================================

def test_timeout_and_transport_failure_trigger_fallback():
    cand1, fake1 = _make_candidate(
        "groq",
        is_local=False,
        error_to_raise=ProviderError("groq", ProviderErrorKind.TIMEOUT, "Request timed out after 30s"),
    )
    cand2, fake2 = _make_candidate(
        "gemini",
        is_local=False,
        error_to_raise=ProviderError("gemini", ProviderErrorKind.TRANSPORT, "Connection refused"),
    )
    cand3, fake3 = _make_candidate("ollama", is_local=True, response_text='{"result": "from_ollama"}')

    fb = FallbackLLMProvider([cand1, cand2, cand3], base_egress="allow")
    res = fb.complete("test prompt")

    assert res.text == '{"result": "from_ollama"}'
    assert fake1.call_count == 1
    assert fake2.call_count == 1
    assert fake3.call_count == 1
    assert fb.active_provider_name == "ollama"
    assert len(fb.fallback_faults) == 2


# ==============================================================================
# Requirement 8: Invalid configuration raises AIConfigError / exit code 2
# ==============================================================================

def test_invalid_configuration_raises_aiconfigerror():
    with pytest.raises(AIConfigError, match=r"--ai-provider must be one of"):
        resolve_config({"provider": "invalid_provider"})

    with pytest.raises(AIConfigError, match=r"is a remote provider and sends code off this machine, but egress is deny"):
        resolve_config({"provider": "groq", "egress": "deny"})


def test_cli_invalid_configuration_exits_2():
    result = runner.invoke(app, ["scan", ".", "--ai-provider", "nonexistent_provider"])
    assert result.exit_code == ExitCode.INVALID_INPUT == 2
    assert "must be one of" in result.stderr


# ==============================================================================
# Requirement 9: Egress denial does NOT fall back to another remote provider
# ==============================================================================

def test_egress_denial_skips_remote_providers_and_uses_local_ollama():
    cand1, fake1 = _make_candidate("groq", is_local=False, response_text="groq")
    cand2, fake2 = _make_candidate("gemini", is_local=False, response_text="gemini")
    cand3, fake3 = _make_candidate("ollama", is_local=True, response_text="ollama")

    # With base_egress="deny", remote providers groq and gemini must be skipped without any calls
    fb = FallbackLLMProvider([cand1, cand2, cand3], base_egress="deny")
    res = fb.complete("test prompt")

    assert res.text == "ollama"
    assert fake1.call_count == 0
    assert fake2.call_count == 0
    assert fake3.call_count == 1
    assert fb.active_provider_name == "ollama"


def test_egress_denial_when_no_local_provider_raises_error():
    cand1, fake1 = _make_candidate("groq", is_local=False, response_text="groq")
    cand2, fake2 = _make_candidate("gemini", is_local=False, response_text="gemini")

    fb = FallbackLLMProvider([cand1, cand2], base_egress="deny")
    with pytest.raises(ProviderError, match=r"all configured fallback providers failed"):
        fb.complete("test prompt")

    assert fake1.call_count == 0
    assert fake2.call_count == 0


# ==============================================================================
# Security Invariant: Egress violations for secrets or size NEVER fall back
# ==============================================================================

def test_secrets_egress_violation_fails_closed_immediately():
    cand1, fake1 = _make_candidate("groq", is_local=False, response_text="groq")
    cand2, fake2 = _make_candidate("gemini", is_local=False, response_text="gemini")

    fb = FallbackLLMProvider([cand1, cand2], base_egress="allow")

    # Prompt with unredacted raw AWS secret pattern
    tainted_prompt = "Vulnerability in code: AKIAIOSFODNN7EXAMPLE"
    with pytest.raises(EgressViolation) as excinfo:
        fb.complete(tainted_prompt)

    assert excinfo.value.check == "secrets"
    # Neither provider was allowed to see the prompt
    assert fake1.call_count == 0
    assert fake2.call_count == 0


def test_size_egress_violation_fails_closed_immediately():
    cand1, fake1 = _make_candidate("groq", is_local=False, response_text="groq")
    cand2, fake2 = _make_candidate("gemini", is_local=False, response_text="gemini")

    fb = FallbackLLMProvider([cand1, cand2], base_egress="allow")

    huge_prompt = "x" * 200_000
    with pytest.raises(EgressViolation) as excinfo:
        fb.complete(huge_prompt)

    assert excinfo.value.check == "size"
    assert fake1.call_count == 0
    assert fake2.call_count == 0


# ==============================================================================
# Requirement 10: Ollama attempted last and retains loopback endpoint validation
# ==============================================================================

def test_ollama_endpoint_retains_loopback_validation():
    # Remote endpoints must be strictly rejected
    with pytest.raises(ProviderError, match=r"loopback"):
        LocalOllamaProvider("llama3.1:8b", host="http://remote-server.evil.com:11434")

    with pytest.raises(ProviderError, match=r"loopback"):
        LocalOllamaProvider("llama3.1:8b", host="http://192.168.1.100:11434")

    # Loopback forms must be accepted
    p_127 = LocalOllamaProvider("llama3.1:8b", host="http://127.0.0.1:11434")
    assert p_127.base_url == "http://127.0.0.1:11434"

    p_lh = LocalOllamaProvider("llama3.1:8b", host="http://localhost:11434")
    assert p_lh.base_url == "http://localhost:11434"


# ==============================================================================
# Requirement 11: Bounded sequence: no provider is called twice
# ==============================================================================

def test_bounded_fallback_sequence_no_provider_called_twice():
    cand1, fake1 = _make_candidate(
        "groq",
        is_local=False,
        error_to_raise=ProviderError("groq", ProviderErrorKind.HTTP, "500 Internal Error"),
    )
    cand2, fake2 = _make_candidate(
        "gemini",
        is_local=False,
        error_to_raise=ProviderError("gemini", ProviderErrorKind.HTTP, "500 Internal Error"),
    )
    cand3, fake3 = _make_candidate(
        "ollama",
        is_local=True,
        error_to_raise=ProviderError("ollama", ProviderErrorKind.HTTP, "500 Internal Error"),
    )

    fb = FallbackLLMProvider([cand1, cand2, cand3], base_egress="allow")

    with pytest.raises(ProviderError):
        fb.complete("prompt 1")

    assert fake1.call_count == 1
    assert fake2.call_count == 1
    assert fake3.call_count == 1

    # On a subsequent call, all candidates are already marked failed; none should be recalled
    with pytest.raises(ProviderError):
        fb.complete("prompt 2")

    assert fake1.call_count == 1
    assert fake2.call_count == 1
    assert fake3.call_count == 1


# ==============================================================================
# Requirement 12: Scanner-only mode does not initialize providers or require API keys
# ==============================================================================

def test_scanner_only_mode_runs_without_ai_or_credentials(monkeypatch):
    import sentinelai.main as main_module
    from sentinelai.providers import MockFindingsProvider

    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GROK_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.delenv("SENTINELAI_AI_LLM_MODEL", raising=False)
    monkeypatch.delenv("SENTINELAI_AI_EMBEDDING_MODEL", raising=False)
    monkeypatch.setattr(main_module, "_get_provider", MockFindingsProvider)

    result = runner.invoke(app, ["scan", "."])
    assert result.exit_code == ExitCode.SUCCESS


# ==============================================================================
# Requirement 13 & 14: Two-stage analysis and verification architecture
# ==============================================================================

def test_two_stage_pipeline_separation_and_rejection():
    finding = ScannerFinding(
        finding_id="FIND-SQLI-01",
        scanner="semgrep",
        rule_id="RULE-SQLI-01",
        category="sql-injection",
        severity=Severity.HIGH,
        path="app/db.py",
        line_start=15,
        line_end=15,
        message="SQL injection flaw detected",
        raw_evidence="cursor.execute('SELECT * FROM users WHERE id = ' + user_id)",
        title="Unsanitized query in database cursor",
    )
    repo_ctx = RepositoryContext()
    chunks = [RetrievedChunk(kb_entry_id="KB-001", text="Parameterized query prevention", references=["CWE-89"], score=0.9)]

    # Stage A output: Candidate finding generated by analysts
    evidence_assessment = EvidenceAssessment(
        title="SQL injection via user_id parameter",
        interpretation="User input is concatenated directly into SQL statement creating vulnerability",
        observed_evidence=["user_id concatenated into SQL query string"],
        limitations=[],
        groundedness=GroundingVerdict.SUPPORTED,
    )
    exploit_assessment = ExploitAssessment(
        exploit_path="Attacker supplies 1 OR 1=1 to bypass authentication",
        impact="Complete database compromise",
        required_assumptions=["user_id comes from HTTP request"],
    )
    remediation_assessment = RemediationPlan(
        remediation="Use parameterized queries with bind variables",
        validation_steps=["Verify SQL statements use ? or %s placeholders"],
        patch_suggestion="cursor.execute('SELECT * FROM users WHERE id = %s', (user_id,))",
    )
    assessment = ExploitRemediationAssessment(
        exploit=exploit_assessment,
        remediation=remediation_assessment,
    )

    # Stage B output: Critic review + Deterministic verifier
    critic = CriticAssessment(
        verdict=GroundingVerdict.SUPPORTED,
        supported_claims=["User input is concatenated into query"],
        unsupported_claims=[],
    )

    # Verification passes when narrative matches category
    verifier_input = build_verifier_input(evidence_assessment, assessment)
    v_status = verify_finding(finding, verifier_input)
    assert v_status == VerificationStatus.VERIFIED

    enriched = synthesize(
        finding=finding,
        evidence=evidence_assessment,
        assessment=assessment,
        critic=critic,
        retrieved_chunks=chunks,
        confidence_score=0.9,
        confidence_label=ConfidenceLabel.HIGH,
        verification_status=v_status,
        repository_context=repo_ctx,
    )
    assert enriched.verification_status == VerificationStatus.VERIFIED
    assert enriched.grounding_verdict == GroundingVerdict.SUPPORTED
    assert enriched.finding_id == "FIND-SQLI-01"
    assert enriched.severity == Severity.HIGH


def test_verification_failure_rejected_is_strictly_preserved():
    finding = ScannerFinding(
        finding_id="FIND-CSRF-01",
        scanner="semgrep",
        rule_id="RULE-CSRF-01",
        category="cross-site-request-forgery",
        severity=Severity.MEDIUM,
        path="app/views.py",
        line_start=20,
        line_end=20,
        message="CSRF flaw detected",
        raw_evidence="def handle_post(request): return Response()",
        title="Missing CSRF token validation",
    )
    repo_ctx = RepositoryContext()

    # Hallucinated narrative discussing SQL injection instead of CSRF
    evidence_assessment = EvidenceAssessment(
        title="Database SQL injection flaw",
        interpretation="An attacker can run arbitrary SQL statements against database",
        observed_evidence=[],
        limitations=[],
        groundedness=GroundingVerdict.INSUFFICIENT_EVIDENCE,
    )
    assessment = ExploitRemediationAssessment(
        exploit=ExploitAssessment(exploit_path="Send SQL UNION payload", impact="Database leak"),
        remediation=RemediationPlan(remediation="Use parameterized database cursors"),
    )
    critic = CriticAssessment(
        verdict=GroundingVerdict.INSUFFICIENT_EVIDENCE,
        supported_claims=[],
        unsupported_claims=["Evidence does not mention SQL queries"],
    )

    verifier_input = build_verifier_input(evidence_assessment, assessment)
    v_status = verify_finding(finding, verifier_input)
    # The verifier deterministically rejects off-topic narrative
    assert v_status == VerificationStatus.REJECTED

    enriched = synthesize(
        finding=finding,
        evidence=evidence_assessment,
        assessment=assessment,
        critic=critic,
        retrieved_chunks=[],
        confidence_score=None,
        confidence_label=None,
        verification_status=v_status,
        repository_context=repo_ctx,
    )

    # REJECTED is preserved in the contract and never upgraded
    assert enriched.verification_status == VerificationStatus.REJECTED
    assert enriched.grounding_verdict == GroundingVerdict.INSUFFICIENT_EVIDENCE


# ==============================================================================
# Requirement 15: Operational errors do not become fabricated vulnerabilities
# ==============================================================================

def test_provider_operational_errors_do_not_fabricate_vulnerabilities():
    from sentinelai.core.errors import AIEnrichmentError

    class DummyRetriever:
        def retrieve(self, query, top_k):
            return []

    finding = ScannerFinding(
        finding_id="FIND-XSS-01",
        scanner="semgrep",
        rule_id="RULE-XSS-01",
        category="xss",
        severity=Severity.MEDIUM,
        path="app/templates/index.html",
        line_start=10,
        line_end=10,
        message="XSS flaw detected",
        raw_evidence="<div>{{ user_input | safe }}</div>",
        title="Unescaped HTML output",
    )

    def failing_generate(*args, **kwargs):
        raise ProviderError("mock", ProviderErrorKind.HTTP, "500 Server Down")

    faults = []
    # Pipeline handles failure on finding: logs warning, calls on_failure, does NOT invent findings
    with pytest.raises(AIEnrichmentError):
        enrich_findings(
            scanner_findings=[finding],
            retriever=DummyRetriever(),
            generate=failing_generate,
            score_confidence=lambda f, c: (None, None),
            verify_finding=lambda f, r: VerificationStatus.UNVERIFIED,
            on_failure=lambda fid, exc: faults.append((fid, exc)),
        )

    # All findings failed, so enrich_findings raised AIEnrichmentError
    # Finding was NOT duplicated or mutated into a fake finding
    assert len(faults) == 1
    assert faults[0][0] == "FIND-XSS-01"
    assert isinstance(faults[0][1], ProviderError)
