"""
Regression tests for new provider identities and role-separation architecture.

Covers:
  1. GrokProvider (xAI) identity: name, endpoint, credential, default model.
  2. NvidiaNimProvider identity: name, endpoint, credential, default model.
  3. GroqProvider does NOT read XAI_API_KEY or GROK_API_KEY.
  4. Provider priority order: Gemini -> Grok -> NVIDIA -> Ollama.
  5. 'grok' and 'groq' are distinct providers — alias is gone.
  6. New providers have retains_data=None → 'unverified' in retention labels.
  7. Fallback chain includes grok + nvidia as real candidates.
  8. Auto-detection priority when credentials are set.
  9. PROVIDERS registry contains all 6 providers.
  10. factory.create_llm_provider dispatches correctly for new providers.

All tests mock network calls; no live API requests are made.
"""
import os
import urllib.request
from unittest.mock import MagicMock, patch

import pytest

from sentinelai.ai.config import (
    DEFAULT_PROVIDER_PRIORITY,
    PROVIDER_ALIASES,
    PROVIDERS,
    UNIMPLEMENTED_PROVIDERS,
    resolve_config,
)
from sentinelai.ai.llm.fallback import (
    DEFAULT_PROVIDER_PRIORITY as FALLBACK_DEFAULT_PRIORITY,
    PROVIDER_ALIASES as FALLBACK_ALIASES,
    UNIMPLEMENTED_PROVIDERS as FALLBACK_UNIMPLEMENTED,
    ProviderCandidate,
    create_fallback_provider,
)
from sentinelai.ai.llm.base import LLMProvider, LLMResponse
from sentinelai.ai.llm.groq import GroqProvider
from sentinelai.ai.llm.nvidia import API_KEY_ENV as NVIDIA_KEY_ENV
from sentinelai.ai.llm.nvidia import DEFAULT_MODEL as NVIDIA_DEFAULT_MODEL
from sentinelai.ai.llm.nvidia import ENDPOINT as NVIDIA_ENDPOINT
from sentinelai.ai.llm.nvidia import NvidiaNimProvider
from sentinelai.ai.llm.xai import API_KEY_ENV as XAI_KEY_ENV
from sentinelai.ai.llm.xai import DEFAULT_MODEL as XAI_DEFAULT_MODEL
from sentinelai.ai.llm.xai import ENDPOINT as XAI_ENDPOINT
from sentinelai.ai.llm.xai import GrokProvider
from sentinelai.contracts import AIConfig
from sentinelai.contracts.scan_result import retention_status_from_provider

# A plausible-looking fake key (not a real secret)
FAKE_KEY = "sk-fake-key-for-testing-not-real"


# ==============================================================================
# 1. GrokProvider (xAI) identity
# ==============================================================================


def test_grok_provider_name():
    assert GrokProvider.name == "grok"


def test_grok_provider_is_not_local():
    assert GrokProvider.is_local is False


def test_grok_provider_endpoint():
    assert XAI_ENDPOINT == "https://api.x.ai/v1/chat/completions"


def test_grok_provider_api_key_env():
    assert XAI_KEY_ENV == "XAI_API_KEY"


def test_grok_provider_default_model():
    assert XAI_DEFAULT_MODEL == "grok-4"


def test_grok_provider_retains_data_is_none():
    assert GrokProvider.retains_data is None


def test_grok_provider_constructs_with_xai_key(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", FAKE_KEY)
    p = GrokProvider("grok-4")
    assert p.model == "grok-4"
    # Key must NOT appear in repr
    assert FAKE_KEY not in repr(p)


def test_grok_provider_raises_without_xai_key(monkeypatch):
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    from sentinelai.ai.llm.errors import ProviderError
    with pytest.raises(ProviderError, match="XAI_API_KEY"):
        GrokProvider("grok-4")


# ==============================================================================
# 2. NvidiaNimProvider identity
# ==============================================================================


def test_nvidia_provider_name():
    assert NvidiaNimProvider.name == "nvidia"


def test_nvidia_provider_is_not_local():
    assert NvidiaNimProvider.is_local is False


def test_nvidia_provider_endpoint():
    assert NVIDIA_ENDPOINT == "https://integrate.api.nvidia.com/v1/chat/completions"


def test_nvidia_provider_api_key_env():
    assert NVIDIA_KEY_ENV == "NVIDIA_API_KEY"


def test_nvidia_provider_default_model():
    assert NVIDIA_DEFAULT_MODEL == "meta/llama-3.1-70b-instruct"


def test_nvidia_provider_retains_data_is_none():
    assert NvidiaNimProvider.retains_data is None


def test_nvidia_provider_constructs_with_key(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", FAKE_KEY)
    p = NvidiaNimProvider("meta/llama-3.1-70b-instruct")
    assert p.model == "meta/llama-3.1-70b-instruct"
    assert FAKE_KEY not in repr(p)


def test_nvidia_provider_raises_without_key(monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    from sentinelai.ai.llm.errors import ProviderError
    with pytest.raises(ProviderError, match="NVIDIA_API_KEY"):
        NvidiaNimProvider("meta/llama-3.1-70b-instruct")


# ==============================================================================
# 3. GroqProvider does NOT read XAI_API_KEY or GROK_API_KEY
# ==============================================================================


def test_groq_provider_does_not_read_xai_api_key(monkeypatch):
    """GroqProvider must not be constructible via XAI_API_KEY."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("XAI_API_KEY", FAKE_KEY)
    from sentinelai.ai.llm.errors import ProviderError
    with pytest.raises(ProviderError, match="GROQ_API_KEY"):
        GroqProvider("llama-3.3-70b-versatile")


def test_groq_provider_does_not_accept_grok_api_key(monkeypatch):
    """The old GROK_API_KEY alias for Groq is gone from the provider itself."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("GROK_API_KEY", FAKE_KEY)
    from sentinelai.ai.llm.errors import ProviderError
    # With GROK_API_KEY set but GROQ_API_KEY absent, GroqProvider should raise
    with pytest.raises(ProviderError, match="GROQ_API_KEY"):
        GroqProvider("llama-3.3-70b-versatile")


# ==============================================================================
# 4. Provider priority order: Gemini -> Grok -> NVIDIA -> Ollama
# ==============================================================================


def test_config_default_priority_order():
    assert DEFAULT_PROVIDER_PRIORITY == ("gemini", "grok", "nvidia", "ollama")


def test_fallback_default_priority_order():
    assert FALLBACK_DEFAULT_PRIORITY == ("gemini", "grok", "nvidia", "ollama")


# ==============================================================================
# 5. 'grok' and 'groq' are distinct providers — alias is gone
# ==============================================================================


def test_no_grok_groq_alias_in_config():
    assert "grok" not in PROVIDER_ALIASES
    assert PROVIDER_ALIASES == {}


def test_no_grok_groq_alias_in_fallback():
    assert "grok" not in FALLBACK_ALIASES
    assert FALLBACK_ALIASES == {}


def test_grok_resolves_to_grok_not_groq(monkeypatch):
    """resolve_config({'provider': 'grok'}) must produce provider='grok', not 'groq'."""
    monkeypatch.setenv("XAI_API_KEY", FAKE_KEY)
    cfg = resolve_config({"provider": "grok", "egress": "allow"})
    assert cfg.provider == "grok"


def test_groq_provider_identity_unchanged(monkeypatch):
    """'groq' must still work independently of the 'grok' changes."""
    monkeypatch.setenv("GROQ_API_KEY", FAKE_KEY)
    cfg = resolve_config({"provider": "groq", "egress": "allow"})
    assert cfg.provider == "groq"


# ==============================================================================
# 6. New providers have retains_data=None → 'unverified'
# ==============================================================================


def test_grok_retention_is_unverified():
    label = retention_status_from_provider(GrokProvider.retains_data)
    assert label == "unverified"


def test_nvidia_retention_is_unverified():
    label = retention_status_from_provider(NvidiaNimProvider.retains_data)
    assert label == "unverified"


def test_grok_retains_data_not_silently_false():
    """retains_data=None must NOT be converted to False."""
    assert GrokProvider.retains_data is not False


def test_nvidia_retains_data_not_silently_false():
    assert NvidiaNimProvider.retains_data is not False


# ==============================================================================
# 7. Fallback chain includes grok + nvidia as real candidates
# ==============================================================================


def test_fallback_chain_includes_grok_and_nvidia(monkeypatch):
    """create_fallback_provider builds candidates for grok and nvidia (both implemented)."""
    monkeypatch.setenv("XAI_API_KEY", FAKE_KEY)
    monkeypatch.setenv("NVIDIA_API_KEY", FAKE_KEY)
    cfg = AIConfig(
        provider="gemini",
        model="gemini-2.5-flash",
        egress="allow",
        fallback_providers=["grok", "nvidia", "ollama"],
    )
    fb = create_fallback_provider(cfg)
    names = [c.name for c in fb.candidates]
    assert "grok" in names
    assert "nvidia" in names
    assert "ollama" in names


def test_fallback_chain_grok_uses_xai_not_groq_endpoint(monkeypatch):
    """The 'grok' candidate in a fallback chain must use the xAI endpoint, not Groq's."""
    monkeypatch.setenv("XAI_API_KEY", FAKE_KEY)
    cfg = AIConfig(
        provider="grok",
        model=XAI_DEFAULT_MODEL,
        egress="allow",
        fallback_providers=["ollama"],
    )
    fb = create_fallback_provider(cfg)
    grok_cand = next(c for c in fb.candidates if c.name == "grok")
    instance = grok_cand.factory()
    assert isinstance(instance, GrokProvider)
    assert instance.ENDPOINT == XAI_ENDPOINT


def test_unimplemented_providers_set_is_empty():
    assert UNIMPLEMENTED_PROVIDERS == set()
    assert FALLBACK_UNIMPLEMENTED == set()


# ==============================================================================
# 8. Auto-detection priority when credentials are present
# ==============================================================================


def test_auto_selects_gemini_first_when_credentials_present(monkeypatch):
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/path/to/sa.json")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "my-project")
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    cfg = resolve_config({"egress": "allow"})
    assert cfg.provider == "gemini"


def test_auto_selects_grok_when_only_xai_key_present(monkeypatch):
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.setenv("XAI_API_KEY", FAKE_KEY)
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    cfg = resolve_config({"egress": "allow"})
    assert cfg.provider == "grok"


def test_auto_selects_nvidia_when_only_nvidia_key_present(monkeypatch):
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.setenv("NVIDIA_API_KEY", FAKE_KEY)
    cfg = resolve_config({"egress": "allow"})
    assert cfg.provider == "nvidia"


def test_auto_falls_back_to_ollama_when_no_remote_credentials(monkeypatch):
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    cfg = resolve_config({"egress": "allow"})
    assert cfg.provider == "ollama"


# ==============================================================================
# 9. PROVIDERS registry contains all 6 providers
# ==============================================================================


def test_providers_registry_has_all_six():
    assert set(PROVIDERS) == {"ollama", "groq", "gemini", "mock", "grok", "nvidia"}


def test_providers_registry_grok_spec():
    spec = PROVIDERS["grok"]
    assert spec.provider_class is GrokProvider
    assert spec.default_model == "grok-4"
    assert spec.required_env == ("XAI_API_KEY",)
    assert not spec.is_local


def test_providers_registry_nvidia_spec():
    spec = PROVIDERS["nvidia"]
    assert spec.provider_class is NvidiaNimProvider
    assert spec.default_model == "meta/llama-3.1-70b-instruct"
    assert spec.required_env == ("NVIDIA_API_KEY",)
    assert not spec.is_local


def test_providers_registry_groq_spec_unchanged():
    from sentinelai.ai.llm.groq import API_KEY_ENV as GROQ_KEY_ENV
    spec = PROVIDERS["groq"]
    assert spec.provider_class is GroqProvider
    assert spec.default_model == "llama-3.3-70b-versatile"
    assert spec.required_env == (GROQ_KEY_ENV,)
    assert not spec.is_local


# ==============================================================================
# 10. factory.create_llm_provider dispatches for new providers
# ==============================================================================


def test_factory_dispatches_grok(monkeypatch):
    """create_llm_provider with provider='grok' returns a GrokProvider."""
    monkeypatch.setenv("XAI_API_KEY", FAKE_KEY)
    from sentinelai.ai.factory import create_llm_provider
    cfg = AIConfig(provider="grok", model="grok-4", egress="allow")
    provider = create_llm_provider(cfg)
    assert isinstance(provider, GrokProvider)


def test_factory_dispatches_nvidia(monkeypatch):
    """create_llm_provider with provider='nvidia' returns a NvidiaNimProvider."""
    monkeypatch.setenv("NVIDIA_API_KEY", FAKE_KEY)
    from sentinelai.ai.factory import create_llm_provider
    cfg = AIConfig(provider="nvidia", model="meta/llama-3.1-70b-instruct", egress="allow")
    provider = create_llm_provider(cfg)
    assert isinstance(provider, NvidiaNimProvider)


def test_factory_groq_still_works(monkeypatch):
    """create_llm_provider with provider='groq' still returns a GroqProvider."""
    monkeypatch.setenv("GROQ_API_KEY", FAKE_KEY)
    from sentinelai.ai.factory import create_llm_provider
    cfg = AIConfig(provider="groq", model="llama-3.3-70b-versatile", egress="allow")
    provider = create_llm_provider(cfg)
    assert isinstance(provider, GroqProvider)


# ==============================================================================
# 11. Role separation and independent verification architecture
# ==============================================================================


class _FakeInnerProvider(LLMProvider):
    def __init__(self, outer, name: str, is_local: bool, model: str):
        self.outer = outer
        self.name = name
        self.is_local = is_local
        self.retains_data = None if not is_local else False
        self.model = model

    def complete(self, prompt, system=None, max_tokens=2048, temperature=0.1):
        self.outer.call_count += 1
        if self.outer._err:
            raise self.outer._err
        return LLMResponse(text=self.outer._resp, model=self.model)


class _FakeRoleCandidate(ProviderCandidate):
    def __init__(self, name: str, is_local: bool = False, model: str = "test-model", response_text: str = '{"test": "ok"}', error_to_raise=None):
        self._resp = response_text
        self._err = error_to_raise
        self.call_count = 0
        cfg = AIConfig(provider=name if name in ("ollama", "groq", "gemini", "mock", "grok", "nvidia") else "mock", model=model, egress="allow")

        super().__init__(
            name=name,
            model=model,
            is_local=is_local,
            retains_data=None if not is_local else False,
            config=cfg,
            factory=lambda: _FakeInnerProvider(self, name, is_local, model),
        )


def test_role_aware_generator_role_separation():
    """Discovery uses Gemini; verification uses Grok; remediation uses Gemini."""
    from sentinelai.ai.llm.roles import RoleAwareGenerator

    cfg = AIConfig(provider="gemini", model="gemini-2.5-flash", egress="allow")
    gen = RoleAwareGenerator(cfg, role_chains={
        "discovery": ["gemini"],
        "verification": ["grok"],
        "remediation": ["gemini"],
    })

    cand_gemini_disc = _FakeRoleCandidate("gemini", response_text='{"stage": "discovery"}')
    cand_grok_ver = _FakeRoleCandidate("grok", response_text='{"stage": "verification"}')
    cand_gemini_rem = _FakeRoleCandidate("gemini", response_text='{"stage": "remediation"}')

    gen.candidates_by_role["discovery"] = [cand_gemini_disc]
    gen.candidates_by_role["verification"] = [cand_grok_ver]
    gen.candidates_by_role["remediation"] = [cand_gemini_rem]

    # Call discovery
    r1 = gen.complete_for_role("discovery", "prompt 1")
    assert r1.text == '{"stage": "discovery"}'
    assert gen.active_discovery_provider == "gemini"

    # Call remediation
    r2 = gen.complete_for_role("remediation", "prompt 2")
    assert r2.text == '{"stage": "remediation"}'

    # Call verification (independent verifier Grok)
    r3 = gen.complete_for_role("verification", "prompt 3")
    assert r3.text == '{"stage": "verification"}'

    assert cand_gemini_disc.call_count == 1
    assert cand_grok_ver.call_count == 1
    assert cand_gemini_rem.call_count == 1


def test_independent_verification_skips_discovery_provider():
    """If Gemini ran discovery, Gemini CANNOT be used as verifier; fallback to next provider."""
    from sentinelai.ai.llm.roles import RoleAwareGenerator

    cfg = AIConfig(provider="gemini", model="gemini-2.5-flash", egress="allow")
    gen = RoleAwareGenerator(cfg)

    # Verification chain has Gemini, then Grok
    cand_gemini_ver = _FakeRoleCandidate("gemini", response_text='{"ver": "gemini"}')
    cand_grok_ver = _FakeRoleCandidate("grok", response_text='{"ver": "grok"}')
    gen.candidates_by_role["verification"] = [cand_gemini_ver, cand_grok_ver]

    # Set active discovery provider to Gemini
    gen.active_discovery_provider = "gemini"

    # Complete verification: Gemini must be SKIPPED, Grok must be called
    resp = gen.complete_for_role("verification", "verify this")
    assert resp.text == '{"ver": "grok"}'
    assert cand_gemini_ver.call_count == 0  # Gemini was NEVER called for verification!
    assert cand_grok_ver.call_count == 1


def test_independent_verification_fails_closed_when_no_distinct_provider():
    """If only the discovery provider is in the verification chain, verification raises."""
    from sentinelai.ai.llm.roles import IndependentVerificationUnavailableError, RoleAwareGenerator

    cfg = AIConfig(provider="gemini", model="gemini-2.5-flash", egress="allow")
    gen = RoleAwareGenerator(cfg)

    cand_gemini = _FakeRoleCandidate("gemini", response_text='{"ver": "gemini"}')
    gen.candidates_by_role["verification"] = [cand_gemini]
    gen.active_discovery_provider = "gemini"

    with pytest.raises(IndependentVerificationUnavailableError) as excinfo:
        gen.complete_for_role("verification", "verify this")

    assert "discovery used 'gemini'" in str(excinfo.value)
    assert cand_gemini.call_count == 0


def test_role_provenance_records_distinct_entries_for_each_role():
    """AIRoleProvenance captures separate (role, provider, model) entries with retention."""
    from sentinelai.ai.llm.roles import RoleAwareGenerator

    cfg = AIConfig(provider="gemini", model="gemini-2.5-flash", egress="allow")
    gen = RoleAwareGenerator(cfg)

    cand_disc = _FakeRoleCandidate("gemini", model="gemini-2.5-flash", response_text="disc")
    cand_ver = _FakeRoleCandidate("grok", model="grok-4", response_text="ver")
    cand_rem = _FakeRoleCandidate("nvidia", model="meta/llama-3.1-70b-instruct", response_text="rem")

    gen.candidates_by_role["discovery"] = [cand_disc]
    gen.candidates_by_role["verification"] = [cand_ver]
    gen.candidates_by_role["remediation"] = [cand_rem]

    gen.complete_for_role("discovery", "d")
    gen.complete_for_role("remediation", "r")
    gen.complete_for_role("verification", "v")

    prov_list = gen.get_role_provenance()
    assert len(prov_list) == 3

    disc_prov = next(p for p in prov_list if p.role == "discovery")
    assert disc_prov.provider == "gemini"
    assert disc_prov.model == "gemini-2.5-flash"
    assert disc_prov.retention_status == "unverified"
    assert disc_prov.completed_requests == 1

    ver_prov = next(p for p in prov_list if p.role == "verification")
    assert ver_prov.provider == "grok"
    assert ver_prov.model == "grok-4"
    assert ver_prov.retention_status == "unverified"
    assert ver_prov.completed_requests == 1

    rem_prov = next(p for p in prov_list if p.role == "remediation")
    assert rem_prov.provider == "nvidia"
    assert rem_prov.model == "meta/llama-3.1-70b-instruct"
    assert rem_prov.retention_status == "unverified"
    assert rem_prov.completed_requests == 1


def test_role_generator_fails_closed_on_secret_detection():
    """EgressViolation for secrets never falls through to another provider."""
    from sentinelai.ai.egress import EgressViolation
    from sentinelai.ai.llm.roles import RoleAwareGenerator

    cfg = AIConfig(provider="gemini", model="gemini-2.5-flash", egress="allow")
    gen = RoleAwareGenerator(cfg)

    cand1 = _FakeRoleCandidate("gemini")
    cand2 = _FakeRoleCandidate("grok")
    gen.candidates_by_role["discovery"] = [cand1, cand2]

    # Valid secret pattern (github personal access token: ghp_ + 36 alphanumeric)
    secret_prompt = "ghp_" + "A" * 36
    with pytest.raises(EgressViolation) as excinfo:
        gen.complete_for_role("discovery", secret_prompt)

    assert excinfo.value.check == "secrets"
    # Neither candidate was actually called because egress guard caught the secret
    assert cand1.call_count == 0
    assert cand2.call_count == 0


def test_pipeline_independent_verification_unavailable_sets_unverified():
    """When verifier is the same as discovery, pipeline marks verification as UNVERIFIED, not VERIFIED."""
    from sentinelai.ai.confidence_scorer import score_confidence
    from sentinelai.ai.llm.roles import IndependentVerificationUnavailableError
    from sentinelai.ai.pipeline import _enrich_one
    from sentinelai.ai.repository_context import RepositoryContext
    from sentinelai.ai.retrieval import RetrievedChunk, Retriever
    from sentinelai.ai.verifier import verify_finding
    from sentinelai.contracts import (
        ConfidenceLabel,
        GroundingVerdict,
        ScannerFinding,
        Severity,
        VerificationStatus,
    )

    finding = ScannerFinding(
        finding_id="FIND-1",
        scanner="semgrep",
        category="sql-injection",
        severity=Severity.HIGH,
        rule_id="r1",
        message="SQLi",
        raw_evidence="SELECT * FROM users",
    )

    class _R(Retriever):
        def retrieve(self, q, k):
            return [RetrievedChunk(kb_entry_id="KB1", text="sql injection", references=["CWE-89"], score=1.0)]

    class _Settings:
        retrieval_top_k = 1
        enable_verification = True

    # Discovery and remediation generate valid JSON; verification raises IndependentVerificationUnavailableError
    import json
    evidence_json = json.dumps({
        "title": "SQL Injection",
        "interpretation": "Raw query concatenation with user input",
        "observed_evidence": ["SELECT statement"],
        "limitations": [],
        "groundedness": "supported",
    })
    exploit_json = json.dumps({
        "exploit": {
            "exploit_path": "Attacker injects SQL payload",
            "impact": "Data exfiltration",
            "required_assumptions": ["unauthenticated access"],
        },
        "remediation": {
            "remediation": "Use parameterized queries",
            "validation_steps": ["check prepared statement"],
            "patch_suggestion": "cursor.execute(sql, (param,))",
        },
    })

    def gen_discovery(p, **kw):
        return evidence_json

    def gen_remediation(p, **kw):
        return exploit_json

    def gen_verification(p, **kw):
        raise IndependentVerificationUnavailableError("gemini")

    role_generates = {
        "discovery": gen_discovery,
        "remediation": gen_remediation,
        "verification": gen_verification,
    }

    enriched = _enrich_one(
        finding,
        RepositoryContext(),
        _R(),
        gen_discovery,
        score_confidence,
        verify_finding,
        _Settings(),
        role_generates=role_generates,
    )

    # Verification must NOT be marked VERIFIED
    assert enriched.verification_status == VerificationStatus.UNVERIFIED
    # Critic verdict must be INSUFFICIENT_EVIDENCE explaining why
    assert enriched.grounding_verdict == GroundingVerdict.INSUFFICIENT_EVIDENCE
    assert any("Independent verification unavailable" in c for c in enriched.unsupported_claims)


def test_enrich_findings_end_to_end_with_role_aware_generator(monkeypatch):
    """enrich_findings() works end-to-end with a real RoleAwareGenerator instance (for_role accepts response_schema and returns str)."""
    import json
    from sentinelai.ai.confidence_scorer import score_confidence
    from sentinelai.ai.llm.roles import RoleAwareGenerator
    from sentinelai.ai.pipeline import enrich_findings
    from sentinelai.ai.repository_context import RepositoryContext
    from sentinelai.ai.retrieval import RetrievedChunk, Retriever
    from sentinelai.ai.verifier import verify_finding
    from sentinelai.contracts import GroundingVerdict, ScannerFinding, Severity

    monkeypatch.setenv("SENTINELAI_AI_LLM_MODEL", "llama3.1:8b")
    monkeypatch.setenv("SENTINELAI_AI_EMBEDDING_MODEL", "nomic-embed-text")
    from sentinelai.ai import config as ai_config_mod
    monkeypatch.setattr(ai_config_mod, "_settings", None)

    finding = ScannerFinding(
        finding_id="FIND-ROLE-E2E",
        scanner="semgrep",
        category="sql-injection",
        severity=Severity.HIGH,
        rule_id="r1",
        message="SQLi in query",
        raw_evidence="SELECT * FROM users WHERE id = " + "user_id",
    )

    class _R(Retriever):
        def retrieve(self, q, k):
            return [RetrievedChunk(kb_entry_id="KB1", text="sql injection", references=["CWE-89"], score=1.0)]

    evidence_json = json.dumps({
        "title": "SQL Injection in user lookup query",
        "interpretation": "The query concatenates untrusted user_id directly into the SQL SELECT statement.",
        "observed_evidence": ["SELECT * FROM users WHERE id = user_id"],
        "limitations": [],
        "groundedness": "supported",
    })
    exploit_json = json.dumps({
        "exploit": {
            "exploit_path": "An attacker supplies a crafted user_id string to alter the SQL query logic.",
            "impact": "Unauthorized read access to all rows in the users table.",
            "required_assumptions": ["user_id is controlled by an external request without prior integer cast"],
        },
        "remediation": {
            "remediation": "Replace string concatenation with a parameterized query placeholder.",
            "validation_steps": ["Verify that database driver binds user_id as a parameter rather than SQL syntax."],
            "patch_suggestion": "cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))",
        },
    })
    critic_json = json.dumps({
        "verdict": "supported",
        "supported_claims": ["The SQL query concatenates user_id directly into the SELECT statement."],
        "unsupported_claims": [],
    })

    cfg = AIConfig(provider="gemini", model="gemini-2.5-flash", egress="allow")
    gen = RoleAwareGenerator(cfg)
    cand_disc = _FakeRoleCandidate("gemini", model="gemini-2.5-flash", response_text=evidence_json)
    cand_rem = _FakeRoleCandidate("nvidia", model="meta/llama-3.1-70b-instruct", response_text=exploit_json)
    cand_ver = _FakeRoleCandidate("grok", model="grok-4", response_text=critic_json)

    gen.candidates_by_role["discovery"] = [cand_disc]
    gen.candidates_by_role["remediation"] = [cand_rem]
    gen.candidates_by_role["verification"] = [cand_ver]

    results = enrich_findings(
        [finding],
        _R(),
        gen,
        score_confidence,
        verify_finding,
        repository_context=RepositoryContext(),
    )
    assert len(results) == 1
    assert results[0].grounding_verdict == GroundingVerdict.SUPPORTED
    assert cand_disc.call_count == 1
    assert cand_rem.call_count == 1
    assert cand_ver.call_count == 1

    # Also verify direct __call__ returns str and accepts response_schema
    direct_out = gen("direct prompt", response_schema=None)
    assert isinstance(direct_out, str)
    assert direct_out == evidence_json


def test_role_aware_generator_no_silent_provider_substitution_on_empty_or_unknown_role():
    """RoleAwareGenerator must not silently substitute cfg.provider when a role has no candidates or is unknown."""
    from sentinelai.ai.llm.base import ProviderError
    from sentinelai.ai.llm.roles import RoleAwareGenerator

    cfg = AIConfig(provider="gemini", model="gemini-2.5-flash", egress="allow")
    gen = RoleAwareGenerator(cfg, role_chains={"discovery": [], "remediation": ["gemini"], "verification": ["grok"]})

    with pytest.raises(ProviderError, match="no provider candidates configured for role 'discovery'"):
        gen.complete_for_role("discovery", "prompt")

    with pytest.raises(ProviderError, match="unknown AI role 'nonexistent'"):
        gen.complete_for_role("nonexistent", "prompt")

    with pytest.raises(ProviderError, match="unknown provider 'bogus'"):
        RoleAwareGenerator(cfg, role_chains={"discovery": ["bogus"]})


def test_verification_without_recorded_discovery_provider_fails_closed():
    """Verification must fail closed with IndependentVerificationUnavailableError if discovery provider was not recorded."""
    from sentinelai.ai.llm.roles import IndependentVerificationUnavailableError, RoleAwareGenerator

    cfg = AIConfig(provider="gemini", model="gemini-2.5-flash", egress="allow")
    gen = RoleAwareGenerator(cfg)
    cand_grok = _FakeRoleCandidate("grok", response_text='{"ver": "grok"}')
    gen.candidates_by_role["verification"] = [cand_grok]

    # active_discovery_provider is None
    with pytest.raises(IndependentVerificationUnavailableError, match="no discovery provider was recorded"):
        gen.complete_for_role("verification", "verify prompt")
    assert cand_grok.call_count == 0


def test_discovery_failure_resets_previous_finding_discovery_state():
    """If finding 2's discovery fails, finding 1's active_discovery_provider must not remain set."""
    from sentinelai.ai.llm.base import ProviderError
    from sentinelai.ai.llm.errors import ProviderErrorKind
    from sentinelai.ai.llm.roles import RoleAwareGenerator

    cfg = AIConfig(provider="gemini", model="gemini-2.5-flash", egress="allow")
    gen = RoleAwareGenerator(cfg)
    cand_ok = _FakeRoleCandidate("gemini", response_text="ok")
    gen.candidates_by_role["discovery"] = [cand_ok]

    gen.complete_for_role("discovery", "finding 1")
    assert gen.active_discovery_provider == "gemini"

    cand_fail = _FakeRoleCandidate(
        "gemini",
        error_to_raise=ProviderError("gemini", ProviderErrorKind.HTTP, "503 unavailable"),
    )
    gen.candidates_by_role["discovery"] = [cand_fail]

    with pytest.raises(ProviderError):
        gen.complete_for_role("discovery", "finding 2")

    assert gen.active_discovery_provider is None
    assert gen.active_discovery_model is None


def test_verification_when_all_non_discovery_candidates_ineligible_marks_unverified(monkeypatch):
    """When discovery uses local ollama under egress=deny and all other verifiers are remote, verification raises IndependentVerificationUnavailableError and pipeline marks UNVERIFIED."""
    import json
    from sentinelai.ai.confidence_scorer import score_confidence
    from sentinelai.ai.llm.roles import IndependentVerificationUnavailableError, RoleAwareGenerator
    from sentinelai.ai.pipeline import enrich_findings
    from sentinelai.ai.repository_context import RepositoryContext
    from sentinelai.ai.retrieval import RetrievedChunk, Retriever
    from sentinelai.ai.verifier import verify_finding
    from sentinelai.contracts import GroundingVerdict, ScannerFinding, Severity, VerificationStatus

    monkeypatch.setenv("SENTINELAI_AI_LLM_MODEL", "llama3.1:8b")
    monkeypatch.setenv("SENTINELAI_AI_EMBEDDING_MODEL", "nomic-embed-text")
    from sentinelai.ai import config as ai_config_mod
    monkeypatch.setattr(ai_config_mod, "_settings", None)

    evidence_json = json.dumps({
        "title": "SQL Injection in user lookup query",
        "interpretation": "The query concatenates untrusted user_id directly into the SQL SELECT statement.",
        "observed_evidence": ["SELECT * FROM users WHERE id = user_id"],
        "limitations": [],
        "groundedness": "supported",
    })
    exploit_json = json.dumps({
        "exploit": {
            "exploit_path": "An attacker supplies a crafted user_id string to alter the SQL query logic.",
            "impact": "Unauthorized read access to all rows in the users table.",
            "required_assumptions": ["user_id is controlled by an external request without prior integer cast"],
        },
        "remediation": {
            "remediation": "Replace string concatenation with a parameterized query placeholder.",
            "validation_steps": ["Verify that database driver binds user_id as a parameter rather than SQL syntax."],
            "patch_suggestion": "cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))",
        },
    })

    cfg = AIConfig(provider="ollama", model="llama3.1:8b", egress="deny")
    gen = RoleAwareGenerator(cfg)
    cand_ollama_disc = _FakeRoleCandidate("ollama", is_local=True, model="llama3.1:8b", response_text=evidence_json)
    cand_ollama_rem = _FakeRoleCandidate("ollama", is_local=True, model="llama3.1:8b", response_text=exploit_json)
    cand_grok_ver = _FakeRoleCandidate("grok", is_local=False, model="grok-4", response_text="should not run")
    cand_ollama_ver = _FakeRoleCandidate("ollama", is_local=True, model="llama3.1:8b", response_text="should not run")

    gen.candidates_by_role["discovery"] = [cand_ollama_disc]
    gen.candidates_by_role["remediation"] = [cand_ollama_rem]
    gen.candidates_by_role["verification"] = [cand_grok_ver, cand_ollama_ver]

    finding = ScannerFinding(
        finding_id="FIND-LOCAL-ONLY",
        scanner="semgrep",
        category="sql-injection",
        severity=Severity.HIGH,
        rule_id="r1",
        message="SQLi in query",
        raw_evidence="SELECT * FROM users WHERE id = user_id",
    )

    class _R(Retriever):
        def retrieve(self, q, k):
            return [RetrievedChunk(kb_entry_id="KB1", text="sql injection", references=["CWE-89"], score=1.0)]

    results = enrich_findings(
        [finding],
        _R(),
        gen,
        score_confidence,
        verify_finding,
        repository_context=RepositoryContext(),
    )
    assert len(results) == 1
    assert results[0].verification_status == VerificationStatus.UNVERIFIED
    assert results[0].grounding_verdict == GroundingVerdict.INSUFFICIENT_EVIDENCE
    assert cand_grok_ver.call_count == 0
    assert cand_ollama_ver.call_count == 0


def test_role_aware_generator_deduplicates_candidates_and_bounds_attempts():
    """Duplicate provider names in a role chain are deduplicated and attempted at most once."""
    from sentinelai.ai.llm.base import ProviderError
    from sentinelai.ai.llm.errors import ProviderErrorKind
    from sentinelai.ai.llm.roles import RoleAwareGenerator

    cfg = AIConfig(provider="gemini", model="gemini-2.5-flash", egress="allow")
    gen = RoleAwareGenerator(cfg, role_chains={"discovery": ["gemini", "gemini", "grok"]})
    assert [c.name for c in gen.candidates_by_role["discovery"]] == ["gemini", "grok"]

    # Even if duplicate candidate objects are injected into candidates_by_role, complete_for_role bounds attempts
    cand_fail = _FakeRoleCandidate(
        "gemini",
        error_to_raise=ProviderError("gemini", ProviderErrorKind.HTTP, "500 error"),
    )
    gen.candidates_by_role["discovery"] = [cand_fail, cand_fail]
    with pytest.raises(ProviderError):
        gen.complete_for_role("discovery", "test prompt")
    assert cand_fail.call_count == 1


def test_egress_policy_violation_in_assert_safe_fails_closed():
    """An EgressViolation(check='policy') from assert_safe fails closed immediately without falling back."""
    from sentinelai.ai.egress import EgressViolation
    from sentinelai.ai.llm.roles import RoleAwareGenerator

    cfg = AIConfig(provider="gemini", model="gemini-2.5-flash", egress="allow")
    gen = RoleAwareGenerator(cfg)

    # Candidate 1 has egress="deny" in its own config while candidate 2 has egress="allow"
    cand1 = _FakeRoleCandidate("gemini", response_text="gemini")
    cand1.config = AIConfig(provider="gemini", model="gemini-2.5-flash", egress="deny")
    cand2 = _FakeRoleCandidate("grok", response_text="grok")
    gen.candidates_by_role["discovery"] = [cand1, cand2]

    with pytest.raises(EgressViolation) as excinfo:
        gen.complete_for_role("discovery", "clean prompt")
    assert excinfo.value.check == "policy"
    assert cand1.call_count == 0
    assert cand2.call_count == 0


def test_resolve_config_parses_and_validates_role_chain_env_vars(monkeypatch):
    """SENTINELAI_AI_ROLE_* env vars populate AIConfig.role_chains and reject unknown providers."""
    from sentinelai.ai.config import AIConfigError

    monkeypatch.setenv("SENTINELAI_AI_ROLE_DISCOVERY", "gemini,ollama")
    monkeypatch.setenv("SENTINELAI_AI_ROLE_VERIFICATION", "grok,ollama")
    cfg = resolve_config({})
    assert cfg.role_chains["discovery"] == ["gemini", "ollama"]
    assert cfg.role_chains["verification"] == ["grok", "ollama"]
    assert cfg.role_chains["remediation"] == ["gemini", "nvidia", "grok", "ollama"]

    monkeypatch.setenv("SENTINELAI_AI_ROLE_DISCOVERY", "gemini,unknown_prov")
    with pytest.raises(AIConfigError, match="SENTINELAI_AI_ROLE_DISCOVERY must contain only"):
        resolve_config({})


def test_cli_role_provenance_conservative_scan_retention_status(tmp_path, monkeypatch):
    """When local ollama (not_retained) runs discovery and remote grok (unverified) runs verification, scan-level retention_status is conservatively 'unverified'."""
    import json
    from sentinelai import main as main_module
    from sentinelai.ai import config as ai_config_mod
    from sentinelai.ai.llm.roles import RoleAwareGenerator
    from sentinelai.ai.retrieval import RetrievedChunk, Retriever

    monkeypatch.setenv("SENTINELAI_AI_LLM_MODEL", "llama3.1:8b")
    monkeypatch.setenv("SENTINELAI_AI_EMBEDDING_MODEL", "nomic-embed-text")
    monkeypatch.setenv("SENTINELAI_AI_ROLE_DISCOVERY", "ollama")
    monkeypatch.setenv("SENTINELAI_AI_ROLE_REMEDIATION", "ollama")
    monkeypatch.setenv("SENTINELAI_AI_ROLE_VERIFICATION", "grok")
    monkeypatch.setenv("XAI_API_KEY", FAKE_KEY)
    monkeypatch.setattr(ai_config_mod, "_settings", None)

    evidence_json = json.dumps({
        "title": "SQL Injection in user lookup query",
        "interpretation": "The query concatenates untrusted user_id directly into the SQL SELECT statement.",
        "observed_evidence": ["SELECT * FROM users WHERE id = user_id"],
        "limitations": [],
        "groundedness": "supported",
    })
    exploit_json = json.dumps({
        "exploit": {
            "exploit_path": "An attacker supplies a crafted user_id string to alter the SQL query logic.",
            "impact": "Unauthorized read access to all rows in the users table.",
            "required_assumptions": ["user_id is controlled by an external request without prior integer cast"],
        },
        "remediation": {
            "remediation": "Replace string concatenation with a parameterized query placeholder.",
            "validation_steps": ["Verify that database driver binds user_id as a parameter rather than SQL syntax."],
            "patch_suggestion": "cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))",
        },
    })
    critic_json = json.dumps({
        "verdict": "supported",
        "supported_claims": ["The SQL query concatenates user_id directly into the SELECT statement."],
        "unsupported_claims": [],
    })

    def fake_create_role_generator(cfg, max_tokens=2048, temperature=0.1):
        gen = RoleAwareGenerator(cfg, max_tokens=max_tokens, temperature=temperature)
        gen.candidates_by_role["discovery"] = [
            _FakeRoleCandidate("ollama", is_local=True, model="llama3.1:8b", response_text=evidence_json)
        ]
        gen.candidates_by_role["remediation"] = [
            _FakeRoleCandidate("ollama", is_local=True, model="llama3.1:8b", response_text=exploit_json)
        ]
        gen.candidates_by_role["verification"] = [
            _FakeRoleCandidate("grok", is_local=False, model="grok-4", response_text=critic_json)
        ]
        return gen

    class _R(Retriever):
        def retrieve(self, q, k):
            return [RetrievedChunk(kb_entry_id="KB1", text="sql injection", references=["CWE-89"], score=1.0)]

    from typer.testing import CliRunner
    from sentinelai.ai import factory as factory_mod
    from sentinelai.providers.mock_provider import MockFindingsProvider
    monkeypatch.setattr(main_module, "_get_provider", lambda: MockFindingsProvider())
    monkeypatch.setattr(factory_mod, "create_role_generator", fake_create_role_generator)
    monkeypatch.setattr(main_module, "create_retriever_for_scan", lambda: _R())

    res = CliRunner().invoke(
        main_module.app,
        ["scan", str(tmp_path), "--format", "json", "--egress", "allow"],
    )
    assert res.exit_code == 0, res.output
    report = json.loads(res.stdout)
    scan_meta = report["scan"]
    assert scan_meta["ai_provider"] == "ollama"
    assert scan_meta["retention_status"] == "unverified"
    role_provs = {rp["role"]: rp for rp in scan_meta["role_provenance"]}
    assert role_provs["discovery"]["provider"] == "ollama"
    assert role_provs["discovery"]["retention_status"] == "not_retained"
    assert role_provs["verification"]["provider"] == "grok"
    assert role_provs["verification"]["retention_status"] == "unverified"


# ==============================================================================
# 12. Default multi-role workflow, verifier prompt & disagreement, retention basis
# ==============================================================================


def test_resolve_config_populates_default_role_chains_without_role_env_vars(monkeypatch):
    """When SENTINELAI_AI_ROLE_* env vars are unset, resolve_config() still populates multi-role chains by default."""
    for env_name in (
        "SENTINELAI_AI_ROLE_DISCOVERY",
        "SENTINELAI_AI_ROLE_REMEDIATION",
        "SENTINELAI_AI_ROLE_VERIFICATION",
        "SENTINELAI_AI_PROVIDER",
    ):
        monkeypatch.delenv(env_name, raising=False)

    cfg = resolve_config({})
    assert cfg.role_chains == {
        "discovery": ["gemini", "grok", "nvidia", "ollama"],
        "verification": ["grok", "nvidia", "gemini", "ollama"],
        "remediation": ["gemini", "nvidia", "grok", "ollama"],
    }


def test_cli_ai_enabled_uses_role_aware_workflow_by_default_without_role_env_vars(tmp_path, monkeypatch):
    """When AI is enabled (--ai or configured models) and SENTINELAI_AI_ROLE_* are unset, scan() uses the role-aware workflow by default."""
    import json
    from typer.testing import CliRunner
    from sentinelai import main as main_module
    from sentinelai.ai import config as ai_config_mod
    from sentinelai.ai import factory as factory_mod
    from sentinelai.ai.llm.roles import RoleAwareGenerator
    from sentinelai.ai.retrieval import RetrievedChunk, Retriever
    from sentinelai.providers.mock_provider import MockFindingsProvider

    for env_name in (
        "SENTINELAI_AI_ROLE_DISCOVERY",
        "SENTINELAI_AI_ROLE_REMEDIATION",
        "SENTINELAI_AI_ROLE_VERIFICATION",
        "SENTINELAI_AI_PROVIDER",
    ):
        monkeypatch.delenv(env_name, raising=False)
    monkeypatch.setenv("SENTINELAI_AI_LLM_MODEL", "llama3.1:8b")
    monkeypatch.setenv("SENTINELAI_AI_EMBEDDING_MODEL", "nomic-embed-text")
    monkeypatch.setattr(ai_config_mod, "_settings", None)

    evidence_json = json.dumps({
        "title": "SQL Injection in user lookup query",
        "interpretation": "The query concatenates untrusted user_id directly into the SQL SELECT statement.",
        "observed_evidence": ["SELECT * FROM users WHERE id = user_id"],
        "limitations": [],
        "groundedness": "supported",
    })
    exploit_json = json.dumps({
        "exploit": {
            "exploit_path": "An attacker supplies a crafted user_id string to alter the SQL query logic.",
            "impact": "Unauthorized read access to all rows in the users table.",
            "required_assumptions": ["user_id is controlled by an external request"],
        },
        "remediation": {
            "remediation": "Replace string concatenation with a parameterized query placeholder.",
            "validation_steps": ["Verify that database driver binds user_id as a parameter."],
            "patch_suggestion": "cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))",
        },
    })
    critic_json = json.dumps({
        "verdict": "supported",
        "supported_claims": ["The SQL query concatenates user_id directly into the SELECT statement."],
        "unsupported_claims": [],
    })

    created_generators = []

    def fake_create_role_generator(cfg, max_tokens=2048, temperature=0.1):
        gen = RoleAwareGenerator(cfg, max_tokens=max_tokens, temperature=temperature)
        gen.candidates_by_role["discovery"] = [
            _FakeRoleCandidate("gemini", is_local=False, model="gemini-2.5-flash", response_text=evidence_json)
        ]
        gen.candidates_by_role["remediation"] = [
            _FakeRoleCandidate("gemini", is_local=False, model="gemini-2.5-flash", response_text=exploit_json)
        ]
        gen.candidates_by_role["verification"] = [
            _FakeRoleCandidate("grok", is_local=False, model="grok-4", response_text=critic_json)
        ]
        created_generators.append(gen)
        return gen

    class _R(Retriever):
        def retrieve(self, q, k):
            return [RetrievedChunk(kb_entry_id="KB1", text="sql injection", references=["CWE-89"], score=1.0)]

    monkeypatch.setattr(main_module, "_get_provider", lambda: MockFindingsProvider())
    monkeypatch.setattr(factory_mod, "create_role_generator", fake_create_role_generator)
    monkeypatch.setattr(main_module, "create_retriever_for_scan", lambda: _R())

    res = CliRunner().invoke(
        main_module.app,
        ["scan", str(tmp_path), "--ai", "--format", "json", "--egress", "allow"],
    )
    assert res.exit_code == 0, res.output
    assert len(created_generators) == 1
    report = json.loads(res.stdout)
    role_provs = {rp["role"]: rp for rp in report["scan"]["role_provenance"]}
    assert role_provs["discovery"]["provider"] == "gemini"
    assert role_provs["remediation"]["provider"] == "gemini"
    assert role_provs["verification"]["provider"] == "grok"


def test_discovery_and_verifier_disagreement_is_recorded_and_insufficient_verdict_never_verified():
    """When discovery claims 'supported' but independent verifier concludes 'insufficient_evidence', disagreement is recorded and verification_status is INSUFFICIENT_EVIDENCE (never VERIFIED)."""
    import json
    from sentinelai.ai.confidence_scorer import score_confidence
    from sentinelai.ai.llm.roles import RoleAwareGenerator
    from sentinelai.ai.pipeline import enrich_findings
    from sentinelai.ai.repository_context import RepositoryContext
    from sentinelai.ai.retrieval import RetrievedChunk, Retriever
    from sentinelai.ai.verifier import verify_finding
    from sentinelai.contracts import GroundingVerdict, ScannerFinding, Severity, VerificationStatus

    finding = ScannerFinding(
        finding_id="FIND-DISAGREE",
        scanner="semgrep",
        category="sql-injection",
        severity=Severity.HIGH,
        rule_id="r1",
        message="SQLi in query",
        raw_evidence="SELECT * FROM users WHERE id = user_id",
    )

    class _R(Retriever):
        def retrieve(self, q, k):
            return [RetrievedChunk(kb_entry_id="KB1", text="sql injection", references=["CWE-89"], score=1.0)]

    evidence_json = json.dumps({
        "title": "SQL Injection in user lookup query",
        "interpretation": "The query concatenates untrusted user_id directly into the SQL SELECT statement.",
        "observed_evidence": ["SELECT * FROM users WHERE id = user_id"],
        "limitations": [],
        "groundedness": "supported",
    })
    exploit_json = json.dumps({
        "exploit": {
            "exploit_path": "An attacker supplies a crafted user_id string to alter the SQL query logic.",
            "impact": "Unauthorized read access to all rows in the users table.",
            "required_assumptions": ["user_id is controlled by an external request"],
        },
        "remediation": {
            "remediation": "Replace string concatenation with a parameterized query placeholder.",
            "validation_steps": ["Verify that database driver binds user_id as a parameter."],
            "patch_suggestion": "cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))",
        },
    })
    critic_json = json.dumps({
        "verdict": "insufficient_evidence",
        "supported_claims": [],
        "unsupported_claims": [
            "The snippet does not show how user_id is constructed or whether it is already an integer."
        ],
    })

    cfg = AIConfig(provider="gemini", model="gemini-2.5-flash", egress="allow")
    gen = RoleAwareGenerator(cfg)
    gen.candidates_by_role["discovery"] = [
        _FakeRoleCandidate("gemini", model="gemini-2.5-flash", response_text=evidence_json)
    ]
    gen.candidates_by_role["remediation"] = [
        _FakeRoleCandidate("gemini", model="gemini-2.5-flash", response_text=exploit_json)
    ]
    gen.candidates_by_role["verification"] = [
        _FakeRoleCandidate("grok", model="grok-4", response_text=critic_json)
    ]

    results = enrich_findings(
        [finding],
        _R(),
        gen,
        score_confidence,
        verify_finding,
        repository_context=RepositoryContext(),
    )
    assert len(results) == 1
    enriched = results[0]
    assert enriched.discovery_groundedness == GroundingVerdict.SUPPORTED
    assert enriched.grounding_verdict == GroundingVerdict.INSUFFICIENT_EVIDENCE
    assert enriched.verification_disagreement is True
    assert enriched.verification_status == VerificationStatus.INSUFFICIENT_EVIDENCE
    assert enriched.verification_status != VerificationStatus.VERIFIED


def test_critic_prompt_includes_independent_evaluation_rules_and_discovery_claims():
    """build_critic_prompt includes observed evidence, limitations, groundedness, validation steps, and independent evaluation instructions."""
    from sentinelai.ai.agents.critic import build_critic_prompt
    from sentinelai.ai.agents.schemas import (
        EvidenceAssessment,
        ExploitAssessment,
        ExploitRemediationAssessment,
        RemediationPlan,
    )
    from sentinelai.ai.repository_context import RepositoryContext
    from sentinelai.contracts import GroundingVerdict, ScannerFinding, Severity

    finding = ScannerFinding(
        finding_id="SENT-CRITIC-1",
        scanner="semgrep",
        category="sql-injection",
        severity=Severity.HIGH,
        rule_id="python.sqli",
        message="Formatted SQL query",
        raw_evidence="cursor.execute(f'SELECT * FROM t WHERE x={x}')",
    )
    evidence = EvidenceAssessment(
        title="SQL injection",
        interpretation="String interpolation builds a SQL query from variable x.",
        observed_evidence=["cursor.execute(f'SELECT * FROM t WHERE x={x}')"],
        limitations=["Caller of the function is not visible in the slice."],
        groundedness=GroundingVerdict.SUPPORTED,
    )
    exploit_rem = ExploitRemediationAssessment(
        exploit=ExploitAssessment(
            exploit_path="Attacker sends crafted x parameter to inject SQL predicates.",
            impact="Arbitrary read access to table t.",
            required_assumptions=["x is user-controlled"],
        ),
        remediation=RemediationPlan(
            remediation="Use parameterized query placeholders.",
            validation_steps=["Run integration test with SQL metacharacters in x."],
            patch_suggestion="cursor.execute('SELECT * FROM t WHERE x = ?', (x,))",
        ),
    )

    prompt = build_critic_prompt(finding, evidence, exploit_rem, RepositoryContext(summary="Repo summary"), [])
    assert "do not simply agree with or rubber-stamp the discovery model's conclusion" in prompt
    assert "Observed evidence cited by discovery:" in prompt
    assert "Discovery limitations noted:" in prompt
    assert "Caller of the function is not visible in the slice." in prompt
    assert "Discovery groundedness claim: supported" in prompt
    assert "Validation steps:" in prompt
    assert "Run integration test with SQL metacharacters in x." in prompt


@pytest.mark.parametrize(
    "verifier_failure_mode",
    ["timeout", "malformed_json", "empty_response", "missing_fields", "same_callable"],
)
def test_role_separated_verifier_failures_override_deterministic_verified_to_unverified(verifier_failure_mode):
    """When deterministic verify_finding returns VERIFIED, any independent verifier failure (timeout, malformed JSON, empty response, missing required fields, or same callable) overrides verification_status to UNVERIFIED."""
    import json
    from sentinelai.ai.confidence_scorer import score_confidence
    from sentinelai.ai.llm.base import ProviderError
    from sentinelai.ai.llm.errors import ProviderErrorKind
    from sentinelai.ai.pipeline import _enrich_one
    from sentinelai.ai.repository_context import RepositoryContext
    from sentinelai.ai.retrieval import RetrievedChunk, Retriever
    from sentinelai.ai.verifier import verify_finding
    from sentinelai.contracts import GroundingVerdict, ScannerFinding, Severity, VerificationStatus

    finding = ScannerFinding(
        finding_id="FIND-VER-FAIL",
        scanner="semgrep",
        category="sql-injection",
        severity=Severity.HIGH,
        rule_id="r1",
        message="SQL injection via string concatenation",
        raw_evidence="SELECT * FROM users WHERE id = user_id",
    )

    class _R(Retriever):
        def retrieve(self, q, k):
            return [RetrievedChunk(kb_entry_id="KB1", text="sql injection", references=["CWE-89"], score=1.0)]

    class _Settings:
        retrieval_top_k = 1
        enable_verification = True

    evidence_json = json.dumps({
        "title": "SQL Injection in user lookup query",
        "interpretation": "The sql injection query concatenates untrusted user_id directly into the SELECT statement.",
        "observed_evidence": ["SELECT * FROM users WHERE id = user_id"],
        "limitations": [],
        "groundedness": "supported",
    })
    exploit_json = json.dumps({
        "exploit": {
            "exploit_path": "An attacker supplies a crafted user_id string to alter the SQL query logic.",
            "impact": "Unauthorized read access to all rows in the users table.",
            "required_assumptions": ["user_id is controlled by an external request"],
        },
        "remediation": {
            "remediation": "Replace string concatenation with a parameterized sql injection-safe query placeholder.",
            "validation_steps": ["Verify that database driver binds user_id as a parameter."],
            "patch_suggestion": "cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))",
        },
    })

    def gen_discovery(p, **kw):
        return evidence_json

    def gen_remediation(p, **kw):
        return exploit_json

    if verifier_failure_mode == "timeout":
        def gen_verification(p, **kw):
            raise ProviderError("grok", ProviderErrorKind.TIMEOUT, "no response within 60s")
    elif verifier_failure_mode == "malformed_json":
        def gen_verification(p, **kw):
            return "{not valid json: SECRET_ECHO_TEST"
    elif verifier_failure_mode == "empty_response":
        def gen_verification(p, **kw):
            return ""
    elif verifier_failure_mode == "missing_fields":
        def gen_verification(p, **kw):
            return json.dumps({"supported_claims": ["missing verdict and unsupported_claims"]})
    else:
        gen_verification = gen_discovery

    role_generates = {
        "discovery": gen_discovery,
        "remediation": gen_remediation,
        "verification": gen_verification,
    }

    enriched = _enrich_one(
        finding,
        RepositoryContext(),
        _R(),
        gen_discovery,
        score_confidence,
        verify_finding,
        _Settings(),
        role_generates=role_generates,
    )

    assert enriched.finding_id == "FIND-VER-FAIL"
    assert enriched.verification_status == VerificationStatus.UNVERIFIED
    assert enriched.grounding_verdict == GroundingVerdict.INSUFFICIENT_EVIDENCE
    assert "SECRET_ECHO_TEST" not in " ".join(enriched.unsupported_claims)


def test_fallback_provider_active_candidate_excludes_skipped_remote_provider():
    """When a local provider fails and a subsequent remote candidate is skipped due to egress='deny', active_candidate remains the attempted local provider, not the skipped remote provider."""
    from sentinelai.ai.llm.base import ProviderError
    from sentinelai.ai.llm.errors import ProviderErrorKind
    from sentinelai.ai.llm.fallback import FallbackLLMProvider

    cand_local = _FakeRoleCandidate(
        "ollama",
        is_local=True,
        model="llama3.1:8b",
        error_to_raise=ProviderError("ollama", ProviderErrorKind.TRANSPORT, "connection refused"),
    )
    cand_remote = _FakeRoleCandidate("gemini", is_local=False, model="gemini-2.5-flash")

    fb = FallbackLLMProvider([cand_local, cand_remote], base_egress="deny")
    with pytest.raises(ProviderError):
        fb.complete("test prompt")

    assert fb.active_provider_name == "ollama"
    assert fb.is_local is True
    assert fb.retains_data is False
    assert cand_remote.call_count == 0





