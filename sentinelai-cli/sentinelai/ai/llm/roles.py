"""
Role-separated AI provider architecture and independent vulnerability verification.

Defines the three pipeline roles:
  1. discovery: Analyzes scanner findings and code context to produce candidate findings.
  2. remediation: Proposes exploit paths, impact assessments, and remediation/patch suggestions.
  3. verification: Independently reviews and challenges candidate findings as a critic.

Security and operational invariants:
  - Independent verification (provider-level): The verifier MUST NOT use the same provider
    that handled discovery for the same finding (`candidate.name != active_discovery_provider`).
    Note that provider-name separation guarantees distinct provider endpoints/vendors, not
    necessarily distinct underlying open-weight model families across third-party hosts.
    If no independent verifier is available, verification is marked as unavailable/unverified,
    never falsely verified.
  - Role-specific fallback chains: Each role has an ordered candidate chain.
    Default order:
      discovery: Gemini -> Grok (xAI) -> NVIDIA NIM -> Ollama
      verification: Grok (xAI) -> NVIDIA NIM -> Gemini -> Ollama (skipping the discovery provider)
      remediation: Gemini -> NVIDIA NIM -> Grok (xAI) -> Ollama
  - Egress and privacy: Remote providers require explicit egress authorization (egress == "allow").
    Egress violations on secrets or payload size fail closed immediately (never fall back).
  - Role-specific provenance: Completed requests per (role, provider, model) are tracked
    in AIRoleProvenance rather than collapsing all roles into a single misleading provider.
"""
import logging
from typing import Callable, ClassVar, Optional

from sentinelai.ai.egress import EgressViolation, assert_safe
from sentinelai.ai.llm.base import LLMProvider, LLMResponse, ProviderError
from sentinelai.ai.llm.errors import ProviderErrorKind
from sentinelai.ai.llm.fallback import ProviderCandidate, _to_ai_fault
from sentinelai.contracts import (
    AIConfig,
    AIFault,
    AIRole,
    AIRoleProvenance,
    AI_ROLES,
    RetentionStatus,
    retention_status_from_provider,
)

logger = logging.getLogger("sentinelai")

DEFAULT_ROLE_CHAINS: dict[str, list[str]] = {
    "discovery": ["gemini", "grok", "nvidia", "ollama"],
    "verification": ["grok", "nvidia", "gemini", "ollama"],
    "remediation": ["gemini", "nvidia", "grok", "ollama"],
}


class IndependentVerificationUnavailableError(ProviderError):
    """Raised when no independent verifier is available (or all independent verifiers failed)."""

    def __init__(self, discovery_provider: str, reason: str = "no independent verifier available distinct from discovery provider"):
        super().__init__(
            "verification",
            ProviderErrorKind.CONFIGURATION,
            f"{reason} (discovery used {discovery_provider!r})",
        )
        self.discovery_provider = discovery_provider


class ProvenanceTracker:
    """Tally completed requests per (role, provider, model, retention_status, is_local, retention_basis, retention_policy_note)."""

    def __init__(self) -> None:
        self._counts: dict[tuple[str, str, str, str, bool, Optional[str], Optional[str]], int] = {}

    def record(self, role: str, candidate: ProviderCandidate) -> None:
        ret_status = retention_status_from_provider(
            candidate.retains_data, is_local=candidate.is_local
        )
        key = (
            role,
            candidate.name,
            candidate.model,
            ret_status,
            candidate.is_local,
            candidate.retention_basis,
            candidate.retention_policy_note,
        )
        self._counts[key] = self._counts.get(key, 0) + 1

    def to_provenance_list(self) -> list[AIRoleProvenance]:
        return [
            AIRoleProvenance(
                role=role,  # type: ignore[arg-type]
                provider=provider,
                model=model,
                retention_status=ret_status,  # type: ignore[arg-type]
                retention_basis=ret_basis,
                retention_policy_note=ret_note,
                is_local=is_local,
                completed_requests=count,
            )
            for (role, provider, model, ret_status, is_local, ret_basis, ret_note), count in self._counts.items()
        ]


def build_candidate_for_provider(name: str, cfg: AIConfig) -> Optional[ProviderCandidate]:
    """Construct one ProviderCandidate for the given provider name using AIConfig settings."""
    from sentinelai.ai.config import PROVIDERS, get_settings
    from sentinelai.ai.llm.gemini import GeminiProvider
    from sentinelai.ai.llm.groq import GroqProvider
    from sentinelai.ai.llm.mock import MockLLMProvider
    from sentinelai.ai.llm.nvidia import NvidiaNimProvider
    from sentinelai.ai.llm.ollama import OllamaProvider as LocalOllamaProvider
    from sentinelai.ai.llm.xai import GrokProvider

    settings = get_settings()
    norm_name = name.strip().lower()

    if norm_name == "gemini":
        model = cfg.model if cfg.provider == "gemini" else PROVIDERS["gemini"].default_model
        cand_cfg = AIConfig(
            provider="gemini",
            model=model,
            egress=cfg.egress,
            max_context_chars=cfg.max_context_chars,
        )
        return ProviderCandidate(
            name="gemini",
            model=model,
            is_local=False,
            retains_data=None,
            config=cand_cfg,
            factory=lambda m=model: GeminiProvider(m),
            retention_basis=GeminiProvider.retention_basis,
            retention_policy_note=GeminiProvider.retention_policy_note,
        )

    if norm_name == "grok":
        model = cfg.model if cfg.provider == "grok" else PROVIDERS["grok"].default_model
        cand_cfg = AIConfig(
            provider="grok",
            model=model,
            egress=cfg.egress,
            max_context_chars=cfg.max_context_chars,
        )
        return ProviderCandidate(
            name="grok",
            model=model,
            is_local=False,
            retains_data=None,
            config=cand_cfg,
            factory=lambda m=model: GrokProvider(m),
            retention_basis=GrokProvider.retention_basis,
            retention_policy_note=GrokProvider.retention_policy_note,
        )

    if norm_name == "nvidia":
        model = cfg.model if cfg.provider == "nvidia" else PROVIDERS["nvidia"].default_model
        cand_cfg = AIConfig(
            provider="nvidia",
            model=model,
            egress=cfg.egress,
            max_context_chars=cfg.max_context_chars,
        )
        return ProviderCandidate(
            name="nvidia",
            model=model,
            is_local=False,
            retains_data=None,
            config=cand_cfg,
            factory=lambda m=model: NvidiaNimProvider(m),
            retention_basis=NvidiaNimProvider.retention_basis,
            retention_policy_note=NvidiaNimProvider.retention_policy_note,
        )

    if norm_name == "groq":
        model = cfg.model if cfg.provider == "groq" else PROVIDERS["groq"].default_model
        cand_cfg = AIConfig(
            provider="groq",
            model=model,
            egress=cfg.egress,
            max_context_chars=cfg.max_context_chars,
        )
        return ProviderCandidate(
            name="groq",
            model=model,
            is_local=False,
            retains_data=None,
            config=cand_cfg,
            factory=lambda m=model: GroqProvider(m),
            retention_basis=GroqProvider.retention_basis,
            retention_policy_note=GroqProvider.retention_policy_note,
        )

    if norm_name == "ollama":
        model = (
            cfg.model
            if cfg.provider == "ollama"
            else (settings.llm_model or PROVIDERS["ollama"].default_model)
        )
        cand_cfg = AIConfig(
            provider="ollama",
            model=model,
            egress=cfg.egress,
            max_context_chars=cfg.max_context_chars,
        )
        return ProviderCandidate(
            name="ollama",
            model=model,
            is_local=True,
            retains_data=False,
            config=cand_cfg,
            factory=lambda m=model, h=settings.llm_host, t=settings.request_timeout_seconds: LocalOllamaProvider(
                m, host=h, timeout_s=t
            ),
            retention_basis=LocalOllamaProvider.retention_basis,
            retention_policy_note=LocalOllamaProvider.retention_policy_note,
        )

    if norm_name == "mock":
        cand_cfg = AIConfig(
            provider="mock",
            model="mock-model",
            egress=cfg.egress,
            max_context_chars=cfg.max_context_chars,
        )
        return ProviderCandidate(
            name="mock",
            model="mock-model",
            is_local=True,
            retains_data=False,
            config=cand_cfg,
            factory=lambda: MockLLMProvider(),
            retention_basis=MockLLMProvider.retention_basis,
            retention_policy_note=MockLLMProvider.retention_policy_note,
        )

    return None


class RoleAwareGenerator:
    """Manages role-specific execution, fallback, and independent verification."""

    def __init__(
        self,
        cfg: AIConfig,
        *,
        role_chains: Optional[dict[str, list[str]]] = None,
        enforce_independence: bool = True,
        max_tokens: int = 2048,
        temperature: float = 0.1,
    ) -> None:
        self.cfg = cfg
        self.enforce_independence = enforce_independence
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.provenance_tracker = ProvenanceTracker()
        self.faults: list[AIFault] = []

        # Current per-finding state
        self.active_discovery_provider: Optional[str] = None
        self.active_discovery_model: Optional[str] = None

        # Build role chains (preserve an explicit empty dict rather than substituting defaults)
        if role_chains is not None:
            raw_chains = {k: list(v) for k, v in role_chains.items()}
        elif cfg.role_chains:
            raw_chains = {k: list(v) for k, v in cfg.role_chains.items()}
        else:
            raw_chains = {k: list(v) for k, v in DEFAULT_ROLE_CHAINS.items()}
        self.role_chains: dict[str, list[str]] = raw_chains

        # Pre-build deduplicated candidates for each role
        self.candidates_by_role: dict[str, list[ProviderCandidate]] = {}
        for role, names in self.role_chains.items():
            if role not in AI_ROLES:
                raise ProviderError(
                    role,
                    ProviderErrorKind.CONFIGURATION,
                    f"unknown AI role {role!r}; must be one of {', '.join(AI_ROLES)}",
                )
            cands: list[ProviderCandidate] = []
            seen_names: set[str] = set()
            for name in names:
                norm_name = name.strip().lower()
                if norm_name in seen_names:
                    continue
                cand = build_candidate_for_provider(norm_name, cfg)
                if cand is None:
                    raise ProviderError(
                        role,
                        ProviderErrorKind.CONFIGURATION,
                        f"unknown provider {name!r} configured for role {role!r}",
                    )
                seen_names.add(cand.name)
                cands.append(cand)
            self.candidates_by_role[role] = cands

    def reset_for_finding(self) -> None:
        """Reset finding-level state before processing a new finding."""
        self.active_discovery_provider = None
        self.active_discovery_model = None

    def for_role(self, role: str) -> Callable[..., str]:
        """Return an LLMFn callable dedicated to one pipeline role."""
        def _role_generate(
            prompt: str,
            response_schema=None,
            system: Optional[str] = None,
            max_tokens: Optional[int] = None,
            temperature: Optional[float] = None,
        ) -> str:
            return self.complete_for_role(
                role,
                prompt,
                system=system,
                max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
                temperature=temperature if temperature is not None else self.temperature,
            ).text
        return _role_generate

    def __call__(
        self,
        prompt: str,
        response_schema=None,
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Direct callable interface defaulting to 'discovery' role for backwards compatibility."""
        return self.complete_for_role(
            "discovery",
            prompt,
            system=system,
            max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
            temperature=temperature if temperature is not None else self.temperature,
        ).text

    def complete_for_role(
        self,
        role: str,
        prompt: str,
        *,
        system: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.1,
        assert_safe_fn: Callable = assert_safe,
    ) -> LLMResponse:
        if role not in AI_ROLES:
            err = ProviderError(
                role,
                ProviderErrorKind.CONFIGURATION,
                f"unknown AI role {role!r}; must be one of {', '.join(AI_ROLES)}",
            )
            self.faults.append(_to_ai_fault(err))
            raise err

        if role == "discovery":
            self.active_discovery_provider = None
            self.active_discovery_model = None

        candidates = list(self.candidates_by_role.get(role, []))
        if not candidates:
            if role == "verification" and self.enforce_independence:
                err = IndependentVerificationUnavailableError(
                    self.active_discovery_provider or "unrecorded",
                    reason="no provider candidates configured for verification role",
                )
            else:
                err = ProviderError(
                    role,
                    ProviderErrorKind.CONFIGURATION,
                    f"no provider candidates configured for role {role!r}",
                )
            self.faults.append(_to_ai_fault(err))
            raise err

        # Enforce independence on verification role
        discovery_prov = self.active_discovery_provider
        if role == "verification" and self.enforce_independence:
            if not discovery_prov:
                err = IndependentVerificationUnavailableError(
                    "unrecorded",
                    reason="cannot verify independence because no discovery provider was recorded",
                )
                self.faults.append(_to_ai_fault(err))
                raise err
            candidates = [c for c in candidates if c.name != discovery_prov]
            if not candidates:
                err = IndependentVerificationUnavailableError(discovery_prov)
                self.faults.append(_to_ai_fault(err))
                raise err

        last_exc: Optional[BaseException] = None
        attempted: set[str] = set()
        for candidate in candidates:
            if candidate.name in attempted:
                continue
            attempted.add(candidate.name)

            # Egress policy check
            if not candidate.is_local and self.cfg.egress != "allow":
                logger.warning(
                    "Role %s: skipping remote provider %r because egress is %r",
                    role,
                    candidate.name,
                    self.cfg.egress,
                )
                continue

            # Instantiation
            try:
                provider = candidate.get_instance()
            except Exception as exc:
                logger.warning("Role %s: provider %r failed initialization: %s", role, candidate.name, exc)
                pe = ProviderError(candidate.name, ProviderErrorKind.CONFIGURATION, str(exc))
                self.faults.append(_to_ai_fault(pe))
                last_exc = pe
                continue

            # Guarded safety check — fail closed on any EgressViolation (secrets, size, or policy)
            try:
                assert_safe_fn(prompt, candidate.config, provider, system=system)
            except EgressViolation as exc:
                self.faults.append(_to_ai_fault(exc))
                raise

            # Completion attempt
            try:
                resp = provider.complete(
                    prompt,
                    system=system,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                # Successful execution: record provenance
                self.provenance_tracker.record(role, candidate)
                if role == "discovery":
                    self.active_discovery_provider = candidate.name
                    self.active_discovery_model = candidate.model
                return resp
            except ProviderError as exc:
                logger.warning(
                    "Role %s: provider %r operational failure (%s): %s",
                    role,
                    candidate.name,
                    getattr(exc, "kind", "error"),
                    exc,
                )
                self.faults.append(_to_ai_fault(exc))
                last_exc = exc
                continue

        # All candidates in role chain failed or were ineligible
        if role == "verification" and self.enforce_independence:
            reason = (
                str(last_exc)
                if last_exc is not None
                else "all independent verification providers failed or were ineligible"
            )
            err = IndependentVerificationUnavailableError(discovery_prov or "unrecorded", reason=reason)
            if last_exc is None:
                self.faults.append(_to_ai_fault(err))
            raise err
        if isinstance(last_exc, ProviderError):
            raise last_exc
        err = ProviderError(
            role,
            ProviderErrorKind.CONFIGURATION,
            f"all configured {role} providers failed or were ineligible",
        )
        self.faults.append(_to_ai_fault(err))
        raise err

    def get_role_provenance(self) -> list[AIRoleProvenance]:
        return self.provenance_tracker.to_provenance_list()

    def get_faults(self) -> list[AIFault]:
        return list(self.faults)

