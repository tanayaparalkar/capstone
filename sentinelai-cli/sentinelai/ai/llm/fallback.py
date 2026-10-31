"""
Centralized automatic provider selection and fallback mechanism.

Default priority order:
1. Gemini (GeminiProvider via Google Cloud Vertex AI)
2. Grok (GrokProvider via native xAI API)
3. NVIDIA NIM (NvidiaNimProvider via NVIDIA Integrate API)
4. Ollama (LocalOllamaProvider / OllamaProvider - final local fallback)

Security and operational behavior:
- Remote egress is strictly opt-in (egress == "allow").
- A policy denial on a remote provider NEVER falls through to another remote provider.
- Ollama is strictly local (loopback-only endpoint).
- Egress violations (policy, secrets, or payload size) NEVER fall through (fail closed immediately).
- Bounded attempts: each candidate provider is called at most once per fallback sequence.
- Operational provider errors (rate limit, auth, timeout, transport, 5xx) trigger fallback.
"""
import logging
from dataclasses import dataclass
from typing import Callable, ClassVar, Optional

from sentinelai.ai.egress import EgressViolation, assert_safe
from sentinelai.ai.llm.base import LLMProvider, LLMResponse, ProviderError
from sentinelai.ai.llm.errors import ProviderErrorKind
from sentinelai.contracts import AIConfig, AIFault

logger = logging.getLogger("sentinelai")


def _to_ai_fault(exc: BaseException) -> AIFault:
    safe = isinstance(exc, (ProviderError, EgressViolation))
    return AIFault(finding_id=None, kind=type(exc).__name__, message=str(exc) if safe else type(exc).__name__)

DEFAULT_PROVIDER_PRIORITY: tuple[str, ...] = ("gemini", "grok", "nvidia", "ollama")
# No alias mapping — "grok" is now a real provider (xAI Grok via ai/llm/xai.py).
# "groq" remains the separate Groq provider. Neither aliases the other.
PROVIDER_ALIASES: dict[str, str] = {}
# All providers are implemented — nothing to skip.
UNIMPLEMENTED_PROVIDERS: set[str] = set()


@dataclass
class ProviderCandidate:
    """One candidate LLM provider in the fallback chain."""

    name: str
    model: str
    is_local: bool
    retains_data: Optional[bool]
    config: AIConfig
    factory: Callable[[], LLMProvider]
    retention_basis: Optional[str] = None
    retention_policy_note: Optional[str] = None
    _instance: Optional[LLMProvider] = None

    def get_instance(self) -> LLMProvider:
        if self._instance is None:
            self._instance = self.factory()
        return self._instance


class FallbackLLMProvider(LLMProvider):
    """An LLMProvider that deterministically falls back through candidate providers on operational failures."""

    # Defaults for ClassVar interface compatibility; dynamically delegated via properties below.
    name: ClassVar[str] = "fallback"
    is_local: ClassVar[bool] = False
    retains_data: ClassVar[Optional[bool]] = None

    def __init__(
        self,
        candidates: list[ProviderCandidate],
        *,
        base_egress: str = "deny",
        on_fallback: Optional[Callable[[str, Optional[str], BaseException], None]] = None,
    ) -> None:
        if not candidates:
            raise ProviderError("fallback", ProviderErrorKind.CONFIGURATION, "no provider candidates configured")
        self.candidates = list(candidates)
        self.base_egress = base_egress
        self.on_fallback = on_fallback
        self.active_index = 0
        self._last_selected_candidate: Optional[ProviderCandidate] = None
        self.failed_providers: set[str] = set()
        self.fallback_faults: list[AIFault] = []

    @property
    def active_candidate(self) -> ProviderCandidate:
        if self._last_selected_candidate is not None:
            return self._last_selected_candidate
        idx = min(self.active_index, len(self.candidates) - 1)
        return self.candidates[idx]

    @property
    def name(self) -> str:  # type: ignore[override]
        return self.active_candidate.name

    @property
    def is_local(self) -> bool:  # type: ignore[override]
        return self.active_candidate.is_local

    @property
    def retains_data(self) -> Optional[bool]:  # type: ignore[override]
        return self.active_candidate.retains_data

    @property
    def active_config(self) -> AIConfig:
        return self.active_candidate.config

    @property
    def active_provider_name(self) -> str:
        return self.active_candidate.name

    @property
    def active_model_name(self) -> str:
        return self.active_candidate.model

    @property
    def active_provider(self) -> LLMProvider:
        return self.active_candidate.get_instance()

    def complete(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.1,
    ) -> LLMResponse:
        last_exc: Optional[BaseException] = None

        while self.active_index < len(self.candidates):
            candidate = self.candidates[self.active_index]

            if candidate.name in self.failed_providers:
                self.active_index += 1
                continue

            # Policy check: remote provider without explicit egress authorization is refused
            if not candidate.is_local and self.base_egress != "allow":
                logger.warning(
                    "Fallback: skipping remote provider %r because egress is %r",
                    candidate.name,
                    self.base_egress,
                )
                self.failed_providers.add(candidate.name)
                self.active_index += 1
                continue

            self._last_selected_candidate = candidate

            # Step 1: Initialize the candidate provider
            try:
                provider = candidate.get_instance()
            except ProviderError as exc:
                logger.warning("Fallback: provider %r configuration failed: %s", candidate.name, exc)
                self.failed_providers.add(candidate.name)
                self.fallback_faults.append(_to_ai_fault(exc))
                next_name = (
                    self.candidates[self.active_index + 1].name
                    if self.active_index + 1 < len(self.candidates)
                    else None
                )
                if self.on_fallback:
                    self.on_fallback(candidate.name, next_name, exc)
                last_exc = exc
                self.active_index += 1
                continue
            except Exception as exc:
                logger.warning("Fallback: provider %r failed initialization: %s", candidate.name, exc)
                pe = ProviderError(candidate.name, ProviderErrorKind.CONFIGURATION, str(exc))
                self.failed_providers.add(candidate.name)
                self.fallback_faults.append(_to_ai_fault(pe))
                next_name = (
                    self.candidates[self.active_index + 1].name
                    if self.active_index + 1 < len(self.candidates)
                    else None
                )
                if self.on_fallback:
                    self.on_fallback(candidate.name, next_name, pe)
                last_exc = pe
                self.active_index += 1
                continue

            # Step 2: Validate egress and safety checks for this specific candidate
            # Critical security invariant: never fall back on any EgressViolation (secrets, size, or policy).
            try:
                assert_safe(prompt, candidate.config, provider, system=system)
            except EgressViolation as exc:
                self.failed_providers.add(candidate.name)
                self.fallback_faults.append(_to_ai_fault(exc))
                raise

            # Step 3: Attempt completion with candidate provider
            try:
                return provider.complete(
                    prompt,
                    system=system,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            except ProviderError as exc:
                logger.warning(
                    "Fallback: provider %r operational failure (%s): %s",
                    candidate.name,
                    getattr(exc, "kind", "error"),
                    exc,
                )
                self.failed_providers.add(candidate.name)
                self.fallback_faults.append(_to_ai_fault(exc))
                next_name = (
                    self.candidates[self.active_index + 1].name
                    if self.active_index + 1 < len(self.candidates)
                    else None
                )
                if self.on_fallback:
                    self.on_fallback(candidate.name, next_name, exc)
                last_exc = exc
                self.active_index += 1
                continue

        # All candidate providers failed
        if isinstance(last_exc, ProviderError):
            raise last_exc
        raise ProviderError(
            "fallback",
            ProviderErrorKind.CONFIGURATION,
            "all configured fallback providers failed",
        )


def create_fallback_provider(cfg: AIConfig) -> FallbackLLMProvider:
    """Build a FallbackLLMProvider from an AIConfig's primary and fallback provider specifications."""
    from sentinelai.ai.config import PROVIDERS, get_settings
    from sentinelai.ai.llm.gemini import GeminiProvider
    from sentinelai.ai.llm.groq import GroqProvider
    from sentinelai.ai.llm.mock import MockLLMProvider
    from sentinelai.ai.llm.nvidia import NvidiaNimProvider
    from sentinelai.ai.llm.ollama import OllamaProvider as LocalOllamaProvider
    from sentinelai.ai.llm.xai import GrokProvider

    all_names: list[str] = []
    for raw_name in [cfg.provider, *cfg.fallback_providers]:
        # No alias mapping; "grok" and "groq" are distinct real providers.
        name = raw_name.strip().lower()
        if name and name not in all_names:
            all_names.append(name)

    candidates: list[ProviderCandidate] = []
    settings = get_settings()

    for name in all_names:
        if name in UNIMPLEMENTED_PROVIDERS:
            logger.info("Provider %r is not implemented in current architecture; skipping", name)
            continue

        if name == "groq":
            model = cfg.model if cfg.provider == "groq" else PROVIDERS["groq"].default_model
            cand_cfg = AIConfig(
                provider="groq",
                model=model,
                egress=cfg.egress,
                max_context_chars=cfg.max_context_chars,
            )
            candidates.append(ProviderCandidate(
                name="groq",
                model=model,
                is_local=False,
                retains_data=None,
                config=cand_cfg,
                factory=lambda m=model: GroqProvider(m),
                retention_basis=GroqProvider.retention_basis,
                retention_policy_note=GroqProvider.retention_policy_note,
            ))
        elif name == "grok":
            model = cfg.model if cfg.provider == "grok" else PROVIDERS["grok"].default_model
            cand_cfg = AIConfig(
                provider="grok",
                model=model,
                egress=cfg.egress,
                max_context_chars=cfg.max_context_chars,
            )
            candidates.append(ProviderCandidate(
                name="grok",
                model=model,
                is_local=False,
                retains_data=None,
                config=cand_cfg,
                factory=lambda m=model: GrokProvider(m),
                retention_basis=GrokProvider.retention_basis,
                retention_policy_note=GrokProvider.retention_policy_note,
            ))
        elif name == "nvidia":
            model = cfg.model if cfg.provider == "nvidia" else PROVIDERS["nvidia"].default_model
            cand_cfg = AIConfig(
                provider="nvidia",
                model=model,
                egress=cfg.egress,
                max_context_chars=cfg.max_context_chars,
            )
            candidates.append(ProviderCandidate(
                name="nvidia",
                model=model,
                is_local=False,
                retains_data=None,
                config=cand_cfg,
                factory=lambda m=model: NvidiaNimProvider(m),
                retention_basis=NvidiaNimProvider.retention_basis,
                retention_policy_note=NvidiaNimProvider.retention_policy_note,
            ))
        elif name == "gemini":
            model = cfg.model if cfg.provider == "gemini" else PROVIDERS["gemini"].default_model
            cand_cfg = AIConfig(
                provider="gemini",
                model=model,
                egress=cfg.egress,
                max_context_chars=cfg.max_context_chars,
            )
            candidates.append(ProviderCandidate(
                name="gemini",
                model=model,
                is_local=False,
                retains_data=None,
                config=cand_cfg,
                factory=lambda m=model: GeminiProvider(m),
                retention_basis=GeminiProvider.retention_basis,
                retention_policy_note=GeminiProvider.retention_policy_note,
            ))
        elif name == "ollama":
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
            candidates.append(ProviderCandidate(
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
            ))
        elif name == "mock":
            cand_cfg = AIConfig(
                provider="mock",
                model="mock-model",
                egress=cfg.egress,
                max_context_chars=cfg.max_context_chars,
            )
            candidates.append(ProviderCandidate(
                name="mock",
                model="mock-model",
                is_local=True,
                retains_data=False,
                config=cand_cfg,
                factory=lambda: MockLLMProvider(),
                retention_basis=MockLLMProvider.retention_basis,
                retention_policy_note=MockLLMProvider.retention_policy_note,
            ))

    return FallbackLLMProvider(candidates, base_egress=cfg.egress)
