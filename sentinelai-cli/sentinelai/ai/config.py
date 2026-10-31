"""
Central runtime settings for the AI Intelligence Layer.

Holds the tunable knobs the rest of sentinelai.ai reads: how many
knowledge-base chunks to retrieve per finding, the confidence-label
thresholds, whether verification runs, the logging level, which
LLM/embedding provider is active, and (added once the team chose
Ollama) where to reach it and which model to use.

No retry_max_attempts/retry_backoff_seconds: an earlier draft of this
file included both, anticipating ai/utils/retry.py. That file was never
built, and nothing anywhere reads either field - unlike llm_provider/
embedding_provider (kept deliberately, per an explicit instruction to
leave room for a future decision), these two were never reaffirmed as
intentional and describe behavior that doesn't exist. Removed during
the whole-layer architecture review: this build has consistently
rejected exactly this pattern (a config field with no consumer) for
everything else, and config.py is not an exception. If retry logic is
built later, its settings arrive with that file, reviewed together -
matching how every other field here was added only once a real
consumer existed.

AI enrichment is built around two separate Ollama models, because
Ollama itself serves generation and embeddings from different model
families:

    llm_model        text generation, e.g. 'llama3.1:8b'   (/api/generate)
    embedding_model  embeddings,      e.g. 'nomic-embed-text' (/api/embed)

Both must be set for AI enrichment to run - sentinelai/main.py skips the
entire stage unless both are present, and a scan with either missing
still succeeds as a scanner-only run. They must also name *different*
models: pointing embedding_model at a generation model is the most
common misconfiguration here and makes Ollama return HTTP 501, which
ai/embeddings_ollama.py translates into an error naming the model and
the fix.

Deliberately NOT here:

- The knowledge-base path. security_kb/loader.py owns its own default
  location, the same way providers/mock_provider.py owns
  DEFAULT_MOCK_DATA_PATH instead of routing it through a shared object.
- Any API key. Ollama is a locally-hosted provider with no credential
  to configure - llm_host/llm_model below name *where* and *what*, never
  a secret. embedding_provider remains reserved/unused pending that
  decision, unchanged.
- Validation that embedding_model is genuinely an embedding model.
  Nothing in a model *name* reliably says whether it supports
  embeddings, so a name-based check here would be a guess that both
  rejects valid models and misses invalid ones. Ollama itself is the
  only authority, so the check happens where the call is made.

llm_model is Optional[str] = None, not required, even though
ai/llm_ollama.py cannot actually run without one: making it required
here would break AISettings()'s zero-configuration-construction
property for every unrelated caller (e.g. security_kb/loader.py's own
tests, which never touch the LLM) - every other consumer of AISettings
would suddenly need SENTINELAI_AI_LLM_MODEL set for no reason of their
own. OllamaProvider.__init__ validates model is non-empty and raises
immediately instead - the "you must configure this" failure happens at
the one place it's actually needed, not at every AISettings()
construction. llm_host does get a real default (Ollama's own standard
local address) since, unlike a model name, there is a genuinely safe,
non-secret value to default to.

Plain pydantic.BaseModel + manual os.environ reading, not
pydantic-settings.BaseSettings: pydantic-settings is not declared in
this project's pyproject.toml/requirements.txt (only bare pydantic is).
It happens to be importable in some ambient environments on this
machine, but a clean `pip install -e .` of this project would not have
it - using it here would silently introduce an undeclared dependency.
Revisit this choice if pydantic-settings is ever added to
pyproject.toml's [project.dependencies] deliberately.

get_settings() is a lazy singleton, mirroring
presentation/console.py's get_console(): built once from the
environment on first call, cached for the life of the process.

resolve_config() is separate from AISettings and from that singleton.
It produces the multi-provider AIConfig (contracts/ai.py) from three
sources, highest first: CLI flag, environment variable, default. It
reads the environment on every call rather than caching, because a
remote provider has to be opted into explicitly *for this run*, and a
cached answer from earlier in the process is not that.

    provider  --ai-provider  SENTINELAI_AI_PROVIDER  auto (ollama when egress=deny; gemini -> grok -> nvidia -> ollama when egress=allow and credentials are present)
    model     --ai-model     SENTINELAI_AI_MODEL     the provider's default model
    egress    --egress       SENTINELAI_EGRESS       deny

The defaults are local and closed: a first run with nothing configured
never sends code off the machine. Explicitly choosing a remote provider
is refused here, before anything runs, unless egress is "allow" and every
credential variable the provider needs is present; automatic remote
selection and remote fallback occur only when egress is explicitly set to
"allow". Credential variables are checked for presence only; their values
are never read into AIConfig, logged, or echoed in an error.
"""
import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Optional, get_args

from pydantic import BaseModel, Field, field_validator, model_validator

from sentinelai.backend.code_context import DEFAULT_CONTEXT_LINES
from sentinelai.contracts import AIConfig

from .llm.base import LLMProvider
from .llm.gemini import CREDENTIALS_ENV, PROJECT_ENV, GeminiProvider
from .llm.groq import API_KEY_ENV as GROQ_API_KEY_ENV, GroqProvider
from .llm.mock import MockLLMProvider
from .llm.nvidia import API_KEY_ENV as NVIDIA_API_KEY_ENV, NvidiaNimProvider
from .llm.ollama import OllamaProvider
from .llm.xai import API_KEY_ENV as XAI_API_KEY_ENV, GrokProvider

# Keep the old name available for any remaining internal references.
API_KEY_ENV = GROQ_API_KEY_ENV

from .local_endpoint import validate_local_endpoint

_ENV_PREFIX = "SENTINELAI_AI_"


class LogLevel(str, Enum):
    """Mirrors contracts.Severity / contracts.ConfidenceLabel's str-Enum pattern rather than a validated string."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class AISettings(BaseModel):
    """Tunable runtime settings for the AI Intelligence Layer, sourced from SENTINELAI_AI_* env vars."""

    retrieval_top_k: int = Field(
        default=5, ge=1, description="Number of knowledge-base chunks to retrieve per finding."
    )
    confidence_high_threshold: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Minimum confidence_score (inclusive) mapped to ConfidenceLabel.HIGH.",
    )
    confidence_medium_threshold: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Minimum confidence_score (inclusive) mapped to ConfidenceLabel.MEDIUM; below this is LOW.",
    )
    enable_verification: bool = Field(
        default=True, description="Whether the verification step runs before a finding is emitted."
    )
    code_context_lines: int = Field(
        default=DEFAULT_CONTEXT_LINES,
        ge=0,
        description=(
            "Lines of surrounding source shown either side of a finding's own location in every "
            "agent prompt. The default matches backend/code_context.py's own default, which in "
            "turn matches backend/snippets.py's retrieval chunk size. Raise it when a fix needs "
            "more of the enclosing function to be visible; 0 shows the finding's own lines only. "
            "This widens the prompt for all three agents at once, so large values cost tokens on "
            "every call - and a model with a small context window will silently truncate."
        ),
    )
    log_level: LogLevel = Field(default=LogLevel.INFO, description="Logging level for sentinelai.ai's logger.")
    llm_provider: Optional[str] = Field(
        default=None,
        description=(
            "Identifier for which LLM implementation is active. Unused while ai/llm_base.py has a single "
            "concrete implementation - reserved so a second implementation has somewhere to be selected "
            "from later without changing this model's shape. No value is hardcoded here; stays unset "
            "until the project owner names the concrete provider."
        ),
    )
    embedding_provider: Optional[str] = Field(
        default=None,
        description="Identifier for which embedding implementation is active. Same rationale as llm_provider.",
    )
    llm_host: str = Field(
        default="http://localhost:11434",
        description=(
            "Base URL of the local Ollama server. Shared between generation and embeddings - Ollama "
            "serves both /api/generate and /api/embed from this same host, so ai/embeddings_ollama.py "
            "reuses this field rather than a separate embedding_host."
        ),
    )
    request_timeout_seconds: float = Field(
        default=120.0,
        gt=0,
        description=(
            "Per-attempt timeout, in seconds, for every Ollama HTTP request (both /api/generate and "
            "/api/embed). Applies to each attempt independently rather than being divided across "
            "retries. The default is the value both clients previously hardcoded, so leaving this "
            "unset reproduces the prior behaviour exactly. Raise it for a slow machine or a large "
            "model; the multi-agent pipeline issues three generation calls per correlated finding, "
            "so a timeout that is too low fails a scan late rather than early."
        ),
    )
    max_attempts: int = Field(
        default=2,
        ge=1,
        description=(
            "Total attempts per Ollama request, including the first - so 1 disables retrying. Only "
            "transient failures are ever retried (connection errors and HTTP 500/502/503/504); "
            "deterministic ones such as 404 and 501 are not, and raising this value does not change "
            "that classification. The default matches the value ai/ollama_http.py previously "
            "hardcoded. Keep it low: ai/pipeline.py handles failures per finding, so the worst-case "
            "added delay is this value multiplied across every finding in a scan."
        ),
    )
    llm_model: Optional[str] = Field(
        default=None,
        description=(
            "Name of the Ollama model to use for generation, e.g. 'llama3'. No safe default exists - "
            "depends entirely on which model is pulled locally. Left unset here rather than required, "
            "so AISettings() stays constructible with zero configuration for callers that never touch "
            "the LLM; OllamaProvider validates this itself when actually constructed."
        ),
    )
    embedding_model: Optional[str] = Field(
        default=None,
        description=(
            "Name of the Ollama embedding model, e.g. 'nomic-embed-text'. Must be a DEDICATED embedding "
            "model and must not be the same value as llm_model: a text-generation model cannot produce "
            "embeddings, and Ollama rejects such a request with HTTP 501 (see ai/embeddings_ollama.py, "
            "which turns that into an actionable error). Same unset-by-default rationale as llm_model - "
            "leaving it unset disables AI enrichment entirely rather than failing a scan, so unrelated "
            "AISettings() constructions never need it."
        ),
    )

    @field_validator("log_level", mode="before")
    @classmethod
    def _normalize_log_level(cls, value: object) -> object:
        return value.upper() if isinstance(value, str) else value

    @field_validator("llm_host")
    @classmethod
    def _require_local_endpoint(cls, value: str) -> str:
        """Refuse a host that is not on this machine.

        Checked here because llm_host is the single place the endpoint is
        configured, so one validator covers both the /api/generate and
        /api/embed clients that read it. The embedding path is why this
        matters most: ai/retrieval/context_retrieval.py builds its query
        from the finding's raw_evidence - repository source - so a remote
        host would send that source off the machine as a side effect of
        retrieval. See ai/local_endpoint.py for what the check does and
        does not guarantee.
        """
        return validate_local_endpoint(value)

    @model_validator(mode="after")
    def _validate_confidence_thresholds(self) -> "AISettings":
        if self.confidence_medium_threshold >= self.confidence_high_threshold:
            raise ValueError(
                "confidence_medium_threshold must be lower than confidence_high_threshold "
                f"(got medium={self.confidence_medium_threshold}, high={self.confidence_high_threshold})"
            )
        return self


def ai_models_configured() -> bool:
    """True when both model variables are present, without building AISettings.

    Exists so a scanner-only run is never refused because of a setting it will
    never read. The caller needs one fact - "did the user configure AI?" - to
    decide whether to enrich at all, and answering it by constructing AISettings
    would also run llm_host's validator, failing a scan that was never going to
    contact Ollama.

    This does not weaken anything. It is only ever used to decide *whether* to
    enrich; every path that actually reaches Ollama goes through get_settings()
    first, which validates in full. The presence test matches what the previous
    `settings.llm_model is not None` check did exactly, including treating a
    variable set to the empty string as present.
    """
    return (
        os.environ.get(_ENV_PREFIX + "LLM_MODEL") is not None
        and os.environ.get(_ENV_PREFIX + "EMBEDDING_MODEL") is not None
    )


def _settings_from_env() -> AISettings:
    """Build AISettings from SENTINELAI_AI_* environment variables, falling back to field defaults for anything unset."""
    raw = {
        field_name: value
        for field_name in AISettings.model_fields
        if (value := os.environ.get(_ENV_PREFIX + field_name.upper())) is not None
    }
    return AISettings(**raw)


_settings: Optional[AISettings] = None


def get_settings() -> AISettings:
    """Return the shared AISettings instance, built from the environment once and cached.

    Process-wide singleton: the environment is read on the first call only, so changing
    os.environ later in the same process is intentionally not picked up by later calls.

    Applies log_level to the shared "sentinelai" logger (the same logger
    main.py/live_provider.py/pipeline.py already use) as part of this
    one-time construction, not on every call - setLevel() only changes
    which records the logger *lets through*; with no handler configured
    anywhere (the default), Python's own logging.lastResort fallback still
    applies its own fixed WARNING floor regardless of this call, so this
    does not by itself introduce any new default output.
    """
    global _settings
    if _settings is None:
        _settings = _settings_from_env()
        logging.getLogger("sentinelai").setLevel(_settings.log_level.value)
    return _settings


PROVIDER_ENV = "SENTINELAI_AI_PROVIDER"
MODEL_ENV = "SENTINELAI_AI_MODEL"
EGRESS_ENV = "SENTINELAI_EGRESS"
PRIORITY_ENV = "SENTINELAI_AI_PROVIDER_PRIORITY"
FALLBACK_ENV = "SENTINELAI_AI_PROVIDER_FALLBACK"
ROLE_DISCOVERY_ENV = "SENTINELAI_AI_ROLE_DISCOVERY"
ROLE_REMEDIATION_ENV = "SENTINELAI_AI_ROLE_REMEDIATION"
ROLE_VERIFICATION_ENV = "SENTINELAI_AI_ROLE_VERIFICATION"
_ROLE_ENVS: dict[str, str] = {
    "discovery": ROLE_DISCOVERY_ENV,
    "remediation": ROLE_REMEDIATION_ENV,
    "verification": ROLE_VERIFICATION_ENV,
}
DEFAULT_PROVIDER = "ollama"
DEFAULT_EGRESS = "deny"

DEFAULT_PROVIDER_PRIORITY: tuple[str, ...] = ("gemini", "grok", "nvidia", "ollama")
# No alias mapping anymore — "grok" is now a real provider (xAI Grok via xai.py).
# "groq" remains the separate Groq provider; neither aliases the other.
PROVIDER_ALIASES: dict[str, str] = {}
# All providers in DEFAULT_PROVIDER_PRIORITY are implemented; nothing to skip.
UNIMPLEMENTED_PROVIDERS: set[str] = set()

_PROVIDER_NAMES: tuple[str, ...] = get_args(AIConfig.model_fields["provider"].annotation)
_EGRESS_VALUES: tuple[str, ...] = get_args(AIConfig.model_fields["egress"].annotation)
_FLAGS = {"provider": "--ai-provider", "model": "--ai-model", "egress": "--egress"}
_ENVS = {"provider": PROVIDER_ENV, "model": MODEL_ENV, "egress": EGRESS_ENV}


@dataclass(frozen=True)
class ProviderSpec:
    """A registered LLM provider: what it is called, what it runs by default, and what it needs to run."""

    name: str
    provider_class: type[LLMProvider]
    default_model: str
    required_env: tuple[str, ...]

    @property
    def is_local(self) -> bool:
        return self.provider_class.is_local is True

    @property
    def retains_data(self) -> Optional[bool]:
        return self.provider_class.retains_data

    @property
    def retention_basis(self) -> str:
        return self.provider_class.retention_basis

    @property
    def retention_policy_note(self) -> str:
        return self.provider_class.retention_policy_note


PROVIDERS: dict[str, ProviderSpec] = {
    spec.name: spec
    for spec in (
        ProviderSpec("ollama", OllamaProvider, "llama3.1:8b", ()),
        ProviderSpec("groq", GroqProvider, "llama-3.3-70b-versatile", (GROQ_API_KEY_ENV,)),
        ProviderSpec("gemini", GeminiProvider, "gemini-2.5-flash", (CREDENTIALS_ENV, PROJECT_ENV)),
        ProviderSpec("mock", MockLLMProvider, "mock-model", ()),
        ProviderSpec("grok", GrokProvider, "grok-4", (XAI_API_KEY_ENV,)),
        ProviderSpec("nvidia", NvidiaNimProvider, "meta/llama-3.1-70b-instruct", (NVIDIA_API_KEY_ENV,)),
    )
}
assert set(PROVIDERS) == set(_PROVIDER_NAMES), "every AIConfig provider needs exactly one registry entry"


class AIConfigError(ValueError):
    """The requested AI configuration is invalid. The message names each problem and where it came from."""


def resolve_config(cli_overrides: dict) -> AIConfig:
    """Resolve AIConfig from CLI flag > environment variable > default.

    cli_overrides holds "provider", "model" and "egress"; a missing key or a None value means the
    flag was not given. Raises AIConfigError listing every problem found, so one run reports them all.
    """
    unknown = set(cli_overrides) - set(_FLAGS)
    if unknown:
        raise ValueError(f"unknown cli_overrides keys: {sorted(unknown)}")

    problems: list[str] = []
    raw_provider_val, provider_source = _pick("provider", cli_overrides, "auto")
    egress, egress_source = _pick("egress", cli_overrides, DEFAULT_EGRESS)
    egress = egress.strip().lower()

    if egress not in _EGRESS_VALUES:
        problems.append(f"{egress_source} must be one of {', '.join(_EGRESS_VALUES)}; got {egress!r}")

    is_auto = False
    raw_provider_str = raw_provider_val.strip().lower()
    fallbacks: list[str] = []

    if provider_source == "default" or raw_provider_str == "auto":
        is_auto = True
        # Automatic provider selection in desired priority order:
        # 1. Gemini -> 2. Grok (xAI) -> 3. NVIDIA -> 4. Ollama
        # Ollama must be the final fallback, not the default first choice when a hosted provider is configured and authorized.
        if egress == "allow":
            if (
                os.environ.get(CREDENTIALS_ENV, "").strip()
                and os.environ.get(PROJECT_ENV, "").strip()
            ):
                provider = "gemini"
                fallbacks = ["grok", "nvidia", "ollama"]
            elif os.environ.get(XAI_API_KEY_ENV, "").strip():
                provider = "grok"
                fallbacks = ["nvidia", "ollama"]
            elif os.environ.get(NVIDIA_API_KEY_ENV, "").strip():
                provider = "nvidia"
                fallbacks = ["ollama"]
            else:
                provider = "ollama"
                fallbacks = []
        else:
            provider = "ollama"
            fallbacks = []
    elif "," in raw_provider_str:
        parts = [p.strip() for p in raw_provider_str.split(",") if p.strip()]
        provider = parts[0] if parts else ""
        fallbacks = parts[1:]
    else:
        provider = raw_provider_str
        if os.environ.get(FALLBACK_ENV, "").strip().lower() not in ("false", "0", "no"):
            priority_env = os.environ.get(PRIORITY_ENV, "").strip().lower()
            if priority_env:
                p_list = [p.strip() for p in priority_env.split(",") if p.strip()]
            else:
                p_list = list(DEFAULT_PROVIDER_PRIORITY)
            if provider in p_list:
                idx = p_list.index(provider)
                fallbacks = [p for p in p_list[idx + 1 :] if p in PROVIDERS and p != provider]

    spec = PROVIDERS.get(provider)
    if spec is None:
        reported_name = raw_provider_val.strip()
        problems.append(f"{provider_source} must be one of {', '.join(PROVIDERS)}; got {reported_name!r}")

    default_model = spec.default_model if spec else ""
    if provider == "ollama" and os.environ.get(_ENV_PREFIX + "LLM_MODEL", "").strip():
        default_model = os.environ[_ENV_PREFIX + "LLM_MODEL"]  # pre-multi-provider configuration
    model, model_source = _pick("model", cli_overrides, default_model)
    model = model.strip()
    if spec is not None and not model:
        problems.append(f"{model_source} must not be empty")

    if spec is not None and not spec.is_local and not is_auto:
        if egress == "deny":
            problems.append(
                f"{raw_provider_str} is a remote provider and sends code off this machine, but egress is deny "
                f"({egress_source}); pass --egress allow to opt in for this run"
            )
        missing = [name for name in spec.required_env if not os.environ.get(name, "").strip()]
        if missing:
            problems.append(f"{raw_provider_str} needs {' and '.join(missing)} set in the environment")

    from .llm.roles import DEFAULT_ROLE_CHAINS

    role_chains: dict[str, list[str]] = {}
    if any(os.environ.get(env_name, "").strip() for env_name in _ROLE_ENVS.values()):
        for role_name, env_name in _ROLE_ENVS.items():
            raw_chain = os.environ.get(env_name, "").strip()
            if raw_chain:
                chain_parts = [p.strip().lower() for p in raw_chain.split(",") if p.strip()]
                if not chain_parts:
                    problems.append(f"{env_name} must list at least one provider")
                for part in chain_parts:
                    if part not in PROVIDERS:
                        problems.append(
                            f"{env_name} must contain only {', '.join(PROVIDERS)}; got {part!r}"
                        )
                role_chains[role_name] = chain_parts
            else:
                role_chains[role_name] = list(DEFAULT_ROLE_CHAINS[role_name])
    elif is_auto:
        role_chains = {k: list(v) for k, v in DEFAULT_ROLE_CHAINS.items()}
    elif provider == "mock" and not fallbacks:
        role_chains = {
            "discovery": ["mock"],
            "remediation": ["mock"],
            "verification": ["mock"],
        }
    elif "," in raw_provider_str or os.environ.get(FALLBACK_ENV, "").strip().lower() in ("false", "0", "no"):
        explicit_chain = [provider, *[p for p in fallbacks if p != provider]] if provider else []
        role_chains = {
            "discovery": list(explicit_chain),
            "remediation": list(explicit_chain),
            "verification": list(explicit_chain),
        }
    elif provider in PROVIDERS:
        disc_chain = [provider, *[p for p in fallbacks if p != provider]]
        rem_chain = [
            provider,
            *[p for p in DEFAULT_ROLE_CHAINS["remediation"] if p != provider and p in PROVIDERS],
        ]
        ver_chain = list(DEFAULT_ROLE_CHAINS["verification"])
        role_chains = {
            "discovery": disc_chain,
            "remediation": rem_chain,
            "verification": ver_chain,
        }

    if problems:
        raise AIConfigError("; ".join(problems))
    return AIConfig(
        provider=provider,
        model=model,
        egress=egress,
        fallback_providers=fallbacks,
        role_chains=role_chains,
    )


def _pick(field: str, cli_overrides: dict, default: str) -> tuple[str, str]:
    """The winning value for one field and a label for where it came from, for error messages."""
    flag_value = cli_overrides.get(field)
    if flag_value is not None:
        return flag_value, _FLAGS[field]
    env_value = os.environ.get(_ENVS[field], "")
    if env_value.strip():
        return env_value, _ENVS[field]
    return default, "default"
