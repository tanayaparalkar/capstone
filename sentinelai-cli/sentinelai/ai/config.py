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
"""
import logging
import os
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator

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

    @model_validator(mode="after")
    def _validate_confidence_thresholds(self) -> "AISettings":
        if self.confidence_medium_threshold >= self.confidence_high_threshold:
            raise ValueError(
                "confidence_medium_threshold must be lower than confidence_high_threshold "
                f"(got medium={self.confidence_medium_threshold}, high={self.confidence_high_threshold})"
            )
        return self


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
