"""
LLM provider interface for the multi-provider AI layer.

Interfaces only - no implementation lives here. Concrete providers
(Ollama, Groq, Gemini on Vertex AI) subclass LLMProvider in later
phases; MockLLMProvider (ai/llm/mock.py) is the one every phase
develops against until then.
"""
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class LLMResponse(BaseModel):
    """One complete (never partial/streamed) completion, plus whatever usage metadata the provider reported."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    text: str = Field(..., description="Complete text of the completion.")
    model: str = Field(..., description="Model identifier as reported by the provider.")
    latency_ms: float = Field(0.0, ge=0.0, description="Wall-clock latency of the call in ms.")
    prompt_tokens: Optional[int] = Field(None, ge=0, description="Prompt tokens, if the provider reports them.")
    completion_tokens: Optional[int] = Field(
        None, ge=0, description="Completion tokens, if the provider reports them."
    )

    @field_validator("latency_ms", mode="before")
    @classmethod
    def _coerce_latency(cls, v: Any) -> float:
        if v is None:
            return 0.0
        return float(v)


class LLMProvider(ABC):
    """Source of a completion for a prompt, swappable behind this interface."""

    name: ClassVar[str]
    is_local: ClassVar[bool]
    retains_data: ClassVar[Optional[bool]]

    @abstractmethod
    def complete(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.1,
    ) -> LLMResponse:
        """Return the complete response for prompt. Raise ProviderError on failure."""
        raise NotImplementedError


class ProviderError(Exception):
    """An LLM provider call failed for an operational reason. Safe to log and record in scan metadata.

    Never includes request bodies or API keys in the message.
    """

    def __init__(
        self,
        provider: str,
        reason: Any,
        detail: Optional[str] = None,
        *,
        status: Optional[int] = None,
        retry_after: Optional[float] = None,
    ) -> None:
        self.provider = provider
        self.status = status
        self.retry_after = retry_after

        if hasattr(reason, "value"):
            self.kind = reason
            r = str(reason.value)
            if detail:
                r = f"{r}: {detail}"
            self.reason = r
        else:
            self.kind = getattr(reason, "kind", None)
            self.reason = str(reason)
            if detail:
                self.reason = f"{self.reason}: {detail}"

        message = f"{provider}: {self.reason}"
        if status is not None and f"HTTP {status}" not in message:
            message += f" (HTTP {status})"
        super().__init__(message)
