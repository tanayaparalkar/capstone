"""
LLM provider interface for the multi-provider AI layer.

Interfaces only - no implementation lives here. Concrete providers
(Ollama, Groq, Gemini on Vertex AI) subclass LLMProvider in later
phases; MockLLMProvider (ai/llm/mock.py) is the one every phase
develops against until then.

Separate from ai/llm_base.py's LLMProvider and ai/llm_response.py's
LLMResponse, which are left untouched: those are the existing
Ollama-only generate(prompt) -> str contract and the parsed agent
output respectively. This LLMResponse is the raw completion plus usage
metadata, before any parsing.

Every provider declares three privacy-relevant class attributes so a
caller can decide whether a prompt may leave the machine *before*
calling complete():

    name          stable identifier, matches AIConfig.provider
    is_local      True when the prompt never leaves the developer's machine
    retains_data  whether the provider retains prompts; None means "not
                  yet verified" and must be treated as retaining (fail
                  closed) - never filled in with a guess

Synchronous, matching the rest of the pipeline. Failures propagate as
exceptions; they are operational faults for scan metadata, never
security findings.
"""
from abc import ABC, abstractmethod
from typing import ClassVar, Optional

from pydantic import BaseModel, ConfigDict, Field


class LLMResponse(BaseModel):
    """One complete (never partial/streamed) completion, plus whatever usage metadata the provider reported."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str = Field(..., description="Complete text of the completion.")
    model: str = Field(..., description="Model identifier as reported by the provider.")
    prompt_tokens: Optional[int] = Field(None, ge=0, description="Prompt tokens, if the provider reports them.")
    completion_tokens: Optional[int] = Field(
        None, ge=0, description="Completion tokens, if the provider reports them."
    )
    latency_ms: Optional[float] = Field(None, ge=0, description="Wall-clock latency of the call, if measured.")


class LLMProvider(ABC):
    """Source of a completion for a prompt, swappable behind this interface."""

    name: ClassVar[str]
    is_local: ClassVar[bool]
    retains_data: ClassVar[Optional[bool]]

    @abstractmethod
    def complete(self, prompt: str, *, max_tokens: int, temperature: float) -> LLMResponse:
        """Return the complete response for prompt. Raise on failure."""
        raise NotImplementedError
