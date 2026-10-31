"""Provider-neutral LLM interface for local and remote providers (Ollama, Groq, Gemini, xAI Grok, NVIDIA, mock)."""
from .base import LLMProvider, LLMProviderError, LLMResponse, ProviderError
from .errors import ProviderErrorKind
from ._openai_compat import OpenAICompatibleProvider
from .fallback import (
    DEFAULT_PROVIDER_PRIORITY,
    FallbackLLMProvider,
    ProviderCandidate,
    create_fallback_provider,
)
from .mock import MockLLMProvider
from .nvidia import NvidiaNimProvider
from .xai import GrokProvider

__all__ = [
    "LLMProvider",
    "LLMProviderError",
    "LLMResponse",
    "MockLLMProvider",
    "OpenAICompatibleProvider",
    "GrokProvider",
    "NvidiaNimProvider",
    "ProviderError",
    "ProviderErrorKind",
    "FallbackLLMProvider",
    "ProviderCandidate",
    "create_fallback_provider",
    "DEFAULT_PROVIDER_PRIORITY",
]
