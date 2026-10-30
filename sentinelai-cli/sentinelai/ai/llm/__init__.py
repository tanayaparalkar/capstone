"""Provider-neutral LLM interface for local and remote providers (Ollama, Groq, Gemini, mock)."""
from .base import LLMProvider, LLMResponse
from .mock import MockLLMProvider

__all__ = ["LLMProvider", "LLMResponse", "MockLLMProvider"]
