"""
Shared client for hosted providers that expose an OpenAI-compatible
chat completions endpoint (native xAI Grok, NVIDIA NIM).

Same discipline as ai/llm/groq.py, which predates this module and is left
as it is:
- The API key is read from one environment variable at construction and
  from nowhere else - there is no api_key parameter, so a key cannot
  arrive from a config file or a CLI flag. It is held on a private
  attribute, sent only in the Authorization header, and kept out of
  __repr__, log records and every ProviderError.
- The endpoint is a fixed class attribute, not configuration: these are
  remote services, and a user-supplied URL would be an unvalidated egress
  destination.
- Transport goes through ai/llm/_http.post_json, so the hard deadline,
  response-size cap and payload-free error mapping are shared with the
  other hosted providers.

The presence of the environment variable is checked here; whether the key
actually works is only known when the provider answers (401/403 becomes
ProviderErrorKind.AUTH).
"""
import logging
import os
from typing import ClassVar, Optional

from ._http import optional_count, post_json, validate_timeout
from .base import LLMProvider, LLMResponse
from .errors import ProviderError, ProviderErrorKind

logger = logging.getLogger("sentinelai")

DEFAULT_TIMEOUT_S = 60.0


class OpenAICompatibleProvider(LLMProvider):
    """Base for a hosted chat-completions provider. Subclasses set name, ENDPOINT and API_KEY_ENV."""

    is_local: ClassVar[bool] = False
    retains_data: ClassVar[Optional[bool]] = None
    ENDPOINT: ClassVar[str]
    API_KEY_ENV: ClassVar[str]

    def __init__(self, model: str, *, timeout_s: float = DEFAULT_TIMEOUT_S) -> None:
        if not model:
            raise ProviderError(self.name, ProviderErrorKind.CONFIGURATION, "no model configured")
        api_key = os.environ.get(self.API_KEY_ENV, "").strip()
        if not api_key:
            raise ProviderError(self.name, ProviderErrorKind.CONFIGURATION, f"{self.API_KEY_ENV} is not set")
        self.model = model
        self._api_key = api_key
        self._timeout_s = validate_timeout(timeout_s)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(model={self.model!r})"

    def complete(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.1,
    ) -> LLMResponse:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        headers = {"Authorization": f"Bearer {self._api_key}"}
        body, latency_ms = post_json(self.name, self.ENDPOINT, payload, headers, self._timeout_s)
        try:
            text = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            text = None
        if not isinstance(text, str):
            raise ProviderError(self.name, ProviderErrorKind.INVALID_RESPONSE, "no message content in choices[0]")
        usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
        logger.debug("%s completion: model=%s latency_ms=%.0f", self.name, self.model, latency_ms)
        return LLMResponse(
            text=text,
            model=body.get("model") if isinstance(body.get("model"), str) else self.model,
            prompt_tokens=optional_count(usage.get("prompt_tokens")),
            completion_tokens=optional_count(usage.get("completion_tokens")),
            latency_ms=latency_ms,
        )

