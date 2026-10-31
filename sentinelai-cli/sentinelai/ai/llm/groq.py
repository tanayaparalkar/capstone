"""
Groq provider: the OpenAI-compatible chat completions endpoint.

The API key is read from GROQ_API_KEY when the provider is constructed,
and from nowhere else - there is no api_key parameter, so it cannot
arrive from a config file or a CLI flag. It is held on a private
attribute, sent only in the Authorization header, and kept out of
__repr__, log records and every ProviderError.

retains_data is None, meaning "not verified", deliberately.
TODO(retains_data): read Groq's current data-handling terms before
setting a value - the "Your Data in GroqCloud" page
(https://console.groq.com/docs/your-data), the GroqCloud Services
Agreement, and the Groq Privacy Policy - and record which account tier
the answer applies to. Zero data retention is generally tied to a
commercial agreement rather than the free developer tier, so the value
may differ by account; until it is verified for the account in use, the
egress policy must treat None as "retains".
"""
import logging
import os
from typing import ClassVar, Optional

from ._http import optional_count, post_json, validate_timeout
from .base import LLMProvider, LLMResponse
from .errors import ProviderError, ProviderErrorKind

logger = logging.getLogger("sentinelai")

API_KEY_ENV = "GROQ_API_KEY"
ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_TIMEOUT_S = 60.0


class GroqProvider(LLMProvider):
    name: ClassVar[str] = "groq"
    is_local: ClassVar[bool] = False
    retains_data: ClassVar[Optional[bool]] = None
    retention_basis: ClassVar[str] = "unverified_account_zdr_or_default_logging"
    retention_policy_note: ClassVar[str] = (
        "Groq documents limited default inference retention for reliability and abuse logging, "
        "with configurable Zero Data Retention; actual account configuration cannot be verified "
        "at runtime."
    )

    def __init__(self, model: str, *, timeout_s: float = DEFAULT_TIMEOUT_S) -> None:
        if not model:
            raise ProviderError(self.name, ProviderErrorKind.CONFIGURATION, "no model configured")
        # GROQ_API_KEY only. Groq is not xAI's Grok; see ai/llm/xai.py.
        api_key = os.environ.get(API_KEY_ENV, "").strip()
        if not api_key:
            raise ProviderError(self.name, ProviderErrorKind.CONFIGURATION, f"{API_KEY_ENV} is not set")
        self.model = model
        self._api_key = api_key
        self._timeout_s = validate_timeout(timeout_s)

    def __repr__(self) -> str:
        return f"GroqProvider(model={self.model!r})"

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
            "max_completion_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        headers = {"Authorization": f"Bearer {self._api_key}"}
        body, latency_ms = post_json(self.name, ENDPOINT, payload, headers, self._timeout_s)
        try:
            text = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            text = None
        if not isinstance(text, str):
            raise ProviderError(self.name, ProviderErrorKind.INVALID_RESPONSE, "no message content in choices[0]")
        usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
        logger.debug("groq completion: model=%s latency_ms=%.0f", self.model, latency_ms)
        return LLMResponse(
            text=text,
            model=body.get("model") if isinstance(body.get("model"), str) else self.model,
            prompt_tokens=optional_count(usage.get("prompt_tokens")),
            completion_tokens=optional_count(usage.get("completion_tokens")),
            latency_ms=latency_ms,
        )
