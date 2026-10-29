"""
Ollama provider for the multi-provider interface (ai/llm/base.py).

Separate from ai/llm_ollama.py, which stays as the existing pipeline's
generate() -> str client until Phase 7 rewires it.

is_local=True is a privacy claim the egress policy relies on, so it is
enforced rather than assumed: the host must be a loopback address. An
Ollama server on another machine receives the source code over the
network, and is not "local" in the sense that matters here.

retains_data=False: the request goes to a process on the developer's own
machine. Ollama's /api/generate keeps no conversation history for a
non-streaming request without a `context` field, and none is sent.
"""
import ipaddress
import logging
import urllib.parse
from typing import ClassVar, Optional

from ._http import optional_count, post_json, validate_timeout
from .base import LLMProvider, LLMResponse
from .errors import ProviderError, ProviderErrorKind

logger = logging.getLogger("sentinelai")

DEFAULT_HOST = "http://localhost:11434"
DEFAULT_TIMEOUT_S = 120.0


def _is_loopback(host: str) -> bool:
    hostname = urllib.parse.urlsplit(host).hostname
    if hostname is None:
        return False
    if hostname == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


class OllamaProvider(LLMProvider):
    name: ClassVar[str] = "ollama"
    is_local: ClassVar[bool] = True
    retains_data: ClassVar[Optional[bool]] = False

    def __init__(self, model: str, *, host: str = DEFAULT_HOST, timeout_s: float = DEFAULT_TIMEOUT_S) -> None:
        if not model:
            raise ProviderError(self.name, ProviderErrorKind.CONFIGURATION, "no model configured")
        if not _is_loopback(host):
            raise ProviderError(
                self.name, ProviderErrorKind.CONFIGURATION, "host must be a loopback address (localhost, 127.0.0.1, ::1)"
            )
        self.model = model
        self._url = host.rstrip("/") + "/api/generate"
        self._timeout_s = validate_timeout(timeout_s)

    def __repr__(self) -> str:
        return f"OllamaProvider(model={self.model!r})"

    def complete(self, prompt: str, *, max_tokens: int, temperature: float) -> LLMResponse:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": max_tokens, "temperature": temperature},
        }
        body, latency_ms = post_json(self.name, self._url, payload, {}, self._timeout_s)
        text = body.get("response") if isinstance(body, dict) else None
        if not isinstance(text, str):
            raise ProviderError(self.name, ProviderErrorKind.INVALID_RESPONSE, "no 'response' text in body")
        logger.debug("ollama completion: model=%s latency_ms=%.0f", self.model, latency_ms)
        return LLMResponse(
            text=text,
            model=body.get("model") if isinstance(body.get("model"), str) else self.model,
            prompt_tokens=optional_count(body.get("prompt_eval_count")),
            completion_tokens=optional_count(body.get("eval_count")),
            latency_ms=latency_ms,
        )
