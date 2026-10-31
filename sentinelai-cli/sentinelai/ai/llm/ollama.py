"""
Ollama provider for the multi-provider interface (ai/llm/base.py).

Separate from ai/llm_ollama.py, which stays as the existing pipeline's
generate() -> str client until Phase 7 rewires it.
"""
import ipaddress
import logging
import os
import urllib.parse
from typing import ClassVar, Optional

from ._http import optional_count, post_json, validate_timeout
from .base import LLMProvider, LLMResponse, ProviderError
from .errors import ProviderErrorKind

logger = logging.getLogger("sentinelai")

DEFAULT_HOST = "http://localhost:11434"
DEFAULT_TIMEOUT_S = 120.0


def _is_loopback(host: str) -> bool:
    parsed = urllib.parse.urlsplit(host)
    hostname = parsed.hostname
    if hostname is None:
        return False
    if hostname in ("localhost", "127.0.0.1", "::1"):
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


class OllamaProvider(LLMProvider):
    name: ClassVar[str] = "ollama"
    is_local: ClassVar[bool] = True
    retains_data: ClassVar[Optional[bool]] = False
    retention_basis: ClassVar[str] = "local_loopback_unverified_host_persistence"
    retention_policy_note: ClassVar[str] = (
        "Processed locally via loopback endpoint (no remote vendor transmission); does not "
        "verify the absence of local server logs, model cache, or host disk persistence."
    )

    def __init__(
        self,
        model: str = "llama3.1:8b",
        *,
        base_url: Optional[str] = None,
        host: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT_S,
        timeout_s: Optional[float] = None,
    ) -> None:
        if not model:
            raise ProviderError(self.name, ProviderErrorKind.CONFIGURATION, "no model configured")

        raw_host = base_url or host or os.environ.get("OLLAMA_HOST", DEFAULT_HOST)
        if not raw_host.startswith("http://") and not raw_host.startswith("https://"):
            raw_host = f"http://{raw_host}"

        if not _is_loopback(raw_host):
            raise ProviderError(
                self.name,
                ProviderErrorKind.CONFIGURATION,
                "host must be a loopback address (localhost, 127.0.0.1, ::1)",
            )
        self.model = model
        self.base_url = raw_host.rstrip("/")
        self._url = f"{self.base_url}/api/generate"
        eff_timeout = timeout_s if timeout_s is not None else timeout
        self._timeout_s = validate_timeout(eff_timeout)

    def __repr__(self) -> str:
        return f"OllamaProvider(model={self.model!r})"

    def complete(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.1,
    ) -> LLMResponse:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": max_tokens, "temperature": temperature},
        }
        if system:
            payload["system"] = system

        body, latency_ms = post_json(self.name, self._url, payload, {}, self._timeout_s)

        if isinstance(body, dict) and "error" in body:
            err_msg = str(body["error"])
            if "not found" in err_msg.lower():
                raise ProviderError(self.name, ProviderErrorKind.INVALID_RESPONSE, "model not found")
            raise ProviderError(self.name, ProviderErrorKind.INVALID_RESPONSE, err_msg)

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
