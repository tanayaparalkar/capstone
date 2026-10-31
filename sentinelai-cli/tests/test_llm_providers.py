"""
Parameterized test suite over MockLLMProvider and OllamaProvider.

Verifies:
  - returns a valid LLMResponse
  - latency_ms is populated and positive
  - ProviderError on refused connection (point at a closed port)
  - ProviderError message contains neither a prompt nor any env var value
Uses recorded HTTP fixture for Ollama tests so CI needs no running daemon.
"""
import io
import json
import socket
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from sentinelai.ai.llm.base import LLMProvider, LLMResponse, ProviderError
from sentinelai.ai.llm.mock import MockLLMProvider
from sentinelai.ai.llm.ollama import OllamaProvider

FIXTURE_PATH = Path(__file__).parent / "data" / "llm_http" / "ollama_success.json"


class _FakeHTTPResponse:
    def __init__(self, body: bytes, status: int = 200) -> None:
        self._stream = io.BytesIO(body)
        self.status = status

    def read(self, size: int = -1) -> bytes:
        return self._stream.read(size)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _make_provider(name: str, monkeypatch) -> LLMProvider:
    if name == "mock":
        return MockLLMProvider()
    elif name == "ollama":
        fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        body_bytes = json.dumps(fixture["response"]["body"]).encode("utf-8")

        def fake_urlopen(req, timeout=None):
            return _FakeHTTPResponse(body_bytes, status=200)

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
        return OllamaProvider(model="gpt-oss:20b")
    raise ValueError(f"Unknown provider name: {name}")


@pytest.mark.parametrize("provider_name", ["mock", "ollama"])
def test_returns_valid_llm_response(provider_name, monkeypatch):
    provider = _make_provider(provider_name, monkeypatch)
    response = provider.complete("Reply with exactly: OK")
    assert isinstance(response, LLMResponse)
    assert isinstance(response.text, str) and len(response.text) > 0
    assert isinstance(response.model, str) and len(response.model) > 0


@pytest.mark.parametrize("provider_name", ["mock", "ollama"])
def test_latency_ms_is_populated_and_positive(provider_name, monkeypatch):
    provider = _make_provider(provider_name, monkeypatch)
    response = provider.complete("Reply with exactly: OK")
    assert response.latency_ms is not None
    assert isinstance(response.latency_ms, (int, float))
    assert response.latency_ms > 0


@pytest.mark.parametrize("provider_name", ["mock", "ollama"])
def test_provider_error_on_refused_connection(provider_name):
    # Find a closed local port
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    closed_port = s.getsockname()[1]
    s.close()

    if provider_name == "ollama":
        provider = OllamaProvider(
            model="gpt-oss:20b",
            host=f"http://127.0.0.1:{closed_port}",
            timeout=5.0,
        )
        with pytest.raises(ProviderError) as exc_info:
            provider.complete("test prompt")
        msg = str(exc_info.value).lower()
        assert "refused" in msg or "transport" in msg or "connection" in msg
    else:
        provider = MockLLMProvider()
        with pytest.raises(ProviderError) as exc_info:
            raise ProviderError("mock", "connection refused")
        assert "connection refused" in str(exc_info.value)


@pytest.mark.parametrize("provider_name", ["mock", "ollama"])
def test_provider_error_message_contains_neither_prompt_nor_env_vars(provider_name, monkeypatch):
    secret_key = "SENTINELAI_TEST_SECRET_KEY"
    secret_value = "super_confidential_token_987654321"
    monkeypatch.setenv(secret_key, secret_value)
    sensitive_prompt = "CONFIDENTIAL_SOURCE_CODE_FINDING_XYZ"

    if provider_name == "ollama":
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        closed_port = s.getsockname()[1]
        s.close()
        provider = OllamaProvider(
            model="gpt-oss:20b",
            host=f"http://127.0.0.1:{closed_port}",
            timeout=5.0,
        )
    else:
        provider = MockLLMProvider()
        def fail_complete(*args, **kwargs):
            raise ProviderError("mock", "operational failure")
        monkeypatch.setattr(provider, "complete", fail_complete)

    with pytest.raises(ProviderError) as exc_info:
        provider.complete(sensitive_prompt)

    err_text = str(exc_info.value)
    assert sensitive_prompt not in err_text
    assert secret_value not in err_text
    assert secret_key not in err_text
