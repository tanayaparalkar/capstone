"""
Loopback-only endpoint validation on the legacy Ollama generation provider
(sentinelai/ai/llm_ollama.py).

AISettings.llm_host validates the configured URL, and embeddings_ollama.py
re-checks its own `host` parameter, so direct construction of
sentinelai.ai.llm_ollama.OllamaProvider must apply the exact same
validate_local_endpoint() check before any network request can be built or
sent.
"""
import json
from unittest.mock import patch

import pytest

from sentinelai.ai.config import AISettings
from sentinelai.ai.factory import create_llm_generate_fn
from sentinelai.ai.llm_ollama import OllamaProvider
from sentinelai.ai.local_endpoint import RemoteEndpointError

MODEL = "llama3.1:8b"
_GOOD_BODY = json.dumps({"response": '{"ok": true}'})


class _FakeResponse:
    def __init__(self, body: str):
        self._body = body.encode("utf-8")

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


# ----------------------------------------- accepted local & loopback forms


@pytest.mark.parametrize(
    "host",
    [
        "http://localhost:11434",
        "http://localhost",
        "https://localhost:11434",
        "HTTP://LOCALHOST:11434",  # case-insensitive scheme and hostname
        "http://127.0.0.1:11434",
        "http://127.0.0.1",
        "http://127.0.0.2:11434",
        "http://127.5.5.5:11434",  # full 127.0.0.0/8 loopback block
        "http://127.255.255.254:11434",
        "http://[::1]:11434",  # bracketed IPv6 loopback
        "http://[::1]",
        "http://[0:0:0:0:0:0:0:1]:11434",  # uncompressed IPv6 loopback
        "http://localhost:11434/",
    ],
)
def test_legacy_provider_accepts_supported_loopback_endpoints(host):
    provider = OllamaProvider(host=host, model=MODEL)
    assert provider._host == host


def test_legacy_provider_strips_surrounding_whitespace_from_valid_host():
    provider = OllamaProvider(host="  http://localhost:11434  ", model=MODEL)
    assert provider._host == "http://localhost:11434"

    with patch("urllib.request.urlopen", return_value=_FakeResponse(_GOOD_BODY)) as urlopen:
        result = provider.generate("prompt")

    assert result == '{"ok": true}'
    request = urlopen.call_args.args[0]
    assert request.full_url == "http://localhost:11434/api/generate"


@pytest.mark.parametrize(
    "host,expected_url",
    [
        ("http://localhost:11434", "http://localhost:11434/api/generate"),
        ("http://127.0.0.1:11434", "http://127.0.0.1:11434/api/generate"),
        ("http://[::1]:11434", "http://[::1]:11434/api/generate"),
    ],
)
def test_legacy_provider_generates_against_accepted_loopback_forms(host, expected_url):
    provider = OllamaProvider(host=host, model=MODEL)
    with patch("urllib.request.urlopen", return_value=_FakeResponse(_GOOD_BODY)) as urlopen:
        result = provider.generate("prompt")

    assert result == '{"ok": true}'
    assert urlopen.call_count == 1
    assert urlopen.call_args.args[0].full_url == expected_url


# ----------------------------------------------- rejected remote endpoints


@pytest.mark.parametrize(
    "host",
    [
        "http://ollama.example.com:11434",
        "https://api.openai.com",
        "http://localhost.evil.com:11434",  # hostname prefix trick
        "http://notlocalhost:11434",
        "http://127.0.0.1.evil.com:11434",
        "http://localhost@evil.com:11434",  # userinfo spoofing
        "http://127.0.0.1@evil.com:11434",
        "http://8.8.8.8:11434",  # public IPv4
        "http://192.168.1.50:11434",  # RFC1918 private LAN, not loopback
        "http://10.0.0.5:11434",
        "http://172.16.0.1:11434",
        "http://169.254.169.254",  # link-local / cloud metadata
        "http://169.254.169.254:11434",
        "http://0.0.0.0:11434",  # unspecified IPv4, not loopback
        "http://[::]:11434",  # unspecified IPv6, not loopback
        "http://[2001:4860:4860::8888]:11434",  # public IPv6
        "http://[fe80::1]:11434",  # link-local IPv6
    ],
)
def test_legacy_provider_rejects_remote_or_non_loopback_endpoints(host):
    with patch("urllib.request.urlopen") as urlopen:
        with pytest.raises(RemoteEndpointError) as excinfo:
            OllamaProvider(host=host, model=MODEL)

    assert isinstance(excinfo.value, ValueError)
    urlopen.assert_not_called()


def test_rejected_endpoint_error_does_not_echo_url_credentials():
    with patch("urllib.request.urlopen") as urlopen:
        with pytest.raises(RemoteEndpointError) as excinfo:
            OllamaProvider(
                host="http://user:s3cret-token@ollama.example.com:11434/api/generate",
                model=MODEL,
            )

    message = str(excinfo.value)
    assert "ollama.example.com" in message
    assert "http://localhost:11434" in message
    assert "s3cret-token" not in message
    assert "user:" not in message
    urlopen.assert_not_called()


# -------------------------------------------- rejected malformed endpoints


@pytest.mark.parametrize(
    "host",
    [
        "ftp://localhost:11434",
        "file:///etc/passwd",
        "ws://localhost:11434",
        "localhost:11434",  # missing scheme
        "127.0.0.1:11434",  # missing scheme
        "//localhost:11434",  # scheme-relative URL
        "http://[::1:11434",  # unterminated IPv6 literal
        "http://::1:11434",  # unbracketed IPv6 literal
        "http://localhost:notaport",
        "http://localhost:99999",  # out-of-range port
        "http://",
        "http:///api/generate",  # missing authority/host
        "not a url at all",
    ],
)
def test_legacy_provider_rejects_malformed_or_bad_scheme_endpoints(host):
    with patch("urllib.request.urlopen") as urlopen:
        with pytest.raises(RemoteEndpointError) as excinfo:
            OllamaProvider(host=host, model=MODEL)

    assert isinstance(excinfo.value, ValueError)
    urlopen.assert_not_called()


@pytest.mark.parametrize("host", ["", "   ", "\t\n", None, 11434])
def test_legacy_provider_rejects_empty_whitespace_or_non_string_host(host):
    """Preserves ValueError compatibility while raising RemoteEndpointError."""
    with patch("urllib.request.urlopen") as urlopen:
        with pytest.raises(RemoteEndpointError) as excinfo:
            OllamaProvider(host=host, model=MODEL)  # type: ignore[arg-type]

    assert isinstance(excinfo.value, ValueError)
    urlopen.assert_not_called()


# ------------------------ legacy factory & remaining constructor contracts


def test_legacy_factory_remains_functional_for_local_settings(monkeypatch):
    monkeypatch.setattr(
        "sentinelai.ai.factory.get_settings",
        lambda: AISettings(llm_host="http://127.0.0.1:11434", llm_model=MODEL),
    )
    generate = create_llm_generate_fn()
    with patch("urllib.request.urlopen", return_value=_FakeResponse(_GOOD_BODY)) as urlopen:
        assert generate("prompt") == '{"ok": true}'

    assert urlopen.call_count == 1


def test_legacy_factory_blocks_unvalidated_remote_settings_at_provider_init(monkeypatch):
    """Even if AISettings validation is bypassed via model_construct, OllamaProvider refuses."""
    unvalidated = AISettings.model_construct(
        llm_host="http://ollama.example.com:11434",
        llm_model=MODEL,
        request_timeout_seconds=120.0,
        max_attempts=2,
    )
    monkeypatch.setattr("sentinelai.ai.factory.get_settings", lambda: unvalidated)

    with patch("urllib.request.urlopen") as urlopen:
        with pytest.raises(RemoteEndpointError):
            create_llm_generate_fn()

    urlopen.assert_not_called()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"model": ""},
        {"timeout": 0},
        {"timeout": -5},
        {"max_attempts": 0},
        {"max_attempts": -1},
    ],
)
def test_other_constructor_validations_remain_unchanged(kwargs):
    params = {"host": "http://localhost:11434", "model": MODEL, **kwargs}
    with pytest.raises(ValueError):
        OllamaProvider(**params)

