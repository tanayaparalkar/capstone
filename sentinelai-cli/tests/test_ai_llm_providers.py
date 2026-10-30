"""LLM providers: one contract suite over all four, plus transport, failure and secrecy tests for the HTTP three.

No network and no API keys: urllib.request.urlopen is replaced by a replay
of the cassettes in tests/data/llm_http/. Those cassettes are hand-authored
from each provider's published response schema, not captured from live
calls - see the "note" field in each.
"""
import email.message
import inspect
import io
import json
import logging
import socket
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import pytest
from redaction_fixtures import FAKE_GEMINI_TOKEN, FAKE_GROQ_KEY

from sentinelai.ai.llm import LLMProvider, LLMResponse, MockLLMProvider
from sentinelai.ai.llm.errors import ProviderError, ProviderErrorKind
from sentinelai.ai.llm.gemini import GeminiProvider
from sentinelai.ai.llm.groq import GroqProvider
from sentinelai.ai.llm.ollama import OllamaProvider

CASSETTES = Path(__file__).parent / "data" / "llm_http"
GROQ_KEY = FAKE_GROQ_KEY
GEMINI_TOKEN = FAKE_GEMINI_TOKEN
SECRETS = (GROQ_KEY, GEMINI_TOKEN)
PROMPT = 'Explain finding SENT-001.\nCode: password = "«REDACTED:secret-assignment»"\n'
HTTP_PROVIDERS = ("ollama", "groq", "gemini")
ALL_PROVIDERS = ("mock",) + HTTP_PROVIDERS


# ---- replay harness ----


def _load(name: str) -> dict:
    return json.loads((CASSETTES / name).read_text(encoding="utf-8"))


def _substitute(body: object) -> bytes:
    text = body if isinstance(body, str) else json.dumps(body)
    return text.replace("{{API_KEY}}", GROQ_KEY + " " + GEMINI_TOKEN).replace("{{PROMPT}}", PROMPT).encode("utf-8")


class _Response:
    def __init__(self, body: bytes, status: int = 200):
        self._stream = io.BytesIO(body)
        self.status = status

    def read(self, size: int = -1) -> bytes:
        return self._stream.read(size)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _http_error(url: str, status: int, body: bytes, headers: Optional[dict] = None) -> urllib.error.HTTPError:
    message = email.message.Message()
    for key, value in (headers or {}).items():
        message[key] = value
    return urllib.error.HTTPError(url, status, "error", message, io.BytesIO(body))


@dataclass
class Replay:
    """Stands in for urlopen: checks each request against the cassette and serves its response."""

    cassette: dict
    requests: list = field(default_factory=list)
    timeouts: list = field(default_factory=list)
    status: Optional[int] = None  # overrides the cassette's status
    headers: Optional[dict] = None
    body: Optional[bytes] = None  # overrides the cassette's body
    raises: Optional[BaseException] = None

    def __call__(self, request: urllib.request.Request, timeout=None):
        self.requests.append(request)
        self.timeouts.append(timeout)
        expected = self.cassette["request"]
        assert request.get_method() == expected["method"]
        assert request.full_url == expected["url"]
        if self.raises is not None:
            raise self.raises
        response = self.cassette["response"]
        status = self.status if self.status is not None else response["status"]
        body = self.body if self.body is not None else _substitute(response["body"])
        if status >= 400:
            raise _http_error(request.full_url, status, body, self.headers or response.get("headers"))
        return _Response(body, status)

    def sent_json(self) -> list:
        return [json.loads(request.data) for request in self.requests]


_PROMPT_IN_BODY: dict[str, Callable[[dict], str]] = {
    "ollama": lambda body: body["prompt"],
    "groq": lambda body: body["messages"][0]["content"],
    "gemini": lambda body: body["contents"][0]["parts"][0]["text"],
}


def _make(name: str, monkeypatch, timeout_s: float = 5.0) -> LLMProvider:
    if name == "mock":
        return MockLLMProvider()
    if name == "ollama":
        return OllamaProvider("llama3.1:8b", timeout_s=timeout_s)
    if name == "groq":
        monkeypatch.setenv("GROQ_API_KEY", GROQ_KEY)
        return GroqProvider("llama-3.3-70b-versatile", timeout_s=timeout_s)
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    return GeminiProvider(
        "gemini-2.5-flash", project="sentinel-test", location="us-central1", timeout_s=timeout_s,
        token_provider=lambda: GEMINI_TOKEN,
    )


@dataclass
class Backend:
    name: str
    provider: LLMProvider
    replay: Optional[Replay]

    def sent_prompts(self) -> list:
        if self.replay is None:
            return list(self.provider.calls)
        return [_PROMPT_IN_BODY[self.name](body) for body in self.replay.sent_json()]


def _backend(name: str, monkeypatch, cassette: str = "success", **replay_overrides) -> Backend:
    provider = _make(name, monkeypatch, timeout_s=replay_overrides.pop("timeout_s", 5.0))
    replay = None
    if name != "mock":
        replay = Replay(_load(f"{name}_{cassette}.json"), **replay_overrides)
        monkeypatch.setattr(urllib.request, "urlopen", replay)
    return Backend(name, provider, replay)


@pytest.fixture(params=ALL_PROVIDERS)
def backend(request, monkeypatch):
    return _backend(request.param, monkeypatch)


@pytest.fixture(params=HTTP_PROVIDERS)
def http_name(request):
    return request.param


def _complete(provider: LLMProvider, prompt: str = PROMPT) -> LLMResponse:
    return provider.complete(prompt, max_tokens=256, temperature=0.1)


# ---- contract suite: identical interface behaviour across all four ----


def test_is_an_llm_provider_with_declared_privacy_attributes(backend):
    provider = backend.provider
    assert isinstance(provider, LLMProvider)
    assert provider.name == backend.name
    assert isinstance(provider.is_local, bool)
    assert provider.retains_data in (True, False, None)


def test_complete_has_exactly_the_base_signature(backend):
    assert inspect.signature(type(backend.provider).complete) == inspect.signature(LLMProvider.complete)


def test_max_tokens_and_temperature_are_keyword_only(backend):
    with pytest.raises(TypeError):
        backend.provider.complete(PROMPT, 256, 0.1)


def test_complete_returns_a_complete_llm_response(backend):
    response = _complete(backend.provider)
    assert isinstance(response, LLMResponse)
    assert isinstance(response.text, str) and response.text
    assert isinstance(response.model, str) and response.model
    for count in (response.prompt_tokens, response.completion_tokens):
        assert count is None or (isinstance(count, int) and count >= 0)
    assert response.latency_ms is None or response.latency_ms >= 0


def test_every_prompt_reaches_the_backend_unchanged_once_and_in_order(backend):
    prompts = [PROMPT, "second «prompt» with\ttabs and \"quotes\"\n", "ünïcödé ✓"]
    for prompt in prompts:
        _complete(backend.provider, prompt)
    assert backend.sent_prompts() == prompts


def test_repeated_calls_return_equal_responses_for_the_same_reply(backend):
    first, second = _complete(backend.provider), _complete(backend.provider)
    assert first.text == second.text and first.model == second.model


def test_repr_carries_no_secret(backend):
    assert not any(secret in repr(backend.provider) for secret in SECRETS)


# ---- privacy attributes ----


def test_privacy_attributes_are_as_specified():
    assert (OllamaProvider.is_local, OllamaProvider.retains_data) == (True, False)
    assert (GroqProvider.is_local, GroqProvider.retains_data) == (False, None)
    assert (GeminiProvider.is_local, GeminiProvider.retains_data) == (False, None)


@pytest.mark.parametrize("host", ["http://localhost:11434", "http://127.0.0.1:11434", "http://127.8.9.1", "http://[::1]:11434"])
def test_ollama_accepts_loopback_hosts(host):
    assert OllamaProvider("m", host=host).is_local is True


@pytest.mark.parametrize("host", ["http://10.0.0.5:11434", "https://ollama.example.com", "http://0.0.0.0:11434", "not a url"])
def test_ollama_refuses_a_non_loopback_host_because_is_local_would_be_false(host):
    with pytest.raises(ProviderError) as excinfo:
        OllamaProvider("m", host=host)
    assert excinfo.value.kind is ProviderErrorKind.CONFIGURATION


# ---- HTTP requests ----


def test_http_providers_populate_latency(http_name, monkeypatch):
    response = _complete(_backend(http_name, monkeypatch).provider)
    assert isinstance(response.latency_ms, float) and response.latency_ms >= 0


def test_token_counts_and_model_come_from_the_response(http_name, monkeypatch):
    response = _complete(_backend(http_name, monkeypatch).provider)
    assert (response.prompt_tokens, response.completion_tokens) == (412, 96)
    assert response.model == {"ollama": "llama3.1:8b", "groq": "llama-3.3-70b-versatile", "gemini": "gemini-2.5-flash"}[http_name]


def test_configured_timeout_reaches_the_socket(http_name, monkeypatch):
    backend = _backend(http_name, monkeypatch, timeout_s=7.5)
    _complete(backend.provider)
    assert backend.replay.timeouts == [7.5]


def test_max_tokens_and_temperature_are_sent(monkeypatch):
    sent = {}
    for name in HTTP_PROVIDERS:
        backend = _backend(name, monkeypatch)
        backend.provider.complete(PROMPT, max_tokens=321, temperature=0.7)
        sent[name] = backend.replay.sent_json()[0]
    assert sent["ollama"]["options"] == {"num_predict": 321, "temperature": 0.7}
    assert sent["ollama"]["stream"] is False
    assert (sent["groq"]["max_completion_tokens"], sent["groq"]["temperature"], sent["groq"]["stream"]) == (321, 0.7, False)
    assert sent["gemini"]["generationConfig"] == {"maxOutputTokens": 321, "temperature": 0.7}


def test_credentials_travel_only_in_the_authorization_header(monkeypatch):
    for name, secret in (("groq", GROQ_KEY), ("gemini", GEMINI_TOKEN)):
        backend = _backend(name, monkeypatch)
        _complete(backend.provider)
        request = backend.replay.requests[0]
        assert request.get_header("Authorization") == f"Bearer {secret}"
        assert secret not in request.full_url and secret.encode() not in request.data
    backend = _backend("ollama", monkeypatch)
    _complete(backend.provider)
    assert backend.replay.requests[0].get_header("Authorization") is None


def test_gemini_global_location_uses_the_unprefixed_host():
    provider = GeminiProvider("gemini-2.5-pro", project="p", location="global", token_provider=lambda: "t")
    assert provider._url == (
        "https://aiplatform.googleapis.com/v1/projects/p/locations/global/publishers/google/models/gemini-2.5-pro:generateContent"
    )


def test_gemini_excludes_thought_parts_and_joins_text_parts(monkeypatch):
    body = {"candidates": [{"content": {"parts": [{"text": "thinking...", "thought": True}, {"text": "an"}, {"text": "swer"}]}}]}
    backend = _backend("gemini", monkeypatch, body=json.dumps(body).encode())
    assert _complete(backend.provider).text == "answer"


def test_gemini_blocked_prompt_is_an_invalid_response_naming_the_reason(monkeypatch):
    body = {"promptFeedback": {"blockReason": "SAFETY"}}
    backend = _backend("gemini", monkeypatch, body=json.dumps(body).encode())
    with pytest.raises(ProviderError, match=r"prompt blocked \(SAFETY\)") as excinfo:
        _complete(backend.provider)
    assert excinfo.value.kind is ProviderErrorKind.INVALID_RESPONSE


# ---- failures are typed ProviderErrors ----

_STATUS_KINDS = [
    (400, ProviderErrorKind.HTTP),
    (401, ProviderErrorKind.AUTH),
    (403, ProviderErrorKind.AUTH),
    (404, ProviderErrorKind.HTTP),
    (408, ProviderErrorKind.TIMEOUT),
    (429, ProviderErrorKind.RATE_LIMIT),
    (500, ProviderErrorKind.HTTP),
    (503, ProviderErrorKind.HTTP),
    (504, ProviderErrorKind.TIMEOUT),
]


def _assert_safe(exc: BaseException) -> None:
    """No secret, prompt or response body anywhere in the error, its repr, args, chain or formatted traceback."""
    rendered = [str(exc), repr(exc), *map(str, exc.args), "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))]
    for text in rendered:
        for secret in SECRETS:
            assert secret not in text
        assert "Explain finding SENT-001" not in text
        assert "Invalid API Key" not in text and "Permission denied for token" not in text
    # Nothing chained: the urllib exception underneath holds the response body.
    assert exc.__cause__ is None and (exc.__context__ is None or exc.__suppress_context__)


@pytest.mark.parametrize("status,kind", _STATUS_KINDS)
def test_http_status_maps_to_a_typed_error_naming_provider_and_status(http_name, monkeypatch, status, kind):
    backend = _backend(http_name, monkeypatch, cassette="error", status=status)
    with pytest.raises(ProviderError) as excinfo:
        _complete(backend.provider)
    error = excinfo.value
    assert (error.provider, error.kind, error.status) == (http_name, kind, status)
    assert http_name in str(error) and f"HTTP {status}" in str(error)
    _assert_safe(error)


def test_rate_limit_carries_retry_after(http_name, monkeypatch):
    backend = _backend(http_name, monkeypatch, cassette="error", status=429, headers={"Retry-After": "17"})
    with pytest.raises(ProviderError) as excinfo:
        _complete(backend.provider)
    assert excinfo.value.retry_after == 17.0


def test_rate_limit_with_an_http_date_retry_after_has_none(http_name, monkeypatch):
    backend = _backend(http_name, monkeypatch, cassette="error", status=429, headers={"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"})
    with pytest.raises(ProviderError) as excinfo:
        _complete(backend.provider)
    assert excinfo.value.retry_after is None


@pytest.mark.parametrize(
    "raised,kind",
    [
        (urllib.error.URLError(ConnectionRefusedError(10061, "refused")), ProviderErrorKind.TRANSPORT),
        (urllib.error.URLError(socket.gaierror(11001, "getaddrinfo failed")), ProviderErrorKind.TRANSPORT),
        (ConnectionResetError(10054, "reset"), ProviderErrorKind.TRANSPORT),
        (socket.timeout("timed out"), ProviderErrorKind.TIMEOUT),
        (urllib.error.URLError(socket.timeout("timed out")), ProviderErrorKind.TIMEOUT),
    ],
    ids=["refused", "dns", "reset", "socket-timeout", "urlerror-timeout"],
)
def test_transport_failures_are_typed(http_name, monkeypatch, raised, kind):
    backend = _backend(http_name, monkeypatch, raises=raised)
    with pytest.raises(ProviderError) as excinfo:
        _complete(backend.provider)
    assert (excinfo.value.provider, excinfo.value.kind, excinfo.value.status) == (http_name, kind, None)
    _assert_safe(excinfo.value)


def test_hard_timeout_bounds_a_server_that_never_answers(http_name, monkeypatch):
    release = threading.Event()
    backend = _backend(http_name, monkeypatch, timeout_s=0.3)

    def hang(request, timeout=None):
        release.wait(10)
        raise socket.timeout("released")

    monkeypatch.setattr(urllib.request, "urlopen", hang)
    started = time.monotonic()
    try:
        with pytest.raises(ProviderError) as excinfo:
            _complete(backend.provider)
    finally:
        release.set()
    assert time.monotonic() - started < 2.0
    assert excinfo.value.kind is ProviderErrorKind.TIMEOUT
    _assert_safe(excinfo.value)


def test_hard_timeout_bounds_a_slow_drip_body(http_name, monkeypatch):
    release = threading.Event()
    backend = _backend(http_name, monkeypatch, timeout_s=0.3)

    class Drip(_Response):
        def read(self, size=-1):
            # One byte per 50ms until the test ends, then EOF so the abandoned thread exits.
            return b"" if release.wait(0.05) else b" "

    monkeypatch.setattr(urllib.request, "urlopen", lambda request, timeout=None: Drip(b""))
    started = time.monotonic()
    try:
        with pytest.raises(ProviderError) as excinfo:
            _complete(backend.provider)
    finally:
        release.set()
    assert time.monotonic() - started < 2.0
    assert excinfo.value.kind is ProviderErrorKind.TIMEOUT


@pytest.mark.parametrize("body", [b"<html>502 Bad Gateway</html>", b"\xff\xfe", b"[]", b'{"unexpected": true}', b"null"])
def test_malformed_success_body_is_an_invalid_response(http_name, monkeypatch, body):
    backend = _backend(http_name, monkeypatch, body=body)
    with pytest.raises(ProviderError) as excinfo:
        _complete(backend.provider)
    assert excinfo.value.kind is ProviderErrorKind.INVALID_RESPONSE
    _assert_safe(excinfo.value)


def test_oversized_body_is_refused(http_name, monkeypatch):
    backend = _backend(http_name, monkeypatch, body=b" " * (9 * 1024 * 1024))
    with pytest.raises(ProviderError, match="too large") as excinfo:
        _complete(backend.provider)
    assert excinfo.value.kind is ProviderErrorKind.INVALID_RESPONSE


@pytest.mark.parametrize("timeout_s", [0, -1, float("inf"), float("nan"), True, "30"])
def test_invalid_timeouts_are_rejected(monkeypatch, timeout_s):
    monkeypatch.setenv("GROQ_API_KEY", GROQ_KEY)
    for build in (
        lambda: OllamaProvider("m", timeout_s=timeout_s),
        lambda: GroqProvider("m", timeout_s=timeout_s),
        lambda: GeminiProvider("m", project="p", timeout_s=timeout_s, token_provider=lambda: "t"),
    ):
        with pytest.raises(ValueError):
            build()


# ---- credentials come from the environment only ----


def test_no_provider_accepts_a_key_as_an_argument():
    for cls in (OllamaProvider, GroqProvider, GeminiProvider):
        params = set(inspect.signature(cls.__init__).parameters)
        assert not params & {"api_key", "key", "credentials", "credentials_path", "token"}, cls.__name__


@pytest.mark.parametrize("value", [None, "", "   "])
def test_groq_without_its_env_key_is_a_configuration_error(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
    else:
        monkeypatch.setenv("GROQ_API_KEY", value)
    with pytest.raises(ProviderError, match="GROQ_API_KEY is not set") as excinfo:
        GroqProvider("m")
    assert excinfo.value.kind is ProviderErrorKind.CONFIGURATION


def test_gemini_without_credentials_env_is_a_configuration_error(monkeypatch):
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    with pytest.raises(ProviderError, match="GOOGLE_APPLICATION_CREDENTIALS is not set") as excinfo:
        GeminiProvider("m", project="p")
    assert excinfo.value.kind is ProviderErrorKind.CONFIGURATION


def test_gemini_without_a_project_is_a_configuration_error(monkeypatch):
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    with pytest.raises(ProviderError, match="GOOGLE_CLOUD_PROJECT is not set"):
        GeminiProvider("m", token_provider=lambda: "t")


def test_gemini_reads_project_and_location_from_env(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "env-project")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "europe-west4")
    provider = GeminiProvider("m", token_provider=lambda: "t")
    assert (provider.project, provider.location) == ("env-project", "europe-west4")
    assert provider._url.startswith("https://europe-west4-aiplatform.googleapis.com/v1/projects/env-project/locations/europe-west4/")


def test_gemini_without_google_auth_installed_is_a_configuration_error(monkeypatch, tmp_path):
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(tmp_path / "sa.json"))
    for module in ("google", "google.auth", "google.auth.transport", "google.auth.transport.requests", "google.oauth2"):
        monkeypatch.setitem(sys.modules, module, None)
    provider = GeminiProvider("m", project="p")
    with pytest.raises(ProviderError, match="google-auth is not installed") as excinfo:
        _complete(provider)
    assert excinfo.value.kind is ProviderErrorKind.CONFIGURATION


def test_gemini_token_failure_is_an_auth_error_without_its_message(monkeypatch):
    def failing():
        raise RuntimeError(f"refresh failed for {GEMINI_TOKEN}: invalid_grant")

    provider = GeminiProvider("m", project="p", token_provider=failing)
    with pytest.raises(ProviderError) as excinfo:
        _complete(provider)
    assert excinfo.value.kind is ProviderErrorKind.AUTH
    assert "RuntimeError" in str(excinfo.value)
    _assert_safe(excinfo.value)


@pytest.mark.parametrize("token", ["", None])
def test_gemini_empty_token_is_an_auth_error(token):
    with pytest.raises(ProviderError) as excinfo:
        _complete(GeminiProvider("m", project="p", token_provider=lambda: token))
    assert excinfo.value.kind is ProviderErrorKind.AUTH


@pytest.mark.parametrize("build", [lambda: OllamaProvider(""), lambda: GroqProvider(""), lambda: GeminiProvider("", project="p", token_provider=lambda: "t")])
def test_empty_model_is_a_configuration_error(monkeypatch, build):
    monkeypatch.setenv("GROQ_API_KEY", GROQ_KEY)
    with pytest.raises(ProviderError) as excinfo:
        build()
    assert excinfo.value.kind is ProviderErrorKind.CONFIGURATION


# ---- no secret in any exception message or log record ----


def test_no_api_key_appears_in_any_exception_or_log_record(monkeypatch, caplog):
    # On the package logger itself: ai/config.py's get_settings() may already have raised its level.
    caplog.set_level(logging.DEBUG, logger="sentinelai")
    errors: list[BaseException] = []
    scenarios = [dict()]
    scenarios += [dict(cassette="error", status=status) for status, _ in _STATUS_KINDS]
    scenarios += [dict(raises=urllib.error.URLError(ConnectionRefusedError(10061, "refused"))), dict(body=b"not json")]
    for name in HTTP_PROVIDERS:
        for scenario in scenarios:
            backend = _backend(name, monkeypatch, **dict(scenario))
            try:
                _complete(backend.provider)
            except ProviderError as exc:
                errors.append(exc)
    assert len(errors) == len(HTTP_PROVIDERS) * (len(scenarios) - 1)
    for error in errors:
        _assert_safe(error)
    assert any(record.levelno == logging.DEBUG for record in caplog.records), "expected the success path to log"
    for record in caplog.records:
        rendered = record.getMessage() + (record.exc_text or "") + repr(record.args)
        for secret in SECRETS:
            assert secret not in rendered
    for secret in SECRETS:
        assert secret not in caplog.text
