"""
Configurable Ollama transport settings (request_timeout_seconds, max_attempts).

Both values were previously hardcoded - a 120-second timeout duplicated
in ai/llm_ollama.py and ai/embeddings_ollama.py, and a 2-attempt policy
in ai/ollama_http.py - so a user on a slow machine, or one running a
larger model, had no way to raise them short of editing the package.

The dominant requirement here is that nothing moved. The defaults must
equal the old constants exactly, because the AI latency figures in
PAPER_RESULTS.md were measured under them; an unconfigured run must be
byte-for-byte what it was. Most of this file tests that, not the feature.

What is deliberately NOT configurable: which failures count as
retryable. That classification encodes which errors a retry can actually
fix (a connection reset, yes; a 404 for an unpulled model, never), so it
is a correctness property rather than a preference.

CONVENTION (see tests/test_ai_ollama_retry.py): any test that simulates a
retryable failure must patch time.sleep, or it spends the real backoff.
"""
import json
import urllib.error
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from sentinelai.ai.config import AISettings
from sentinelai.ai.embeddings_ollama import make_ollama_embed_fn
from sentinelai.ai.llm_ollama import _REQUEST_TIMEOUT_SECONDS, OllamaProvider
from sentinelai.ai.ollama_http import MAX_ATTEMPTS, RETRY_BACKOFF_SECONDS, read_with_retry

HOST = "http://localhost:11434"
MODEL = "llama3.1:8b"

_GENERATE_BODY = json.dumps({"response": '{"ok": true}'})
_EMBED_BODY = json.dumps({"embeddings": [[0.1, 0.2]]})


class _FakeResponse:
    def __init__(self, body: str):
        self._body = body.encode("utf-8")

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _request():
    import urllib.request

    return urllib.request.Request(f"{HOST}/api/embed", data=b"{}", method="POST")


# --- defaults match the previously hardcoded values -----------------------------------------------------------------


def test_default_timeout_is_the_old_hardcoded_value():
    assert AISettings().request_timeout_seconds == 120.0
    assert _REQUEST_TIMEOUT_SECONDS == 120


def test_default_attempts_is_the_old_hardcoded_policy():
    assert AISettings().max_attempts == 2
    assert MAX_ATTEMPTS == 2


def test_backoff_is_unchanged_and_still_not_configurable():
    """Only the attempt count became tunable; the 2-second backoff did not."""
    assert RETRY_BACKOFF_SECONDS == 2.0
    assert "backoff" not in AISettings.model_fields
    assert "retry_backoff_seconds" not in AISettings.model_fields


def test_exported_constants_remain_importable_for_existing_callers():
    """tests/test_ai_ollama_retry.py imports both by name; they must keep working."""
    assert isinstance(MAX_ATTEMPTS, int)
    assert isinstance(RETRY_BACKOFF_SECONDS, float)


# --- environment overrides ---------------------------------------------------------------------------------------------


def _settings_from(monkeypatch, **env):
    import sentinelai.ai.config as config

    for key, value in env.items():
        monkeypatch.setenv(f"SENTINELAI_AI_{key.upper()}", value)
    monkeypatch.setattr(config, "_settings", None)
    return config._settings_from_env()


def test_timeout_is_settable_from_the_environment(monkeypatch):
    settings = _settings_from(monkeypatch, request_timeout_seconds="300")

    assert settings.request_timeout_seconds == 300.0


def test_attempts_are_settable_from_the_environment(monkeypatch):
    settings = _settings_from(monkeypatch, max_attempts="5")

    assert settings.max_attempts == 5


def test_unset_environment_yields_the_defaults(monkeypatch):
    import sentinelai.ai.config as config

    monkeypatch.delenv("SENTINELAI_AI_REQUEST_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("SENTINELAI_AI_MAX_ATTEMPTS", raising=False)
    monkeypatch.setattr(config, "_settings", None)

    settings = config._settings_from_env()

    assert (settings.request_timeout_seconds, settings.max_attempts) == (120.0, 2)


# --- validation ------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("value", [0, -1, -0.5])
def test_non_positive_timeout_is_rejected(value):
    with pytest.raises(ValidationError):
        AISettings(request_timeout_seconds=value)


@pytest.mark.parametrize("value", [0, -1])
def test_attempts_below_one_are_rejected(value):
    with pytest.raises(ValidationError):
        AISettings(max_attempts=value)


def test_one_attempt_is_valid_and_means_no_retrying():
    assert AISettings(max_attempts=1).max_attempts == 1


@pytest.mark.parametrize(
    "kwargs",
    [{"timeout": 0}, {"timeout": -1}, {"max_attempts": 0}, {"max_attempts": -3}],
)
def test_clients_reject_invalid_transport_values(kwargs):
    with pytest.raises(ValueError):
        OllamaProvider(host=HOST, model=MODEL, **kwargs)
    with pytest.raises(ValueError):
        make_ollama_embed_fn(host=HOST, model="nomic-embed-text", **kwargs)


# --- the values actually reach urllib -------------------------------------------------------------------------------


def test_generation_passes_the_configured_timeout_to_urlopen():
    with patch("urllib.request.urlopen", return_value=_FakeResponse(_GENERATE_BODY)) as urlopen:
        OllamaProvider(host=HOST, model=MODEL, timeout=7.5).generate("prompt")

    assert urlopen.call_args.kwargs["timeout"] == 7.5


def test_embedding_passes_the_configured_timeout_to_urlopen():
    with patch("urllib.request.urlopen", return_value=_FakeResponse(_EMBED_BODY)) as urlopen:
        make_ollama_embed_fn(host=HOST, model="nomic-embed-text", timeout=9.0)(["text"])

    assert urlopen.call_args.kwargs["timeout"] == 9.0


def test_default_construction_still_uses_the_120_second_timeout():
    with patch("urllib.request.urlopen", return_value=_FakeResponse(_GENERATE_BODY)) as urlopen:
        OllamaProvider(host=HOST, model=MODEL).generate("prompt")

    assert urlopen.call_args.kwargs["timeout"] == 120


def test_configured_attempts_change_how_many_times_a_transient_failure_is_retried():
    refused = urllib.error.URLError("refused")

    with patch("urllib.request.urlopen", side_effect=[refused] * 4) as urlopen, patch("time.sleep") as sleep:
        with pytest.raises(Exception):
            OllamaProvider(host=HOST, model=MODEL, max_attempts=4).generate("prompt")

    assert urlopen.call_count == 4
    assert sleep.call_count == 3


def test_one_attempt_disables_retrying_entirely():
    refused = urllib.error.URLError("refused")

    with patch("urllib.request.urlopen", side_effect=[refused]) as urlopen, patch("time.sleep") as sleep:
        with pytest.raises(Exception):
            OllamaProvider(host=HOST, model=MODEL, max_attempts=1).generate("prompt")

    assert urlopen.call_count == 1
    sleep.assert_not_called()


def test_read_with_retry_keeps_its_three_argument_form():
    """Existing call sites and tests pass three positional arguments."""
    refused = urllib.error.URLError("refused")

    with patch("urllib.request.urlopen", side_effect=[refused] * MAX_ATTEMPTS) as urlopen, patch("time.sleep"):
        with pytest.raises(urllib.error.URLError):
            read_with_retry(_request(), 120, HOST)

    assert urlopen.call_count == MAX_ATTEMPTS


# --- read_with_retry validates max_attempts itself -------------------------------------------------------------------------
#
# The transport used to clamp with max(1, int(max_attempts)), which turned every
# invalid value into a working call: 0 and -1 became one attempt, 2.9 became 2,
# and True - an int subclass - became 1. The clients already reject values below
# 1, so this is the second line of defence rather than the first; what it adds is
# catching the wrong *type*, which a range check cannot.


@pytest.mark.parametrize("attempts", [1, 2, 3])
def test_valid_attempt_counts_are_accepted(attempts):
    with patch("urllib.request.urlopen", return_value=_FakeResponse(_EMBED_BODY)) as urlopen:
        read_with_retry(_request(), 120, HOST, attempts)

    assert urlopen.call_count == 1  # a success never needs a second attempt


def test_one_attempt_never_retries():
    refused = urllib.error.URLError("refused")

    with patch("urllib.request.urlopen", side_effect=[refused]) as urlopen, patch("time.sleep") as sleep:
        with pytest.raises(urllib.error.URLError):
            read_with_retry(_request(), 120, HOST, 1)

    assert urlopen.call_count == 1
    sleep.assert_not_called()


def test_two_attempts_preserves_the_default_behaviour():
    """Explicitly passing the default must equal not passing it at all."""
    refused = urllib.error.URLError("refused")

    with patch("urllib.request.urlopen", side_effect=[refused] * 2) as urlopen, patch("time.sleep") as sleep:
        with pytest.raises(urllib.error.URLError):
            read_with_retry(_request(), 120, HOST, 2)

    assert urlopen.call_count == 2
    assert sleep.call_count == 1


@pytest.mark.parametrize("attempts", [0, -1, -99])
def test_attempt_counts_below_one_raise_value_error(attempts):
    """Previously clamped to 1 and silently succeeded."""
    with patch("urllib.request.urlopen") as urlopen:
        with pytest.raises(ValueError, match="at least 1"):
            read_with_retry(_request(), 120, HOST, attempts)

    urlopen.assert_not_called()  # rejected before any network call


@pytest.mark.parametrize(
    "attempts",
    [2.5, 2.0, "2", None, [2], True, False],
    ids=["float", "whole-float", "str", "none", "list", "true", "false"],
)
def test_non_integer_attempt_counts_raise_type_error(attempts):
    """Integer-only contract, and bool is deliberately included.

    isinstance(True, int) is True in Python, so a plain int check would accept
    True and silently mean "one attempt" - exactly the kind of coercion this
    validation exists to stop. `2.0` is rejected too: it is unambiguous to a
    human but accepting it would reintroduce int() coercion by the back door.
    """
    with patch("urllib.request.urlopen") as urlopen:
        with pytest.raises(TypeError, match="must be an int"):
            read_with_retry(_request(), 120, HOST, attempts)

    urlopen.assert_not_called()


def test_the_error_names_the_offending_type_and_value():
    with pytest.raises(TypeError) as excinfo:
        read_with_retry(_request(), 120, HOST, "2")

    assert "str" in str(excinfo.value)
    assert "'2'" in str(excinfo.value)


def test_validation_does_not_change_the_default_call():
    """The three-argument form still uses MAX_ATTEMPTS and still works."""
    with patch("urllib.request.urlopen", return_value=_FakeResponse(_EMBED_BODY)) as urlopen:
        body = read_with_retry(_request(), 120, HOST)

    assert json.loads(body.decode())["embeddings"] == [[0.1, 0.2]]
    assert urlopen.call_count == 1


def test_retry_classification_is_unaffected_by_validation():
    """404 still not retried, 500 still retried, at an explicit attempt count."""
    not_found = urllib.error.HTTPError(f"{HOST}/api/embed", 404, "Not Found", {}, None)
    server_error = urllib.error.HTTPError(f"{HOST}/api/embed", 500, "Server Error", {}, None)

    with patch("urllib.request.urlopen", side_effect=[not_found] * 3) as urlopen, patch("time.sleep"):
        with pytest.raises(urllib.error.HTTPError):
            read_with_retry(_request(), 120, HOST, 3)
    assert urlopen.call_count == 1

    with patch(
        "urllib.request.urlopen", side_effect=[server_error, _FakeResponse(_EMBED_BODY)]
    ) as urlopen, patch("time.sleep"):
        read_with_retry(_request(), 120, HOST, 3)
    assert urlopen.call_count == 2


# --- the factory is the only place that reads configuration ------------------------------------------------------------


def test_factory_passes_settings_through_to_the_generation_client(monkeypatch):
    import sentinelai.ai.config as config
    from sentinelai.ai.factory import create_llm_generate_fn

    monkeypatch.setenv("SENTINELAI_AI_LLM_MODEL", MODEL)
    monkeypatch.setenv("SENTINELAI_AI_REQUEST_TIMEOUT_SECONDS", "42")
    monkeypatch.setenv("SENTINELAI_AI_MAX_ATTEMPTS", "1")
    monkeypatch.setattr(config, "_settings", None)

    generate = create_llm_generate_fn()
    with patch("urllib.request.urlopen", return_value=_FakeResponse(_GENERATE_BODY)) as urlopen:
        generate("prompt")

    assert urlopen.call_args.kwargs["timeout"] == 42.0
    monkeypatch.setattr(config, "_settings", None)


def test_factory_passes_settings_through_to_the_embedding_client(monkeypatch):
    import sentinelai.ai.config as config
    from sentinelai.ai.factory import create_embedding_fn

    monkeypatch.setenv("SENTINELAI_AI_EMBEDDING_MODEL", "nomic-embed-text")
    monkeypatch.setenv("SENTINELAI_AI_REQUEST_TIMEOUT_SECONDS", "17")
    monkeypatch.setattr(config, "_settings", None)

    embed = create_embedding_fn()
    with patch("urllib.request.urlopen", return_value=_FakeResponse(_EMBED_BODY)) as urlopen:
        embed(["text"])

    assert urlopen.call_args.kwargs["timeout"] == 17.0
    monkeypatch.setattr(config, "_settings", None)


# --- retry CLASSIFICATION is not configurable ----------------------------------------------------------------------------


def test_raising_attempts_does_not_make_404_retryable():
    """A model that was never pulled will not appear between two attempts."""
    error = urllib.error.HTTPError(f"{HOST}/api/generate", 404, "Not Found", {}, None)

    with patch("urllib.request.urlopen", side_effect=[error] * 9) as urlopen, patch("time.sleep") as sleep:
        with pytest.raises(Exception):
            OllamaProvider(host=HOST, model=MODEL, max_attempts=9).generate("prompt")

    assert urlopen.call_count == 1
    sleep.assert_not_called()


def test_raising_attempts_does_not_make_501_retryable():
    error = urllib.error.HTTPError(f"{HOST}/api/embed", 501, "Not Implemented", {}, None)

    with patch("urllib.request.urlopen", side_effect=[error] * 9) as urlopen, patch("time.sleep") as sleep:
        with pytest.raises(Exception):
            make_ollama_embed_fn(host=HOST, model=MODEL, max_attempts=9)(["text"])

    assert urlopen.call_count == 1
    sleep.assert_not_called()


def test_500_remains_retryable_at_the_configured_attempt_count():
    error = urllib.error.HTTPError(f"{HOST}/api/generate", 500, "Server Error", {}, None)

    with patch(
        "urllib.request.urlopen", side_effect=[error, error, _FakeResponse(_GENERATE_BODY)]
    ) as urlopen, patch("time.sleep") as sleep:
        result = OllamaProvider(host=HOST, model=MODEL, max_attempts=3).generate("prompt")

    assert result == '{"ok": true}'
    assert urlopen.call_count == 3
    assert sleep.call_count == 2
