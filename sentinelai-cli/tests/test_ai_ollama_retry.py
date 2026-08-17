"""
Tests for the Ollama transport retry (ai/ollama_http.py) and its effect
on both clients.

The policy under test is narrow on purpose: retry only failures a second
attempt could plausibly fix, and never spend the backoff on a
deterministic one. The negative cases matter as much as the positive
one - a retry on malformed model output would double the cost of the
most common failure mode while changing nothing about the outcome.

urllib.request.urlopen and time.sleep are both patched, so no network
call and no real delay occurs; the suite stays fast and offline.

CONVENTION - any test that simulates a *retryable* failure must patch
time.sleep, anywhere in this suite, not only in this file. Connection
errors and HTTP 500/502/503/504 now trigger a real 2-second backoff in
ai/ollama_http.py, so an unpatched test silently costs 2 seconds. This
was caught for real: adding the retry pushed
tests/test_ai_embedding_config.py::test_unreachable_server_says_so from
milliseconds to 2.01s and the whole suite from 1.59s to 3.34s. That test
now patches time.sleep for exactly this reason. Deterministic failures
(4xx, 501, malformed JSON, schema violations, configuration errors) are
not retried and need no patch.
"""
import json
import urllib.error
from unittest.mock import patch

import pytest

from sentinelai.ai.embeddings_ollama import make_ollama_embed_fn
from sentinelai.ai.llm_ollama import OllamaProvider
from sentinelai.ai.ollama_http import MAX_ATTEMPTS, RETRY_BACKOFF_SECONDS, read_with_retry
from sentinelai.core.errors import AIEnrichmentError

HOST = "http://localhost:11434"

_GENERATE_BODY = json.dumps({"response": json.dumps({"title": "t", "explanation": "e", "remediation": "r"})})
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


def _http_error(code: int, reason: str = "Error") -> urllib.error.HTTPError:
    return urllib.error.HTTPError(f"{HOST}/api/embed", code, reason, {}, None)


# --- policy constants ------------------------------------------------------------------------------------------------


def test_policy_is_two_attempts_with_two_second_backoff():
    assert MAX_ATTEMPTS == 2
    assert RETRY_BACKOFF_SECONDS == 2.0


# --- 1. transient failure then success --------------------------------------------------------------------------------


def test_transient_failure_then_success():
    responses = [urllib.error.URLError("connection refused"), _FakeResponse(_EMBED_BODY)]

    with patch("urllib.request.urlopen", side_effect=responses) as urlopen, patch("time.sleep"):
        body = read_with_retry(_request(), 120, HOST)

    assert json.loads(body.decode("utf-8")) == {"embeddings": [[0.1, 0.2]]}
    assert urlopen.call_count == 2


def test_retry_is_logged_once_without_leaking_payload(caplog):
    import logging

    responses = [urllib.error.URLError("connection refused"), _FakeResponse(_EMBED_BODY)]

    with patch("urllib.request.urlopen", side_effect=responses), patch("time.sleep"):
        with caplog.at_level(logging.WARNING, logger="sentinelai"):
            read_with_retry(_request(), 120, HOST)

    assert len(caplog.records) == 1
    assert "attempt 1 of 2" in caplog.text
    assert HOST in caplog.text
    # The request payload must never appear in a log line.
    assert "embeddings" not in caplog.text


# --- 2. both attempts fail -> original error preserved ------------------------------------------------------------------


def test_both_attempts_fail_raises_original_error():
    original = urllib.error.URLError("connection refused")

    with patch("urllib.request.urlopen", side_effect=[original, original]) as urlopen, patch("time.sleep"):
        with pytest.raises(urllib.error.URLError) as excinfo:
            read_with_retry(_request(), 120, HOST)

    assert excinfo.value is original
    assert urlopen.call_count == MAX_ATTEMPTS


def test_embeddings_still_raise_the_actionable_message_after_retries():
    # The retry must not swallow or reword ai/embeddings_ollama.py's guidance.
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("refused")), patch("time.sleep"):
        with pytest.raises(AIEnrichmentError) as excinfo:
            make_ollama_embed_fn(host=HOST, model="nomic-embed-text")(["text"])

    assert "could not reach the Ollama server" in str(excinfo.value)
    assert "ollama serve" in str(excinfo.value)


# --- 3. deterministic failures are NOT retried ---------------------------------------------------------------------------


def test_malformed_model_output_is_not_retried():
    # Prose instead of JSON. The transport succeeded, so generate() returns the text
    # as-is with a single HTTP call; the schema failure surfaces downstream in
    # explain(), which performs no further request. Asserted end-to-end through
    # explain() because that - not generate() - is where model output is parsed.
    from sentinelai.ai.agents.explainer import explain

    body = json.dumps({"response": "Sure! Here is my analysis, in prose."})
    provider = OllamaProvider(host=HOST, model="llama3.1:8b")

    with patch("urllib.request.urlopen", return_value=_FakeResponse(body)) as urlopen, patch("time.sleep") as sleep:
        with pytest.raises(json.JSONDecodeError):
            explain("prompt", provider.generate)

    assert urlopen.call_count == 1  # no retry spent on a deterministic failure
    sleep.assert_not_called()


def test_schema_violation_in_model_output_is_not_retried():
    # Well-formed JSON that violates LLMResponse's schema - equally deterministic.
    from pydantic import ValidationError

    from sentinelai.ai.agents.explainer import explain

    body = json.dumps({"response": json.dumps({"title": "only a title"})})
    provider = OllamaProvider(host=HOST, model="llama3.1:8b")

    with patch("urllib.request.urlopen", return_value=_FakeResponse(body)) as urlopen, patch("time.sleep") as sleep:
        with pytest.raises(ValidationError):
            explain("prompt", provider.generate)

    assert urlopen.call_count == 1
    sleep.assert_not_called()


def test_non_json_transport_body_is_not_retried():
    with patch("urllib.request.urlopen", return_value=_FakeResponse("<html>proxy</html>")) as urlopen, patch(
        "time.sleep"
    ) as sleep:
        with pytest.raises(AIEnrichmentError) as excinfo:
            make_ollama_embed_fn(host=HOST, model="nomic-embed-text")(["text"])

    assert "non-JSON response" in str(excinfo.value)
    assert urlopen.call_count == 1
    sleep.assert_not_called()


def test_http_501_is_not_retried():
    # The most common misconfiguration: deterministic, so retrying only delays the fix.
    with patch("urllib.request.urlopen", side_effect=_http_error(501, "Not Implemented")) as urlopen, patch(
        "time.sleep"
    ) as sleep:
        with pytest.raises(AIEnrichmentError) as excinfo:
            make_ollama_embed_fn(host=HOST, model="llama3.1:8b")(["text"])

    assert "does not support embeddings" in str(excinfo.value)
    assert urlopen.call_count == 1
    sleep.assert_not_called()


def test_http_404_is_not_retried():
    with patch("urllib.request.urlopen", side_effect=_http_error(404, "Not Found")) as urlopen, patch(
        "time.sleep"
    ) as sleep:
        with pytest.raises(AIEnrichmentError):
            make_ollama_embed_fn(host=HOST, model="nomic-embed-text")(["text"])

    assert urlopen.call_count == 1
    sleep.assert_not_called()


def test_configuration_errors_are_not_retried():
    with patch("urllib.request.urlopen") as urlopen, patch("time.sleep") as sleep:
        with pytest.raises(ValueError):
            make_ollama_embed_fn(host=HOST, model="")

    urlopen.assert_not_called()
    sleep.assert_not_called()


# --- transient 5xx statuses ARE retried ---------------------------------------------------------------------------------


@pytest.mark.parametrize("code", [500, 502, 503, 504])
def test_transient_server_statuses_are_retried(code):
    with patch("urllib.request.urlopen", side_effect=[_http_error(code), _FakeResponse(_EMBED_BODY)]) as urlopen, patch(
        "time.sleep"
    ):
        result = make_ollama_embed_fn(host=HOST, model="nomic-embed-text")(["text"])

    assert result == [[0.1, 0.2]]
    assert urlopen.call_count == 2


# --- 4. sleep is called exactly once, with 2 seconds, only when retrying ---------------------------------------------------


def test_sleep_called_once_with_two_seconds_on_retryable_failure():
    with patch("urllib.request.urlopen", side_effect=[urllib.error.URLError("refused"), _FakeResponse(_EMBED_BODY)]), patch(
        "time.sleep"
    ) as sleep:
        read_with_retry(_request(), 120, HOST)

    sleep.assert_called_once_with(2.0)


def test_no_sleep_on_success():
    with patch("urllib.request.urlopen", return_value=_FakeResponse(_EMBED_BODY)), patch("time.sleep") as sleep:
        read_with_retry(_request(), 120, HOST)

    sleep.assert_not_called()


def test_no_sleep_after_the_final_attempt():
    # Two failures means one retry, so sleep happens once - never after the last attempt.
    with patch("urllib.request.urlopen", side_effect=[urllib.error.URLError("a"), urllib.error.URLError("b")]), patch(
        "time.sleep"
    ) as sleep:
        with pytest.raises(urllib.error.URLError):
            read_with_retry(_request(), 120, HOST)

    assert sleep.call_count == 1


# --- 5. healthy path is unchanged: exactly one HTTP call --------------------------------------------------------------------


def test_healthy_generate_makes_exactly_one_http_call():
    with patch("urllib.request.urlopen", return_value=_FakeResponse(_GENERATE_BODY)) as urlopen, patch(
        "time.sleep"
    ) as sleep:
        out = OllamaProvider(host=HOST, model="llama3.1:8b").generate("prompt")

    assert urlopen.call_count == 1
    sleep.assert_not_called()
    assert json.loads(out)["title"] == "t"


def test_healthy_embed_makes_exactly_one_http_call():
    with patch("urllib.request.urlopen", return_value=_FakeResponse(_EMBED_BODY)) as urlopen, patch(
        "time.sleep"
    ) as sleep:
        result = make_ollama_embed_fn(host=HOST, model="nomic-embed-text")(["text"])

    assert urlopen.call_count == 1
    sleep.assert_not_called()
    assert result == [[0.1, 0.2]]


def test_empty_embed_input_still_makes_no_http_call():
    with patch("urllib.request.urlopen") as urlopen, patch("time.sleep") as sleep:
        assert make_ollama_embed_fn(host=HOST, model="nomic-embed-text")([]) == []

    urlopen.assert_not_called()
    sleep.assert_not_called()


def test_per_attempt_timeout_is_preserved():
    # 120s applies to each attempt independently, not divided across them.
    with patch("urllib.request.urlopen", return_value=_FakeResponse(_EMBED_BODY)) as urlopen, patch("time.sleep"):
        make_ollama_embed_fn(host=HOST, model="nomic-embed-text")(["text"])

    assert urlopen.call_args.kwargs["timeout"] == 120
