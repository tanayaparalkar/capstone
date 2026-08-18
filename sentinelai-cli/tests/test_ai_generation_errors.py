"""
Generation-side error translation (ai/llm_ollama.py).

Before this, /api/generate failures propagated as raw urllib exceptions
while /api/embed failures were translated into actionable
AIEnrichmentError messages. The asymmetry was measured against a real
Ollama server, not theorised: pointing SENTINELAI_AI_LLM_MODEL at a model
that had not been pulled produced

    AI enrichment failed: all 13 finding(s) failed AI enrichment; first
    failure on 'bandit-0': HTTP Error 404: Not Found

once per finding - naming neither the model nor the fix - while Ollama's
own response body had said `{"error": "model 'llama3.1' not found"}`.
urllib raises HTTPError before anything reads that body, so the one
sentence that identified the problem was being discarded.

These tests pin the translation and, just as importantly, pin what it
must NOT change: the retry classification (500 retryable, 404/501 not),
the number of HTTP calls, and the success path.

CONVENTION (see tests/test_ai_ollama_retry.py): any test that simulates a
retryable failure must patch time.sleep, or it spends the real 2-second
backoff.
"""
import json
import urllib.error
from unittest.mock import patch

import pytest

from sentinelai.ai.llm_ollama import OllamaProvider
from sentinelai.ai.ollama_http import MAX_ATTEMPTS
from sentinelai.core.errors import AIEnrichmentError

HOST = "http://localhost:11434"
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


def _http_error(code: int, reason: str = "Error", body: bytes = None) -> urllib.error.HTTPError:
    """An HTTPError carrying a readable body, the way a real server response does."""
    import io

    fp = io.BytesIO(body) if body is not None else None
    return urllib.error.HTTPError(f"{HOST}/api/generate", code, reason, {}, fp)


def _generate(side_effect):
    with patch("urllib.request.urlopen", side_effect=side_effect) as urlopen, patch("time.sleep") as sleep:
        with pytest.raises(AIEnrichmentError) as excinfo:
            OllamaProvider(host=HOST, model=MODEL).generate("prompt")
    return excinfo.value, urlopen, sleep


# --- 404: the measured failure ------------------------------------------------------------------------------------------


def test_404_names_the_model_and_the_exact_fix():
    error, _, _ = _generate([_http_error(404, "Not Found")])
    message = str(error)

    assert MODEL in message
    assert f"ollama pull {MODEL}" in message
    assert HOST in message
    assert "SENTINELAI_AI_LLM_MODEL" in message


def test_404_surfaces_ollamas_own_explanation():
    """The body Ollama actually returns - previously discarded by urllib."""
    body = json.dumps({"error": "model 'llama3.1:8b' not found"}).encode("utf-8")

    error, _, _ = _generate([_http_error(404, "Not Found", body=body)])

    assert "model 'llama3.1:8b' not found" in str(error)
    assert f"ollama pull {MODEL}" in str(error)  # the fix is still there, not replaced by the raw string


# --- unreachable server -------------------------------------------------------------------------------------------------


def test_unreachable_server_matches_the_embedding_path_quality():
    """Same guidance the /api/embed path has always given: host, `ollama serve`, host variable."""
    refused = urllib.error.URLError("[Errno 61] Connection refused")

    error, _, _ = _generate([refused] * MAX_ATTEMPTS)
    message = str(error)

    assert "could not reach the Ollama server" in message
    assert HOST in message
    assert "ollama serve" in message
    assert "SENTINELAI_AI_LLM_HOST" in message


def test_os_error_is_also_translated():
    error, _, _ = _generate([OSError("socket timeout")] * MAX_ATTEMPTS)

    assert "could not reach the Ollama server" in str(error)


# --- other HTTP statuses ------------------------------------------------------------------------------------------------


def test_other_status_retains_status_and_reason():
    error, _, _ = _generate([_http_error(403, "Forbidden")])
    message = str(error)

    assert "403" in message
    assert "Forbidden" in message
    assert MODEL in message


def test_error_body_detail_is_appended_for_other_statuses():
    body = json.dumps({"error": 'invalid format: expected "json" or a valid JSON Schema object'}).encode()

    error, _, sleep = _generate([_http_error(500, "Internal Server Error", body=body)] * MAX_ATTEMPTS)

    assert "invalid format" in str(error)
    assert "500" in str(error)


# --- reading the error body must never raise a second exception -----------------------------------------------------------


@pytest.mark.parametrize(
    "label,body",
    [
        ("no body at all", None),
        ("empty body", b""),
        ("not JSON", b"<html>502 Bad Gateway</html>"),
        ("JSON but not an object", b'["nope"]'),
        ("object without an error key", b'{"detail": "something"}'),
        ("error key that is not a string", b'{"error": {"nested": true}}'),
        ("error key that is blank", b'{"error": "   "}'),
        ("invalid utf-8", b'{"error": "\xff\xfe bad"}'),
    ],
)
def test_unreadable_error_bodies_fall_back_to_the_status_line(label, body):
    """Surfacing a message must never be able to fail on top of the original failure."""
    error, _, _ = _generate([_http_error(404, "Not Found", body=body)])

    assert f"ollama pull {MODEL}" in str(error), f"fallback failed for: {label}"


def test_an_error_body_that_raises_on_read_is_tolerated():
    class _Exploding(urllib.error.HTTPError):
        def read(self, *args):
            raise OSError("stream already consumed")

    exploding = _Exploding(f"{HOST}/api/generate", 404, "Not Found", {}, None)

    error, _, _ = _generate([exploding])

    assert f"ollama pull {MODEL}" in str(error)


def test_an_overlong_error_body_is_capped():
    """A server returning an unbounded body must not paste all of it into the message."""
    from sentinelai.ai.llm_ollama import _MAX_ERROR_BODY_BYTES

    huge = json.dumps({"error": "x" * (_MAX_ERROR_BODY_BYTES * 4)}).encode("utf-8")

    error, _, _ = _generate([_http_error(404, "Not Found", body=huge)])

    # Truncated mid-JSON, so it does not parse and the status-line fallback applies.
    assert len(str(error)) < _MAX_ERROR_BODY_BYTES * 2
    assert f"ollama pull {MODEL}" in str(error)


# --- malformed successful responses ---------------------------------------------------------------------------------------


def test_non_json_transport_body_is_translated():
    error, _, sleep = _generate([_FakeResponse("<html>proxy</html>")])

    assert "non-JSON response" in str(error)
    sleep.assert_not_called()  # deterministic: not retried


def test_response_without_the_response_field_is_translated():
    error, _, _ = _generate([_FakeResponse(json.dumps({"model": MODEL, "done": True}))])

    assert "no 'response' field" in str(error)
    assert MODEL in str(error)


# --- retry classification is UNCHANGED ------------------------------------------------------------------------------------


def test_404_is_still_not_retried():
    _, urlopen, sleep = _generate([_http_error(404, "Not Found")])

    assert urlopen.call_count == 1
    sleep.assert_not_called()


def test_501_is_still_not_retried():
    _, urlopen, sleep = _generate([_http_error(501, "Not Implemented")])

    assert urlopen.call_count == 1
    sleep.assert_not_called()


def test_500_is_still_retried():
    """Translation must not change what counts as transient."""
    with patch(
        "urllib.request.urlopen",
        side_effect=[_http_error(500, "Internal Server Error"), _FakeResponse(_GOOD_BODY)],
    ) as urlopen, patch("time.sleep") as sleep:
        result = OllamaProvider(host=HOST, model=MODEL).generate("prompt")

    assert result == '{"ok": true}'
    assert urlopen.call_count == 2
    sleep.assert_called_once()


def test_connection_failure_is_still_retried_before_translating():
    _, urlopen, sleep = _generate([urllib.error.URLError("refused")] * MAX_ATTEMPTS)

    assert urlopen.call_count == MAX_ATTEMPTS
    assert sleep.call_count == MAX_ATTEMPTS - 1


# --- the success path is untouched ------------------------------------------------------------------------------------------


def test_successful_generation_is_unchanged():
    with patch("urllib.request.urlopen", return_value=_FakeResponse(_GOOD_BODY)) as urlopen:
        result = OllamaProvider(host=HOST, model=MODEL).generate("prompt")

    assert result == '{"ok": true}'
    assert urlopen.call_count == 1


def test_successful_generation_still_strips_markdown_fences():
    fenced = json.dumps({"response": '```json\n{"ok": true}\n```'})

    with patch("urllib.request.urlopen", return_value=_FakeResponse(fenced)):
        result = OllamaProvider(host=HOST, model=MODEL).generate("prompt")

    assert result == '{"ok": true}'


def test_the_translated_error_is_a_sentinelai_error():
    """So main.py's existing handler reports it as PROVIDER_ERROR, unchanged."""
    from sentinelai.core.errors import SentinelAIError

    error, _, _ = _generate([_http_error(404, "Not Found")])

    assert isinstance(error, SentinelAIError)


def test_no_payload_is_leaked_into_the_message():
    """Prompts and generated content must never appear in an error message."""
    secret_prompt = "SUPER-SECRET-SOURCE-SNIPPET-abc123"

    with patch("urllib.request.urlopen", side_effect=_http_error(404, "Not Found")), patch("time.sleep"):
        with pytest.raises(AIEnrichmentError) as excinfo:
            OllamaProvider(host=HOST, model=MODEL).generate(secret_prompt)

    assert secret_prompt not in str(excinfo.value)
