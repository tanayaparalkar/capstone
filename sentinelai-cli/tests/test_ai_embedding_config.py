"""
Focused tests for the two-model AI configuration and the embedding
error messages.

Deliberately small - this covers the behaviour Section 1 introduced and
nothing else:

1. Missing model configuration disables AI enrichment; it is never a
   failure, and a scan still succeeds scanner-only.
2. Ollama's HTTP 501 (a text-generation model configured as the
   embedding model) becomes an error that names the model, the reason,
   and the fix, rather than a bare "HTTP Error 501: Not Implemented".

The messages are asserted, not just the exception type: the entire point
of this change is what the user reads when it goes wrong.

The network layer is faked with unittest.mock.patch on
urllib.request.urlopen, matching the convention ai/llm_ollama.py's
docstring already establishes for this codebase.
"""
import json
import urllib.error
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from sentinelai.ai.config import AISettings
from sentinelai.ai.embeddings_ollama import make_ollama_embed_fn
from sentinelai.core import ExitCode
from sentinelai.core.errors import AIEnrichmentError, SentinelAIError
from sentinelai.main import app
from sentinelai.providers import MockFindingsProvider

runner = CliRunner()
HOST = "http://localhost:11434"


def _flat(text: str) -> str:
    """Collapse whitespace so assertions survive Rich's console-width line wrapping."""
    return " ".join(text.split())


@pytest.fixture(autouse=True)
def _use_mock_provider(monkeypatch):
    import sentinelai.main as main_module

    monkeypatch.setattr(main_module, "_get_provider", lambda: MockFindingsProvider())


class _FakeResponse:
    def __init__(self, body: str):
        self._body = body.encode("utf-8")

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _http_error(code: int, reason: str) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(f"{HOST}/api/embed", code, reason, {}, None)


def _embed_fn(model: str = "nomic-embed-text"):
    return make_ollama_embed_fn(host=HOST, model=model)


# --- 1. two-model configuration: missing either model disables AI ---------------------------------------------------


@pytest.mark.parametrize(
    "settings",
    [
        AISettings(),  # neither set
        AISettings(llm_model="llama3.1:8b"),  # embedding model missing
        AISettings(embedding_model="nomic-embed-text"),  # llm model missing
    ],
    ids=["neither", "no-embedding-model", "no-llm-model"],
)
def test_incomplete_configuration_disables_ai_without_failing(monkeypatch, tmp_path, settings):
    monkeypatch.setattr("sentinelai.main.get_settings", lambda: settings)
    # Constructing either AI dependency would mean AI was not skipped.
    monkeypatch.setattr(
        "sentinelai.main.create_default_retriever",
        lambda: (_ for _ in ()).throw(AssertionError("AI must be skipped")),
    )
    monkeypatch.setattr(
        "sentinelai.main.create_llm_generate_fn",
        lambda: (_ for _ in ()).throw(AssertionError("AI must be skipped")),
    )

    result = runner.invoke(app, ["scan", str(tmp_path), "--format", "json"])

    assert result.exit_code == ExitCode.SUCCESS
    assert json.loads(result.output)["findings"]["ai_enriched"] == []


def test_scanner_only_mode_still_reports_and_gates(tmp_path):
    # Scanner-only is a first-class mode: findings, statistics, and --fail-on all work.
    result = runner.invoke(app, ["scan", str(tmp_path), "--fail-on", "critical", "--format", "json"])

    assert result.exit_code == ExitCode.SECURITY_FINDINGS
    data = json.loads(result.output)
    assert data["statistics"]["ai_enrichment_status"] == "unavailable"
    assert len(data["findings"]["scanner"]) > 0


def test_embedding_model_is_optional_on_the_settings_model():
    # AISettings() must stay constructible with zero configuration.
    assert AISettings().embedding_model is None
    assert AISettings().llm_model is None


# --- 2. HTTP 501: the reported misconfiguration ----------------------------------------------------------------------


def test_http_501_names_the_model_the_reason_and_the_fix():
    with patch("urllib.request.urlopen", side_effect=_http_error(501, "Not Implemented")):
        with pytest.raises(AIEnrichmentError) as excinfo:
            _embed_fn("llama3.1:8b")(["text"])

    message = str(excinfo.value)
    assert "llama3.1:8b" in message                                   # which model
    assert "does not support embeddings" in message                   # why
    assert "ollama pull nomic-embed-text" in message                  # how to fix
    assert "SENTINELAI_AI_EMBEDDING_MODEL=nomic-embed-text" in message
    assert "must not be set to the same value" in message


def test_http_404_says_the_model_is_not_pulled():
    with patch("urllib.request.urlopen", side_effect=_http_error(404, "Not Found")):
        with pytest.raises(AIEnrichmentError) as excinfo:
            _embed_fn("nomic-embed-text")(["text"])

    assert "no model named 'nomic-embed-text'" in str(excinfo.value)


def test_unreachable_server_says_so():
    # time.sleep is patched because a connection failure is retryable
    # (ai/ollama_http.py) and would otherwise spend the real 2-second backoff
    # here. This test is about the actionable message, not the retry; retry
    # behaviour has its own coverage in tests/test_ai_ollama_retry.py.
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("Connection refused")), patch("time.sleep"):
        with pytest.raises(AIEnrichmentError) as excinfo:
            _embed_fn()(["text"])

    assert "could not reach the Ollama server" in str(excinfo.value)
    assert "ollama serve" in str(excinfo.value)


def test_successful_embedding_is_unchanged():
    body = json.dumps({"embeddings": [[0.1, 0.2]]})

    with patch("urllib.request.urlopen", return_value=_FakeResponse(body)):
        assert _embed_fn()(["text"]) == [[0.1, 0.2]]


def test_empty_input_still_short_circuits_without_a_network_call():
    with patch("urllib.request.urlopen", side_effect=AssertionError("must not be called")):
        assert _embed_fn()([]) == []


def test_ai_enrichment_error_is_a_sentinelai_error():
    assert issubclass(AIEnrichmentError, SentinelAIError)


# --- the error reaches the user as one clean line, not a traceback ---------------------------------------------------


def _misconfigure_embedding_model(monkeypatch):
    """The reported setup: embedding model set to the generation model, so /api/embed 501s.

    main.py and factory.py each call get_settings() for themselves. In production that
    returns the same cached singleton to both; in a test they have to be patched
    together, or main would see AI as configured while the factory still saw an unset
    environment and failed on validation instead of reaching Ollama.
    """
    settings = AISettings(llm_model="llama3.1:8b", embedding_model="llama3.1:8b")
    monkeypatch.setattr("sentinelai.main.get_settings", lambda: settings)
    monkeypatch.setattr("sentinelai.ai.factory.get_settings", lambda: settings)
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *a, **k: (_ for _ in ()).throw(_http_error(501, "Not Implemented")),
    )


def test_cli_reports_the_501_cleanly_and_exits_provider_error(monkeypatch, tmp_path):
    _misconfigure_embedding_model(monkeypatch)

    result = runner.invoke(app, ["scan", str(tmp_path), "--format", "json"])

    assert result.exit_code == ExitCode.PROVIDER_ERROR
    stderr = _flat(result.stderr)
    assert "AI enrichment failed" in stderr
    assert "does not support embeddings" in stderr
    assert "nomic-embed-text" in stderr
    assert "Traceback" not in result.stderr


def test_debug_flag_still_shows_the_traceback(monkeypatch, tmp_path):
    _misconfigure_embedding_model(monkeypatch)

    result = runner.invoke(app, ["--debug", "scan", str(tmp_path), "--format", "json"])

    assert result.exit_code == ExitCode.PROVIDER_ERROR
    assert "Traceback" in result.stderr
