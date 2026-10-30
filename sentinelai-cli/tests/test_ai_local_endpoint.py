"""
Loopback-only validation of the Ollama endpoint (ai/local_endpoint.py).

What is being protected: ai/retrieval/context_retrieval.py builds its
retrieval query from the finding's raw_evidence - a verbatim excerpt of
the scanned repository - and ai/embeddings_ollama.py posts that text to
`{host}/api/embed`. A non-loopback host therefore sends repository
source off the machine. These tests pin the check at both places it is
applied: AISettings.llm_host, and make_ollama_embed_fn's own argument.

The second of those matters independently. A direct caller that never
builds an AISettings would otherwise bypass the field validator
entirely, so the "no request is sent" case below patches urlopen and
asserts it was never reached, rather than only asserting that an
exception was raised.
"""
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from sentinelai.ai.config import AISettings
from sentinelai.core import ExitCode
from sentinelai.ai.embeddings_ollama import make_ollama_embed_fn
from sentinelai.ai.local_endpoint import RemoteEndpointError, validate_local_endpoint

DEFAULT_HOST = "http://localhost:11434"


# --------------------------------------------------------------- accepted

@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:11434",
        "http://localhost",
        "https://localhost:11434",
        "HTTP://LOCALHOST:11434",  # scheme and host are both case-insensitive
        "http://127.0.0.1:11434",
        "http://127.0.0.1",
        "http://127.5.5.5:11434",  # the whole 127.0.0.0/8 block is loopback
        "http://[::1]:11434",  # IPv6 loopback, bracketed form
        "http://localhost:11434/",
    ],
)
def test_loopback_urls_are_accepted(url):
    assert validate_local_endpoint(url) == url


def test_the_shipped_default_is_accepted():
    """AISettings() must stay constructible with zero configuration."""
    assert AISettings().llm_host == DEFAULT_HOST
    assert validate_local_endpoint(DEFAULT_HOST) == DEFAULT_HOST


def test_surrounding_whitespace_is_tolerated_and_stripped():
    assert validate_local_endpoint("  http://localhost:11434  ") == DEFAULT_HOST


# --------------------------------------------------------------- rejected

@pytest.mark.parametrize(
    "url",
    [
        "http://ollama.example.com:11434",  # remote hostname
        "https://api.openai.com",
        "http://localhost.evil.com:11434",  # prefix trick, not localhost
        "http://notlocalhost:11434",
        "http://8.8.8.8:11434",  # public IPv4
        "http://192.168.1.50:11434",  # private but not loopback
        "http://10.0.0.5",
        "http://169.254.169.254",  # cloud metadata endpoint
        "http://[2001:4860:4860::8888]:11434",  # public IPv6
        "http://0.0.0.0:11434",  # unspecified, not loopback
    ],
)
def test_non_loopback_hosts_are_rejected(url):
    with pytest.raises(RemoteEndpointError):
        validate_local_endpoint(url)


def test_userinfo_cannot_disguise_a_remote_host():
    """`localhost@` is userinfo, not the host - the real host is still evil.com."""
    with pytest.raises(RemoteEndpointError):
        validate_local_endpoint("http://localhost@evil.com:11434")


@pytest.mark.parametrize(
    "url",
    [
        "ftp://localhost:11434",
        "file:///etc/passwd",
        "ws://localhost:11434",
        "localhost:11434",  # no scheme: urlsplit reads "localhost" as the scheme
        "//localhost:11434",
    ],
)
def test_unexpected_or_missing_schemes_are_rejected(url):
    with pytest.raises(RemoteEndpointError):
        validate_local_endpoint(url)


@pytest.mark.parametrize(
    "url",
    [
        "http://[::1:11434",  # unterminated IPv6 literal
        "http://localhost:notaport",
        "http://",
        "http:///api/embed",  # no host
        "not a url at all",
    ],
)
def test_malformed_urls_are_rejected(url):
    with pytest.raises(RemoteEndpointError):
        validate_local_endpoint(url)


@pytest.mark.parametrize("url", ["", "   ", None, 11434])
def test_empty_or_non_string_configuration_is_rejected(url):
    with pytest.raises(RemoteEndpointError):
        validate_local_endpoint(url)


def test_remote_endpoint_error_is_a_value_error():
    """Callers that already catch ValueError for a bad host keep working."""
    assert issubclass(RemoteEndpointError, ValueError)


# ------------------------------------------------- messages leak nothing

def test_the_message_names_the_host_but_not_the_whole_url():
    """A credential pasted into the setting must not be echoed back."""
    with pytest.raises(RemoteEndpointError) as excinfo:
        validate_local_endpoint("http://user:hunter2@ollama.example.com:11434/api/embed")
    message = str(excinfo.value)
    assert "ollama.example.com" in message
    assert "hunter2" not in message
    assert "user:" not in message


def test_the_message_says_how_to_fix_it():
    with pytest.raises(RemoteEndpointError) as excinfo:
        validate_local_endpoint("http://ollama.example.com")
    assert "http://localhost:11434" in str(excinfo.value)


# ------------------------------------------------------ applied in config

def test_settings_accept_a_loopback_host():
    assert AISettings(llm_host="http://127.0.0.1:11434").llm_host == "http://127.0.0.1:11434"


@pytest.mark.parametrize("url", ["http://ollama.example.com:11434", "http://8.8.8.8", "ftp://localhost"])
def test_settings_reject_a_non_local_host(url):
    with pytest.raises(ValueError):
        AISettings(llm_host=url)


def test_settings_reject_a_non_local_host_from_the_environment(monkeypatch):
    """The exposure being closed is an environment variable, so it must be refused there too."""
    monkeypatch.setenv("SENTINELAI_AI_LLM_HOST", "http://ollama.example.com:11434")
    from sentinelai.ai.config import _settings_from_env

    with pytest.raises(ValueError):
        _settings_from_env()


# --------------------------------------------- applied at the embed call

def test_embed_fn_builds_for_a_loopback_host():
    assert make_ollama_embed_fn(host=DEFAULT_HOST, model="nomic-embed-text") is not None


def test_embed_fn_refuses_a_remote_host():
    with pytest.raises(RemoteEndpointError):
        make_ollama_embed_fn(host="http://ollama.example.com:11434", model="nomic-embed-text")


def test_no_request_is_sent_when_the_host_is_refused():
    """The refusal happens before any socket is opened, not after a failed send."""
    with patch("urllib.request.urlopen") as urlopen:
        with pytest.raises(RemoteEndpointError):
            make_ollama_embed_fn(host="http://8.8.8.8:11434", model="nomic-embed-text")
    urlopen.assert_not_called()


def test_an_empty_host_is_still_refused_at_the_embed_call():
    """Preserves the previous contract: an empty host was already a ValueError here."""
    with pytest.raises(ValueError):
        make_ollama_embed_fn(host="", model="nomic-embed-text")


# --------------------------------------------------- surfaced by the CLI

def _run(monkeypatch, env, *args):
    """Invoke `scan` against the mock provider with a clean AI environment."""
    import sentinelai.ai.config as config_module
    from sentinelai.main import app
    from sentinelai.providers import MockFindingsProvider
    import sentinelai.main as main_module

    monkeypatch.setattr(main_module, "_get_provider", lambda: MockFindingsProvider())
    monkeypatch.setattr(config_module, "_settings", None)  # defeat the process-wide cache
    for name in ("SENTINELAI_AI_LLM_HOST", "SENTINELAI_AI_LLM_MODEL", "SENTINELAI_AI_EMBEDDING_MODEL"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    return CliRunner().invoke(app, ["scan", ".", *args])


AI_ON = {"SENTINELAI_AI_LLM_MODEL": "llama3.1:8b", "SENTINELAI_AI_EMBEDDING_MODEL": "nomic-embed-text"}


def test_cli_rejects_a_remote_host_when_ai_is_requested(monkeypatch):
    """Exit 2, not 1.

    A pydantic ValidationError escaping the command would make Typer exit 1,
    which is SECURITY_FINDINGS - a CI gate could not then distinguish a
    misconfigured host from findings crossing its threshold.
    """
    result = _run(monkeypatch, {**AI_ON, "SENTINELAI_AI_LLM_HOST": "http://ollama.example.com:11434"})
    assert result.exit_code == ExitCode.INVALID_INPUT
    flat = " ".join(result.output.split())
    assert "ollama.example.com" in flat
    assert "not on this machine" in flat


def test_cli_rejects_a_remote_host_under_the_explicit_ai_flag(monkeypatch):
    result = _run(monkeypatch, {**AI_ON, "SENTINELAI_AI_LLM_HOST": "http://8.8.8.8:11434"}, "--ai")
    assert result.exit_code == ExitCode.INVALID_INPUT


def test_cli_error_is_clean_not_a_traceback(monkeypatch):
    result = _run(monkeypatch, {**AI_ON, "SENTINELAI_AI_LLM_HOST": "http://8.8.8.8:11434"})
    assert result.exit_code == ExitCode.INVALID_INPUT
    assert "Traceback" not in result.output


# --- scanner-only runs must not be affected by AI settings they never read ---

def test_scanner_only_run_with_no_ai_environment(monkeypatch):
    """The common invocation. Unchanged."""
    result = _run(monkeypatch, {})
    assert result.exit_code == ExitCode.SUCCESS


def test_scanner_only_run_is_not_refused_by_an_unused_remote_host(monkeypatch):
    """A stale remote host must not fail a scan that never reads it.

    No model variables are set, so enrichment was never going to run and
    llm_host is never used. Validating it here would refuse a scanner-only run
    over a setting with no bearing on it.
    """
    result = _run(monkeypatch, {"SENTINELAI_AI_LLM_HOST": "http://ollama.example.com:11434"})
    assert result.exit_code == ExitCode.SUCCESS


def test_no_ai_flag_is_not_refused_by_a_remote_host(monkeypatch):
    """--no-ai is an explicit refusal of AI, so its configuration is irrelevant."""
    result = _run(
        monkeypatch,
        {**AI_ON, "SENTINELAI_AI_LLM_HOST": "http://ollama.example.com:11434"},
        "--no-ai",
    )
    assert result.exit_code == ExitCode.SUCCESS


def test_cli_accepts_the_default_loopback_host(monkeypatch):
    result = _run(monkeypatch, {"SENTINELAI_AI_LLM_HOST": DEFAULT_HOST})
    assert result.exit_code == ExitCode.SUCCESS


def test_ai_enabled_run_with_a_valid_local_host_passes_validation(monkeypatch):
    """Enrichment is stubbed: this asserts the host check passed, not that a model ran."""
    import sentinelai.main as main_module

    monkeypatch.setattr(main_module, "enrich_findings", lambda *a, **k: [])
    result = _run(monkeypatch, {**AI_ON, "SENTINELAI_AI_LLM_HOST": "http://127.0.0.1:11434"}, "--ai")
    assert result.exit_code == ExitCode.SUCCESS


def test_validation_happens_before_any_enrichment_call(monkeypatch):
    """A remote host must be refused before enrich_findings is ever reached."""
    import sentinelai.main as main_module

    calls = []

    def _spy(*args, **kwargs):
        calls.append(1)
        return []

    monkeypatch.setattr(main_module, "enrich_findings", _spy)
    result = _run(monkeypatch, {**AI_ON, "SENTINELAI_AI_LLM_HOST": "http://8.8.8.8:11434"})
    assert result.exit_code == ExitCode.INVALID_INPUT
    assert calls == [], "enrichment ran despite an invalid host"


def test_missing_models_reported_separately_from_a_bad_host(monkeypatch):
    """--ai with a valid host but no models keeps its own INVALID_INPUT message."""
    result = _run(monkeypatch, {"SENTINELAI_AI_LLM_HOST": DEFAULT_HOST}, "--ai")
    assert result.exit_code == ExitCode.INVALID_INPUT
    flat = " ".join(result.output.split())
    assert "SENTINELAI_AI_LLM_MODEL" in flat
    assert "not on this machine" not in flat
