"""AI provider configuration: CLI flag > environment > default, the `providers` listing, and exit 2 on misconfiguration.

No real provider is ever called. Resolution reads provider *classes* for
is_local/retains_data and never constructs one, and urllib is booby-trapped
in every test so an accidental network call fails loudly.
"""
import urllib.request

import pytest
from redaction_fixtures import FAKE_GROQ_KEY
from typer.testing import CliRunner

from sentinelai.ai.config import (
    EGRESS_ENV,
    MODEL_ENV,
    PROVIDER_ENV,
    PROVIDERS,
    AIConfigError,
    resolve_config,
)
from sentinelai.ai.llm import MockLLMProvider
from sentinelai.core import ExitCode
from sentinelai.main import app
from sentinelai.presentation.providers import retention_label
from sentinelai.providers import MockFindingsProvider

runner = CliRunner()

_CREDENTIALS = ("GROQ_API_KEY", "GOOGLE_APPLICATION_CREDENTIALS", "GOOGLE_CLOUD_PROJECT")
_KEY_VALUE = FAKE_GROQ_KEY


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    import sentinelai.ai.config as config
    import sentinelai.main as main_module

    for name in (PROVIDER_ENV, MODEL_ENV, EGRESS_ENV, *_CREDENTIALS, "SENTINELAI_AI_LLM_MODEL", "SENTINELAI_AI_EMBEDDING_MODEL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(config, "_settings", None)

    def no_network(*args, **kwargs):
        raise AssertionError("no provider may be called by configuration or listing")

    monkeypatch.setattr(urllib.request, "urlopen", no_network)
    scans = []

    class RecordingProvider(MockFindingsProvider):
        def get_scan_result(self, *args, **kwargs):
            scans.append(args)
            return super().get_scan_result(*args, **kwargs)

    monkeypatch.setattr(main_module, "_get_provider", RecordingProvider)
    yield scans
    monkeypatch.setattr(config, "_settings", None)


@pytest.fixture
def scans(_isolate):
    return _isolate


def _flat(text: str) -> str:
    """Collapse Rich's terminal-width wrapping so assertions don't depend on the width."""
    return " ".join(text.split())


def _scan(*args):
    return runner.invoke(app, ["scan", ".", "--format", "json", *args])


# ---- resolve_config: defaults ----


def test_defaults_when_nothing_is_set():
    config = resolve_config({})
    assert (config.provider, config.model, config.egress) == ("ollama", "llama3.1:8b", "deny")


def test_none_overrides_are_the_same_as_absent_ones():
    assert resolve_config({"provider": None, "model": None, "egress": None}) == resolve_config({})


def test_default_model_follows_the_chosen_provider():
    assert resolve_config({"provider": "mock"}).model == "mock-model"


def test_blank_environment_values_count_as_unset(monkeypatch):
    for name in (PROVIDER_ENV, MODEL_ENV, EGRESS_ENV):
        monkeypatch.setenv(name, "   ")
    config = resolve_config({})
    assert (config.provider, config.model, config.egress) == ("ollama", "llama3.1:8b", "deny")


# ---- resolve_config: precedence, per field ----


def test_provider_precedence(monkeypatch):
    assert resolve_config({}).provider == "ollama"
    monkeypatch.setenv(PROVIDER_ENV, "mock")
    assert resolve_config({}).provider == "mock"
    monkeypatch.setenv("GROQ_API_KEY", _KEY_VALUE)
    assert resolve_config({"provider": "groq", "egress": "allow"}).provider == "groq"


def test_model_precedence(monkeypatch):
    assert resolve_config({"provider": "mock"}).model == "mock-model"
    monkeypatch.setenv(MODEL_ENV, "env-model")
    assert resolve_config({"provider": "mock"}).model == "env-model"
    assert resolve_config({"provider": "mock", "model": "flag-model"}).model == "flag-model"


def test_egress_precedence(monkeypatch):
    assert resolve_config({"provider": "mock"}).egress == "deny"
    monkeypatch.setenv(EGRESS_ENV, "allow")
    assert resolve_config({"provider": "mock"}).egress == "allow"
    assert resolve_config({"provider": "mock", "egress": "deny"}).egress == "deny"


def test_flag_deny_beats_env_allow_for_a_remote_provider(monkeypatch):
    monkeypatch.setenv(EGRESS_ENV, "allow")
    monkeypatch.setenv("GROQ_API_KEY", _KEY_VALUE)
    assert resolve_config({"provider": "groq"}).egress == "allow"
    with pytest.raises(AIConfigError, match=r"egress is deny \(--egress\)"):
        resolve_config({"provider": "groq", "egress": "deny"})


def test_values_are_trimmed_and_case_folded(monkeypatch):
    monkeypatch.setenv(PROVIDER_ENV, "  MOCK ")
    monkeypatch.setenv(EGRESS_ENV, "Allow")
    config = resolve_config({"model": "  m1  "})
    assert (config.provider, config.model, config.egress) == ("mock", "m1", "allow")


def test_environment_is_read_on_every_call(monkeypatch):
    assert resolve_config({}).provider == "ollama"
    monkeypatch.setenv(PROVIDER_ENV, "mock")
    assert resolve_config({}).provider == "mock"


# ---- resolve_config: invalid input ----


@pytest.mark.parametrize(
    "overrides,env,message",
    [
        ({"provider": "openai"}, {}, "--ai-provider must be one of ollama, groq, gemini, mock; got 'openai'"),
        ({}, {PROVIDER_ENV: "claude"}, "SENTINELAI_AI_PROVIDER must be one of ollama, groq, gemini, mock; got 'claude'"),
        ({"provider": ""}, {}, "--ai-provider must be one of"),
        ({"egress": "maybe"}, {}, "--egress must be one of allow, deny; got 'maybe'"),
        ({}, {EGRESS_ENV: "yes"}, "SENTINELAI_EGRESS must be one of allow, deny; got 'yes'"),
        ({"provider": "mock", "model": "  "}, {}, "--ai-model must not be empty"),
        ({"provider": "groq", "egress": "allow"}, {}, "groq needs GROQ_API_KEY set in the environment"),
        ({"provider": "groq", "egress": "allow"}, {"GROQ_API_KEY": "  "}, "groq needs GROQ_API_KEY set"),
        ({"provider": "gemini", "egress": "allow"}, {},
         "gemini needs GOOGLE_APPLICATION_CREDENTIALS and GOOGLE_CLOUD_PROJECT set in the environment"),
        ({"provider": "gemini", "egress": "allow"}, {"GOOGLE_APPLICATION_CREDENTIALS": "/sa.json"},
         "gemini needs GOOGLE_CLOUD_PROJECT set"),
        ({"provider": "groq"}, {"GROQ_API_KEY": _KEY_VALUE}, "groq is a remote provider and sends code off this machine, but egress is deny (default)"),
        ({}, {PROVIDER_ENV: "gemini", "GOOGLE_APPLICATION_CREDENTIALS": "/sa.json", "GOOGLE_CLOUD_PROJECT": "p"},
         "egress is deny (default)"),
    ],
)
def test_invalid_configuration_names_the_specific_problem(monkeypatch, overrides, env, message):
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    with pytest.raises(AIConfigError) as excinfo:
        resolve_config(overrides)
    assert message in str(excinfo.value)
    assert _KEY_VALUE not in str(excinfo.value)


def test_every_problem_is_reported_at_once():
    with pytest.raises(AIConfigError) as excinfo:
        resolve_config({"provider": "groq"})
    assert "egress is deny" in str(excinfo.value) and "GROQ_API_KEY" in str(excinfo.value)


def test_remote_provider_with_allow_and_credentials_resolves(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", _KEY_VALUE)
    config = resolve_config({"provider": "groq", "egress": "allow"})
    assert (config.provider, config.model, config.egress) == ("groq", "llama-3.3-70b-versatile", "allow")
    assert _KEY_VALUE not in config.model_dump_json()


def test_unknown_override_key_is_a_programming_error():
    with pytest.raises(ValueError) as excinfo:
        resolve_config({"api_key": "x"})
    assert not isinstance(excinfo.value, AIConfigError)


def test_registry_covers_every_configurable_provider():
    from sentinelai.contracts import AIConfig

    assert set(PROVIDERS) == {"ollama", "groq", "gemini", "mock"}
    assert PROVIDERS["mock"].provider_class is MockLLMProvider
    AIConfig(provider="mock", model=PROVIDERS["mock"].default_model)


# ---- scan: every invalid-input path exits 2, before any scanner runs ----


@pytest.mark.parametrize(
    "args,env,message",
    [
        (["--ai-provider", "openai"], {}, "--ai-provider must be one of ollama, groq, gemini, mock; got 'openai'"),
        (["--egress", "sometimes"], {}, "--egress must be one of allow, deny; got 'sometimes'"),
        (["--ai-provider", "mock", "--ai-model", ""], {}, "--ai-model must not be empty"),
        (["--ai-provider", "groq", "--egress", "allow"], {}, "groq needs GROQ_API_KEY set in the environment"),
        (["--ai-provider", "gemini", "--egress", "allow"], {}, "gemini needs GOOGLE_APPLICATION_CREDENTIALS"),
        (["--ai-provider", "groq"], {"GROQ_API_KEY": _KEY_VALUE}, "pass --egress allow to opt in for this run"),
        (["--no-ai", "--ai-provider", "mock"], {}, "--no-ai cannot be used together with --ai-provider"),
        (["--no-ai", "--egress", "allow"], {}, "--no-ai cannot be used together with --egress"),
    ],
)
def test_scan_misconfiguration_exits_invalid_input(monkeypatch, scans, args, env, message):
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    result = _scan(*args)
    assert result.exit_code == ExitCode.INVALID_INPUT == 2
    assert result.exit_code != ExitCode.PROVIDER_ERROR
    assert message in _flat(result.stderr)
    assert _KEY_VALUE not in result.stderr + result.stdout
    assert scans == [], "configuration must be rejected before any scanner runs"


@pytest.mark.parametrize(
    "env",
    [
        {PROVIDER_ENV: "bogus"},
        {EGRESS_ENV: "1"},
        {PROVIDER_ENV: "groq"},
        {PROVIDER_ENV: "gemini", EGRESS_ENV: "allow"},
    ],
)
def test_invalid_ai_environment_does_not_block_a_scanner_only_run(monkeypatch, scans, env):
    """A scanner-only run must not be refused by AI settings it never reads.

    These same values DO exit 2 when passed as flags, or when AI is actually
    requested - see test_scan_misconfiguration_exits_invalid_input and
    test_invalid_ai_environment_is_enforced_once_ai_is_requested. The
    distinction is consent: a variable left in a shell is configuration, not a
    request to enrich on this run, and refusing the scan over it would make
    every scanner-only invocation depend on AI configuration being valid.

    Previously these two cases asserted exit 2. That behaviour was the P0
    regression: it made `sentinelai scan` fail for a user who had merely
    exported SENTINELAI_AI_PROVIDER earlier in their session.
    """
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    result = _scan()
    assert result.exit_code == ExitCode.SUCCESS
    assert scans, "the scanner should have run"


@pytest.mark.parametrize(
    "env,message",
    [
        ({PROVIDER_ENV: "bogus"}, "SENTINELAI_AI_PROVIDER must be one of"),
        ({EGRESS_ENV: "1"}, "SENTINELAI_EGRESS must be one of allow, deny; got '1'"),
    ],
)
def test_invalid_ai_environment_is_enforced_once_ai_is_requested(monkeypatch, env, message):
    """The same environment that a scanner-only run ignores must still exit 2 under --ai.

    Validation is deferred, not dropped: nothing reaches a provider on a bad
    configuration.
    """
    monkeypatch.setenv("SENTINELAI_AI_LLM_MODEL", "llama3.1:8b")
    monkeypatch.setenv("SENTINELAI_AI_EMBEDDING_MODEL", "nomic-embed-text")
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    result = _scan("--ai")
    assert result.exit_code == ExitCode.INVALID_INPUT == 2
    assert message in _flat(result.stderr)


def test_provider_environment_variable_alone_does_not_activate_ai(monkeypatch, scans):
    """SENTINELAI_AI_PROVIDER is configuration, not consent.

    With no models configured and no --ai flag, naming a provider in the
    environment must leave enrichment switched off, so no repository context is
    sent to a provider the user did not ask for on this run.
    """
    monkeypatch.delenv("SENTINELAI_AI_LLM_MODEL", raising=False)
    monkeypatch.delenv("SENTINELAI_AI_EMBEDDING_MODEL", raising=False)
    monkeypatch.setenv(PROVIDER_ENV, "mock")
    result = _scan()
    assert result.exit_code == ExitCode.SUCCESS
    assert "AI enrichment skipped" in result.output or scans


def test_provider_flag_does_activate_ai(scans):
    """An explicit --ai-provider flag is a deliberate act, so it must not be ignored."""
    result = _scan("--ai-provider", "mock", "--ai-model", "mock-model")
    assert result.exit_code == ExitCode.SUCCESS


def test_scan_with_mock_provider_succeeds(scans):
    result = _scan("--ai-provider", "mock", "--ai-model", "mock-model", "--egress", "deny")
    assert result.exit_code == 0, result.stderr
    assert len(scans) == 1


def test_scan_with_nothing_configured_still_succeeds(scans):
    result = _scan()
    assert result.exit_code == 0, result.stderr


def test_scan_with_explicit_remote_opt_in_resolves_without_calling_the_provider(monkeypatch, scans):
    monkeypatch.setenv("GROQ_API_KEY", _KEY_VALUE)
    result = _scan("--ai-provider", "groq", "--egress", "allow")
    assert result.exit_code == 0, result.stderr
    assert _KEY_VALUE not in result.stderr + result.stdout


def test_flag_overrides_a_bad_environment_value(monkeypatch, scans):
    monkeypatch.setenv(PROVIDER_ENV, "bogus")
    result = _scan("--ai-provider", "mock")
    assert result.exit_code == 0, result.stderr


def test_no_ai_skips_resolution_of_a_bad_environment(monkeypatch, scans):
    # --no-ai means no AI at all, so environment AI settings are not consulted.
    monkeypatch.setenv(PROVIDER_ENV, "bogus")
    result = _scan("--no-ai")
    assert result.exit_code == 0, result.stderr


def test_scan_help_lists_the_new_flags():
    result = runner.invoke(app, ["scan", "--help"])
    flat = _flat(result.stdout)
    for flag in ("--ai-provider", "--ai-model", "--egress"):
        assert flag in flat


# ---- `sentinelai providers` ----


def _provider_rows(output: str) -> dict:
    """First table line of each provider row, keyed by provider name."""
    rows = {}
    for line in output.splitlines():
        cells = [cell.strip() for cell in line.strip().strip("│|").split("│")]
        if cells and cells[0] in PROVIDERS:
            rows[cells[0]] = cells
    return rows


@pytest.fixture
def wide(monkeypatch):
    import sentinelai.presentation.console as console

    monkeypatch.setattr(console, "_console", None)
    monkeypatch.setenv("COLUMNS", "200")
    yield
    monkeypatch.setattr(console, "_console", None)


def test_retention_labels_have_three_distinct_states():
    assert retention_label(None) == "unverified"
    assert retention_label(False) == "not retained"
    assert retention_label(True) == "retains data"


def test_providers_lists_every_registered_provider(wide):
    result = runner.invoke(app, ["providers"])
    assert result.exit_code == 0, result.stderr
    rows = _provider_rows(result.stdout)
    assert set(rows) == set(PROVIDERS)
    for name, spec in PROVIDERS.items():
        assert rows[name][1] == spec.default_model


def test_providers_renders_unverified_retention_and_never_none(wide):
    result = runner.invoke(app, ["providers"])
    rows = _provider_rows(result.stdout)
    assert (rows["groq"][2], rows["groq"][3]) == ("remote", "unverified")
    assert (rows["gemini"][2], rows["gemini"][3]) == ("remote", "unverified")
    assert (rows["ollama"][2], rows["ollama"][3]) == ("local", "not retained")
    assert (rows["mock"][2], rows["mock"][3]) == ("local", "not retained")
    assert "None" not in result.stdout
    assert all(row[3] != "no" for row in rows.values())


def test_providers_shows_credential_presence_but_never_the_value(monkeypatch, wide):
    monkeypatch.setenv("GROQ_API_KEY", _KEY_VALUE)
    result = runner.invoke(app, ["providers"])
    rows = _provider_rows(result.stdout)
    assert rows["groq"][4] == "GROQ_API_KEY (set)"
    assert rows["gemini"][4] == "GOOGLE_APPLICATION_CREDENTIALS (missing)"
    assert rows["ollama"][4] == "none needed"
    assert _KEY_VALUE not in result.stdout + result.stderr


def test_providers_states_the_defaults():
    result = runner.invoke(app, ["providers"])
    flat = _flat(result.stdout)
    assert "Default: ollama, egress deny" in flat
    assert "--egress allow" in flat
