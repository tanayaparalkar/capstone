"""
Explicit --ai / --no-ai control over AI enrichment.

Before these flags, enrichment was decided entirely by invisible
environment state: it ran whenever SENTINELAI_AI_LLM_MODEL and
SENTINELAI_AI_EMBEDDING_MODEL both happened to be set, with no way to
demand it and no way to decline it. Two consequences, both real:
a user who mistyped a variable name got a silently scanner-only report
that looked like a successful AI run, and a user with the variables
exported in their shell profile could not get a fast deterministic scan
without unsetting them.

The backward-compatibility tests here matter most. With neither flag the
behaviour must be bit-for-bit what it always was, because every existing
invocation, every benchmark, and this project's own CI pass no flag.
"""
import pytest
from typer.testing import CliRunner

from sentinelai.contracts import ScanMode, ScannerTier
from sentinelai.core import ExitCode
from sentinelai.main import app
from sentinelai.providers import MockFindingsProvider

runner = CliRunner()

_LLM = "SENTINELAI_AI_LLM_MODEL"
_EMBED = "SENTINELAI_AI_EMBEDDING_MODEL"


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    """Pin the provider and reset the cached settings singleton for every test.

    get_settings() caches AISettings process-wide on first call, so without
    clearing it one test's environment would decide every later test's answer.
    """
    import sentinelai.ai.config as config
    import sentinelai.main as main_module

    monkeypatch.setattr(main_module, "_get_provider", lambda: MockFindingsProvider())
    monkeypatch.setattr(config, "_settings", None)
    monkeypatch.delenv(_LLM, raising=False)
    monkeypatch.delenv(_EMBED, raising=False)
    yield
    monkeypatch.setattr(config, "_settings", None)


def _flat(result) -> str:
    """stderr with wrapping collapsed.

    Rich hard-wraps error output to the terminal width, so a phrase like
    "ollama pull nomic-embed-text" can arrive split across two lines. Asserting
    on the raw text would be asserting on the terminal width.
    """
    return " ".join(result.stderr.split())


def _diagnosis(result) -> str:
    """The clause that states what is wrong, excluding the remediation that follows.

    The message deliberately names *both* variables when telling the user what to
    set - a complete recipe is more useful than a minimal one - while naming only
    the missing variable(s) when saying what is wrong. This isolates the latter.
    """
    flat = _flat(result)
    for marker in (" is not set.", " are not set."):
        if marker in flat:
            return flat.split(marker)[0]
    return flat


def _configure(monkeypatch):
    monkeypatch.setenv(_LLM, "llama3.1:8b")
    monkeypatch.setenv(_EMBED, "nomic-embed-text")


def _spy_enrichment(monkeypatch):
    """Record whether enrich_findings was reached, without running the AI layer."""
    calls = []

    def fake_enrich(*args, **kwargs):
        calls.append(True)
        return []

    import sentinelai.main as main_module

    monkeypatch.setattr(main_module, "enrich_findings", fake_enrich)
    return calls


# --- case 1: --ai with missing configuration ------------------------------------------------------------------------


def test_ai_flag_without_configuration_exits_invalid_input(monkeypatch):
    result = runner.invoke(app, ["scan", "."])
    assert result.exit_code == 0  # baseline: without the flag this is a normal scanner-only run

    result = runner.invoke(app, ["scan", ".", "--ai"])

    assert result.exit_code == ExitCode.INVALID_INPUT


def test_the_message_names_both_missing_variables(monkeypatch):
    result = runner.invoke(app, ["scan", ".", "--ai"])

    diagnosis = _diagnosis(result)
    assert _LLM in diagnosis
    assert _EMBED in diagnosis


def test_the_message_names_only_the_variable_that_is_missing(monkeypatch):
    """A user who set one of the two should not be told to set both."""
    monkeypatch.setenv(_LLM, "llama3.1:8b")

    result = runner.invoke(app, ["scan", ".", "--ai"])

    diagnosis = _diagnosis(result)
    assert _EMBED in diagnosis
    assert _LLM not in diagnosis
    # The remediation still gives the full recipe, including the variable that
    # is already set - that is deliberate, not a leak of the diagnosis.
    assert _LLM in _flat(result)


def test_the_message_recommends_the_benchmarked_models_and_no_api_key(monkeypatch):
    result = runner.invoke(app, ["scan", ".", "--ai"])

    flat = _flat(result)
    assert "ollama pull llama3.1:8b" in flat
    assert "ollama pull nomic-embed-text" in flat
    assert "ollama serve" in flat
    assert "No API key is required" in flat


def test_failure_happens_before_any_scan_runs(monkeypatch):
    """Validation must not cost the user a full scan first."""
    scans = []

    def fake_provider():
        class _Recording(MockFindingsProvider):
            def get_scan_result(self, *args, **kwargs):
                scans.append(True)
                return super().get_scan_result(*args, **kwargs)

        return _Recording()

    import sentinelai.main as main_module

    monkeypatch.setattr(main_module, "_get_provider", fake_provider)

    result = runner.invoke(app, ["scan", ".", "--ai"])

    assert result.exit_code == ExitCode.INVALID_INPUT
    assert scans == [], "a scan ran before --ai was validated"


# --- case 2: --ai with configuration --------------------------------------------------------------------------------


def test_ai_flag_with_configuration_enriches(monkeypatch):
    _configure(monkeypatch)
    calls = _spy_enrichment(monkeypatch)

    result = runner.invoke(app, ["scan", ".", "--ai"])

    assert result.exit_code == 0
    assert calls == [True]


# --- case 3: --no-ai ------------------------------------------------------------------------------------------------


def test_no_ai_skips_enrichment_even_when_fully_configured(monkeypatch):
    _configure(monkeypatch)
    calls = _spy_enrichment(monkeypatch)

    result = runner.invoke(app, ["scan", ".", "--no-ai"])

    assert result.exit_code == 0
    assert calls == [], "--no-ai still ran enrichment"


def test_no_ai_without_configuration_is_simply_a_normal_scan(monkeypatch):
    calls = _spy_enrichment(monkeypatch)

    result = runner.invoke(app, ["scan", ".", "--no-ai"])

    assert result.exit_code == 0
    assert calls == []


# --- case 4: neither flag preserves auto-detect exactly ---------------------------------------------------------------


def test_neither_flag_enriches_when_configured(monkeypatch):
    _configure(monkeypatch)
    calls = _spy_enrichment(monkeypatch)

    result = runner.invoke(app, ["scan", "."])

    assert result.exit_code == 0
    assert calls == [True], "auto-detect no longer enriches when configured"


def test_neither_flag_stays_scanner_only_when_unconfigured(monkeypatch):
    calls = _spy_enrichment(monkeypatch)

    result = runner.invoke(app, ["scan", "."])

    assert result.exit_code == 0
    assert calls == []


def test_partial_configuration_without_the_flag_still_silently_skips(monkeypatch):
    """Unchanged pre-existing behaviour: only --ai turns a partial config into an error."""
    monkeypatch.setenv(_LLM, "llama3.1:8b")
    calls = _spy_enrichment(monkeypatch)

    result = runner.invoke(app, ["scan", "."])

    assert result.exit_code == 0
    assert calls == []


# --- mutual exclusion -------------------------------------------------------------------------------------------------


def test_ai_and_no_ai_together_is_rejected(monkeypatch):
    _configure(monkeypatch)

    result = runner.invoke(app, ["scan", ".", "--ai", "--no-ai"])

    assert result.exit_code == ExitCode.INVALID_INPUT
    assert "cannot be used together" in _flat(result)


# --- composition with the existing flags ------------------------------------------------------------------------------


@pytest.mark.parametrize("other", [["--quick"], ["--full"], ["--extended"], ["--fail-on", "high"]])
def test_no_ai_composes_with_existing_flags(monkeypatch, other):
    _configure(monkeypatch)
    calls = _spy_enrichment(monkeypatch)

    result = runner.invoke(app, ["scan", ".", "--no-ai", *other])

    assert result.exit_code in (0, ExitCode.SECURITY_FINDINGS)
    assert calls == []


def test_ai_flag_does_not_affect_the_scanner_tier(monkeypatch):
    """Enrichment and scanner selection are independent axes."""
    _configure(monkeypatch)
    _spy_enrichment(monkeypatch)
    seen = []

    original = MockFindingsProvider.get_scan_result

    def wrapper(self, repo_path, mode=ScanMode.STANDARD, tier=ScannerTier.CORE):
        seen.append(tier)
        return original(self, repo_path, mode=mode, tier=tier)

    monkeypatch.setattr(MockFindingsProvider, "get_scan_result", wrapper)

    assert runner.invoke(app, ["scan", ".", "--ai"]).exit_code == 0
    assert seen == [ScannerTier.CORE]


# --- help text --------------------------------------------------------------------------------------------------------


def test_both_flags_are_documented_in_help():
    output = runner.invoke(app, ["scan", "--help"]).output

    assert "--ai" in output
    assert "--no-ai" in output
