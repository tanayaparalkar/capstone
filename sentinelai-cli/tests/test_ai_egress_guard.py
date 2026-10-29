"""Egress guard: policy, then secrets, then size - and nothing secret in any violation it raises."""
import ast
import json
from pathlib import Path
from typing import ClassVar, Optional

import pytest
from redaction_fixtures import PLANTED, PLANTED_TEXT

from sentinelai.ai.egress import EgressViolation, assert_safe
from sentinelai.ai.egress.guard import SAFETY_FACTOR
from sentinelai.ai.llm import LLMProvider, LLMResponse, MockLLMProvider
from sentinelai.ai.redaction import PATTERN_NAMES, redact_text
from sentinelai.contracts import AIConfig

DATA = Path(__file__).parent / "data" / "redaction"
CLEAN = (DATA / "clean_code.txt").read_bytes().decode("utf-8")
CLEAN_PAYLOAD = "Explain finding SENT-001 (sql-injection) at app/db/queries.py:11.\n" + CLEAN[:1500]


class _RemoteFake(LLMProvider):
    name: ClassVar[str] = "groq"
    is_local: ClassVar[bool] = False
    retains_data: ClassVar[Optional[bool]] = None

    def complete(self, prompt: str, *, max_tokens: int, temperature: float) -> LLMResponse:
        raise AssertionError("the guard must never call the provider")


def _cfg(provider: str, egress: str = "allow", **overrides) -> AIConfig:
    return AIConfig(provider=provider, model="m", egress=egress, **overrides)


REMOTE = _RemoteFake()
LOCAL = MockLLMProvider()


def _context_for(secret: str) -> str:
    """The fixture lines a planted secret sits on, plus one line either side (PEM header/footer, URL scheme)."""
    index = PLANTED_TEXT.index(secret)
    start = PLANTED_TEXT.rfind("\n", 0, PLANTED_TEXT.rfind("\n", 0, index)) + 1
    end = PLANTED_TEXT.find("\n", PLANTED_TEXT.find("\n", index + len(secret)) + 1)
    return PLANTED_TEXT[start : end if end != -1 else len(PLANTED_TEXT)]


def _assert_no_secret_in(exc: EgressViolation, secrets: list[str]) -> None:
    rendered = [str(exc), repr(exc), *map(str, exc.args), *exc.pattern_names]
    for secret in secrets:
        windows = {secret[i : i + 8] for i in range(len(secret) - 7)}
        for text in rendered:
            assert not any(window in text for window in windows), "violation message leaks a secret"


# ---- 1. policy ----


def test_deny_with_remote_provider_raises_even_for_a_clean_payload():
    with pytest.raises(EgressViolation) as excinfo:
        assert_safe(CLEAN_PAYLOAD, _cfg("groq", egress="deny"), REMOTE)
    assert excinfo.value.check == "policy"


def test_deny_is_the_default_and_refuses_remote():
    with pytest.raises(EgressViolation, match="policy"):
        assert_safe(CLEAN_PAYLOAD, AIConfig(provider="groq", model="m"), REMOTE)


def test_deny_with_remote_provider_does_not_inspect_the_payload():
    # A payload the secrets check would reject as unverifiable: policy must fire first.
    for payload in (None, b"bytes", PLANTED_TEXT, "x" * 10_000_000):
        with pytest.raises(EgressViolation) as excinfo:
            assert_safe(payload, _cfg("groq", egress="deny"), REMOTE)
        assert excinfo.value.check == "policy"
        assert excinfo.value.pattern_names == []


def test_deny_with_local_provider_passes():
    assert assert_safe(CLEAN_PAYLOAD, _cfg("mock", egress="deny"), LOCAL) is None


def test_allow_with_remote_provider_passes_a_clean_payload_under_budget():
    assert assert_safe(CLEAN_PAYLOAD, _cfg("groq"), REMOTE) is None


@pytest.mark.parametrize("is_local", [None, 1, "yes", False])
def test_anything_but_is_local_true_counts_as_remote(is_local):
    class Unverified(_RemoteFake):
        pass

    Unverified.is_local = is_local
    with pytest.raises(EgressViolation, match="policy"):
        assert_safe(CLEAN_PAYLOAD, _cfg("groq", egress="deny"), Unverified())


def test_provider_that_does_not_match_the_config_is_refused():
    # A config that names the local provider must not be usable to reach a remote one.
    with pytest.raises(EgressViolation, match="does not match") as excinfo:
        assert_safe(CLEAN_PAYLOAD, _cfg("mock", egress="deny"), REMOTE)
    assert excinfo.value.check == "policy"
    with pytest.raises(EgressViolation, match="does not match"):
        assert_safe(CLEAN_PAYLOAD, _cfg("ollama"), REMOTE)


def test_egress_value_outside_the_literal_counts_as_deny():
    cfg = AIConfig.model_construct(provider="groq", model="m", egress="ALLOW", max_context_chars=2000)
    with pytest.raises(EgressViolation, match="policy"):
        assert_safe(CLEAN_PAYLOAD, cfg, REMOTE)


# ---- 2. secrets ----


def test_planted_fixture_covers_every_pattern():
    assert {entry["pattern"] for entry in PLANTED} == PATTERN_NAMES


@pytest.mark.parametrize("entry", PLANTED, ids=lambda e: e["pattern"])
def test_allow_with_each_secret_type_raises_naming_the_pattern(entry):
    payload = "Explain this finding.\n" + _context_for(entry["secret"])
    with pytest.raises(EgressViolation) as excinfo:
        assert_safe(payload, _cfg("groq"), REMOTE)
    assert excinfo.value.check == "secrets"
    assert entry["pattern"] in excinfo.value.pattern_names
    assert entry["pattern"] in str(excinfo.value)


@pytest.mark.parametrize("entry", PLANTED, ids=lambda e: e["pattern"])
def test_violation_messages_never_contain_a_secret_value(entry):
    payload = _context_for(entry["secret"])
    with pytest.raises(EgressViolation) as excinfo:
        assert_safe(payload, _cfg("groq"), REMOTE)
    _assert_no_secret_in(excinfo.value, [entry["secret"]])


def test_whole_unredacted_fixture_reports_every_pattern_and_no_value():
    with pytest.raises(EgressViolation) as excinfo:
        assert_safe(PLANTED_TEXT, _cfg("groq", max_context_chars=10_000), REMOTE)
    assert set(excinfo.value.pattern_names) == PATTERN_NAMES
    _assert_no_secret_in(excinfo.value, [entry["secret"] for entry in PLANTED])


def test_no_secret_assertion_would_catch_a_leak():
    secret = PLANTED[0]["secret"]
    with pytest.raises(AssertionError):
        _assert_no_secret_in(EgressViolation("secrets", f"found {secret[2:12]}"), [secret])


def test_secrets_are_checked_for_local_providers_too():
    with pytest.raises(EgressViolation, match="secrets"):
        assert_safe(_context_for(PLANTED[0]["secret"]), _cfg("mock", egress="deny"), LOCAL)


def test_correctly_redacted_payload_passes():
    redacted, hits = redact_text(PLANTED_TEXT)
    assert hits
    assert assert_safe(redacted, _cfg("groq", max_context_chars=10_000), REMOTE) is None


def test_forged_marker_name_is_refused():
    payload = 'password = "«REDACTED:hunter2-plus»"'
    with pytest.raises(EgressViolation) as excinfo:
        assert_safe(payload, _cfg("groq"), REMOTE)
    assert excinfo.value.pattern_names == ["forged-redaction-marker"]
    assert "hunter2" not in str(excinfo.value)


def test_unterminated_marker_is_refused():
    with pytest.raises(EgressViolation, match="forged-redaction-marker"):
        assert_safe("token = «REDACTED:jwt and the rest of the line", _cfg("groq"), REMOTE)


def test_text_hiding_behind_a_genuine_marker_is_caught():
    # The shared accept filters skip a value that merely starts with a marker.
    payload = 'password = "«REDACTED:secret-assignment»hunter2-plus"'
    with pytest.raises(EgressViolation) as excinfo:
        assert_safe(payload, _cfg("groq"), REMOTE)
    assert excinfo.value.pattern_names == ["secret-assignment"]


def test_gitleaks_marker_names_are_genuine():
    payload = 'GENERIC = "«REDACTED:gitleaks:generic-api-key»"'
    assert assert_safe(payload, _cfg("groq"), REMOTE) is None


def test_non_string_payload_is_refused_when_egress_is_allowed():
    with pytest.raises(EgressViolation, match="cannot be verified"):
        assert_safe(b"bytes", _cfg("groq"), REMOTE)


def test_guard_works_even_if_the_redactor_is_broken(monkeypatch):
    import sentinelai.ai.redaction.redactor as redactor

    def broken(*args, **kwargs):
        raise AssertionError("the guard must not depend on the redactor's code path")

    monkeypatch.setattr(redactor, "redact_text", broken)
    monkeypatch.setattr(redactor, "_redact", broken)
    monkeypatch.setattr(redactor, "_pattern_spans", broken)
    with pytest.raises(EgressViolation, match="secrets"):
        assert_safe(_context_for(PLANTED[0]["secret"]), _cfg("groq"), REMOTE)
    assert assert_safe(CLEAN_PAYLOAD, _cfg("groq"), REMOTE) is None


def test_guard_imports_pattern_definitions_only():
    source = Path(__file__).parent.parent / "sentinelai" / "ai" / "egress" / "guard.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    modules = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert "sentinelai.ai.redaction.patterns" in modules
    assert not any(m and m.startswith("sentinelai.ai.redaction") and m != "sentinelai.ai.redaction.patterns" for m in modules)
    names = {alias.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) for alias in node.names}
    assert not names & {"redact_text", "SecretIndex", "_redact"}
    assert "logging" not in {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}


# ---- 3. size ----


def test_oversized_payload_raises():
    cfg = _cfg("groq", max_context_chars=2000)
    payload = "x = 1\n" * int(2000 * SAFETY_FACTOR)
    with pytest.raises(EgressViolation) as excinfo:
        assert_safe(payload, cfg, REMOTE)
    assert excinfo.value.check == "size"
    assert str(len(payload)) in str(excinfo.value)


def test_size_limit_boundary():
    cfg = _cfg("groq", max_context_chars=100)
    limit = int(100 * SAFETY_FACTOR)
    assert assert_safe("a" * limit, cfg, REMOTE) is None
    with pytest.raises(EgressViolation, match="size"):
        assert_safe("a" * (limit + 1), cfg, REMOTE)


def test_safety_factor_can_be_tightened():
    with pytest.raises(EgressViolation, match="size"):
        assert_safe("a" * 101, _cfg("groq", max_context_chars=100), REMOTE, safety_factor=1.0)


def test_nan_safety_factor_fails_closed():
    with pytest.raises(EgressViolation, match="size"):
        assert_safe("a", _cfg("groq"), REMOTE, safety_factor=float("nan"))


def test_size_applies_to_local_providers_too():
    with pytest.raises(EgressViolation, match="size"):
        assert_safe("a" * 10_000, _cfg("mock", max_context_chars=10), LOCAL)


def test_default_budget_admits_the_largest_current_prompt_shape():
    # ~6.4k fixed instructions + a full slice + five KB chunks + finding/prior sections.
    assert assert_safe("a" * 14_000, _cfg("groq"), REMOTE) is None


# ---- order ----


def test_policy_is_checked_before_secrets_and_size():
    with pytest.raises(EgressViolation) as excinfo:
        assert_safe(PLANTED_TEXT * 50, _cfg("groq", egress="deny"), REMOTE)
    assert excinfo.value.check == "policy"


def test_secrets_are_checked_before_size():
    with pytest.raises(EgressViolation) as excinfo:
        assert_safe(PLANTED_TEXT * 50, _cfg("groq"), REMOTE)
    assert excinfo.value.check == "secrets"
