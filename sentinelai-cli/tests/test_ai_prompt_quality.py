"""
Regression tests for two prompt-quality failures observed in live runs.

Both were found by running the real model, not by reasoning about the
code, and both had the same root cause: a schema field the prompt never
described. Given an uninstructed field, llama3.1:8b filled it with the
nearest salient token from the prompt - the finding id, or the location
line - and the result validated cleanly and reached a rendered report.

    Failure 1  EvidenceAssessment.interpretation was "SENT-001", so every
               finding's explanation was its own id. Observed on 3/3
               findings sampled, then on all 17 in a full run.
    Failure 2  ExploitAssessment.exploit_path was "/Users/.../tasks.py:13".
               Observed on 5/5 non-null exploit paths in a full run.

Neither was a model capability problem: both disappeared once the prompt
named the field and said what belongs in it. These tests hold the fix in
place from two directions - the prompts must keep carrying the explicit
anti-degeneration instructions, and the schema must keep rejecting the
degenerate values that were actually produced.

The prompt assertions are deliberately unit-level and offline: they
assert on the built prompt string, so they run in milliseconds and cannot
regress silently the way a live-only check would.
"""
import pytest
from pydantic import ValidationError

from sentinelai.ai.agents.critic import build_critic_prompt
from sentinelai.ai.agents.evidence_analyst import build_evidence_prompt
from sentinelai.ai.agents.exploit_remediation_analyst import build_exploit_remediation_prompt
from sentinelai.ai.agents.schemas import (
    CriticAssessment,
    EvidenceAssessment,
    ExploitAssessment,
    ExploitRemediationAssessment,
    RemediationPlan,
)
from sentinelai.ai.repository_context import RepositoryContext
from sentinelai.contracts import GroundingVerdict, ScannerFinding, Severity

# The exact degenerate values a live llama3.1:8b run produced.
OBSERVED_DEGENERATE_INTERPRETATIONS = [
    "SENT-001",
    "SENT-002: hardcoded_sql_expressions (B608)",
    "/Users/tanaya/Desktop/capstone/sentinelai-manual-test/app/tasks.py:13",
]
OBSERVED_DEGENERATE_EXPLOIT_PATHS = [
    "/Users/tanaya/Desktop/capstone/sentinelai-manual-test/app/tasks.py:13",
    "/Users/tanaya/Desktop/capstone/sentinelai-manual-test/app/db.py:7",
]


def _finding() -> ScannerFinding:
    return ScannerFinding(
        finding_id="SENT-001",
        scanner="bandit",
        category="hashlib",
        severity=Severity.HIGH,
        file="app/crypto_utils.py",
        line_start=6,
        rule_id="B324",
        message="Use of weak MD5 hash for security.",
        raw_evidence="return hashlib.md5(password.encode()).hexdigest()",
    )


def _evidence() -> EvidenceAssessment:
    return EvidenceAssessment(
        title="Weak MD5 hash used for passwords",
        observed_evidence=["hashlib.md5 is used to hash a password value."],
        interpretation="MD5 is fast and collision-prone, so hashed passwords can be recovered cheaply.",
        limitations=["Whether this hash is used for authentication is not shown."],
        groundedness=GroundingVerdict.PARTIALLY_SUPPORTED,
    )


def _assessment() -> ExploitRemediationAssessment:
    return ExploitRemediationAssessment(
        exploit=ExploitAssessment(
            exploit_path="An attacker who obtains the hash database recovers passwords by brute force.",
            impact="Account compromise across the application.",
            required_assumptions=["The attacker can read stored hashes."],
        ),
        remediation=RemediationPlan(remediation="Use bcrypt or Argon2 for password hashing."),
    )


def _evidence_prompt() -> str:
    return build_evidence_prompt(_finding(), RepositoryContext(summary="Repository 'demo'."), [])


def _exploit_prompt() -> str:
    return build_exploit_remediation_prompt(
        _finding(), _evidence(), RepositoryContext(summary="Repository 'demo'."), []
    )


def _critic_prompt() -> str:
    return build_critic_prompt(
        _finding(), _evidence(), _assessment(), RepositoryContext(summary="Repository 'demo'."), []
    )


# --- 1. interpretation cannot be a bare identifier (schema-enforced) -----------------------------------------------------


@pytest.mark.parametrize("degenerate", OBSERVED_DEGENERATE_INTERPRETATIONS)
def test_interpretation_rejects_the_values_a_live_run_actually_produced(degenerate):
    with pytest.raises(ValidationError):
        EvidenceAssessment(
            title="A title",
            interpretation=degenerate,
            groundedness=GroundingVerdict.SUPPORTED,
        )


@pytest.mark.parametrize(
    "degenerate",
    ["SENT-042", "B324", "hashlib", "app/crypto_utils.py:6", "sql-injection", "CWE-327"],
)
def test_interpretation_rejects_ids_locations_categories_and_rule_ids(degenerate):
    with pytest.raises(ValidationError):
        EvidenceAssessment(
            title="A title", interpretation=degenerate, groundedness=GroundingVerdict.SUPPORTED
        )


def test_interpretation_accepts_genuine_analysis():
    # Real output from the same model after the prompt fix - must keep validating.
    real = (
        "The use of MD5 hashing for password storage creates a security problem because MD5 is "
        "vulnerable to collisions, allowing an attacker to recover the original password."
    )
    a = EvidenceAssessment(
        title="Weak MD5 hash", interpretation=real, groundedness=GroundingVerdict.SUPPORTED
    )
    assert a.interpretation == real


def test_interpretation_error_names_the_field_and_the_offending_value():
    with pytest.raises(ValidationError) as excinfo:
        EvidenceAssessment(
            title="t", interpretation="SENT-001", groundedness=GroundingVerdict.SUPPORTED
        )

    message = str(excinfo.value)
    assert "interpretation" in message
    assert "SENT-001" in message


def test_the_guard_does_not_leak_into_other_fields():
    # Only `interpretation` is guarded; `title` is *supposed* to be a short label,
    # and short list entries remain legitimate.
    a = EvidenceAssessment(
        title="B324",
        interpretation="MD5 is unsuitable for password hashing because it is cheap to brute force.",
        observed_evidence=["hashlib.md5 is called."],
        groundedness=GroundingVerdict.SUPPORTED,
    )
    assert a.title == "B324"


# --- 2. exploit prompt demands a narrative, never a location ---------------------------------------------------------------


def test_exploit_prompt_requires_a_narrative_of_attacker_action():
    prompt = _exploit_prompt()

    assert "must be prose describing the steps by which an attacker would reach and abuse" in prompt
    assert "what they control, what they send, and what happens as a result" in prompt


def test_exploit_prompt_forbids_a_location_in_the_exploit_path():
    prompt = _exploit_prompt()

    assert "Do NOT put a file path, line number, or location here" in prompt


def test_exploit_prompt_still_permits_declining_with_null():
    # The anti-degeneration rule must not push the model into inventing a path.
    prompt = _exploit_prompt()

    assert "Set it to null unless the evidence supports a concrete path" in prompt
    assert "a guess stated as fact is worse than null" in prompt.lower()


def test_exploit_prompt_requires_impact_to_be_a_consequence():
    prompt = _exploit_prompt()

    assert "must state in prose what an attacker gains" in prompt
    assert "not a severity word and not a location" in prompt


def test_remediation_prompt_requires_prose_not_a_restated_rule():
    prompt = _exploit_prompt()

    assert "Not a location, not a restatement of the rule name" in prompt


# --- 3. critic prompt reviews exploit paths as narratives ---------------------------------------------------------------


def test_critic_prompt_treats_an_exploit_path_as_a_narrative():
    prompt = _critic_prompt()

    assert "An exploit path is a narrative" in prompt
    assert "describe attacker action and how data or control reaches the vulnerable code" in prompt


def test_critic_prompt_flags_a_location_as_unsupported():
    prompt = _critic_prompt()

    assert "merely a file path, a line number, a finding id, a category, or a rule id" in prompt
    assert "treat it as unsupported" in prompt
    assert "a location is not evidence of exploitability" in prompt


def test_critic_prompt_requires_claims_to_be_readable():
    prompt = _critic_prompt()

    assert "must restate the claim being judged, in prose" in prompt
    assert "Do not write a file path, line number, or bare identifier" in prompt


# --- 4. evidence prompt names the interpretation field explicitly -----------------------------------------------------------


def test_evidence_prompt_says_what_interpretation_must_contain():
    # The omission that caused failure 1: the field existed in the schema but the
    # prompt never described it.
    prompt = _evidence_prompt()

    assert "`interpretation` must be one to three sentences explaining WHY" in prompt
    assert "what weakness it creates and what could go wrong because of it" in prompt


def test_evidence_prompt_forbids_restating_identifiers_as_the_interpretation():
    prompt = _evidence_prompt()

    assert "Do NOT put the finding id, rule id, or category here" in prompt
    assert "leaves the report with no explanation at all" in prompt


@pytest.mark.parametrize(
    "field", ["title", "interpretation", "observed_evidence", "limitations", "groundedness"]
)
def test_evidence_prompt_gives_a_rule_for_every_schema_field(field):
    # The root cause of failure 1 was a field with no prompt rule. This makes that
    # class of omission fail a test rather than a live run.
    assert f"`{field}`" in _evidence_prompt()


@pytest.mark.parametrize(
    "field",
    [
        "exploit.exploit_path",
        "exploit.impact",
        "exploit.required_assumptions",
        "remediation.remediation",
        "remediation.patch_suggestion",
        "remediation.structured_patch",
        "remediation.validation_steps",
    ],
)
def test_exploit_prompt_gives_a_rule_for_every_schema_field(field):
    assert f"`{field}`" in _exploit_prompt()


@pytest.mark.parametrize("field", ["supported_claims", "unsupported_claims", "verdict"])
def test_critic_prompt_gives_a_rule_for_every_schema_field(field):
    assert f"`{field}`" in _critic_prompt()


# --- prompts still carry the shared evidence, unchanged ---------------------------------------------------------------------


def test_every_agent_prompt_shows_the_same_finding_evidence():
    # The critic can only review what the other agents saw; drift here would make
    # its verdict meaningless.
    for prompt in (_evidence_prompt(), _exploit_prompt(), _critic_prompt()):
        assert "Finding: SENT-001" in prompt
        assert "hashlib.md5(password.encode())" in prompt
