"""
Schema acceptance and rejection for every multi-agent output contract.

These schemas are the pipeline's only defence against a small local model
returning something plausible-looking but wrong, so the rejection cases
matter more than the acceptance ones. Each model forbids extra keys,
requires non-blank strings where a reader will see them, and keeps
observations, assumptions, and unsupported claims in separate fields so
the distinction survives into the report.
"""
import pytest
from pydantic import ValidationError

from sentinelai.ai.agents.schemas import (
    CriticAssessment,
    EvidenceAssessment,
    ExploitAssessment,
    ExploitRemediationAssessment,
    RemediationPlan,
)
from sentinelai.contracts import GroundingVerdict

# --- EvidenceAssessment ----------------------------------------------------------------------------------------------


def test_evidence_accepts_a_well_formed_assessment():
    a = EvidenceAssessment(
        title="sql-injection in db.py",
        observed_evidence=["A query is built by concatenation."],
        interpretation="Concatenating untrusted input into a query permits statement manipulation.",
        limitations=["Reachability from untrusted input is not shown."],
        groundedness=GroundingVerdict.PARTIALLY_SUPPORTED,
    )
    assert a.groundedness is GroundingVerdict.PARTIALLY_SUPPORTED


def test_evidence_lists_default_to_empty():
    a = EvidenceAssessment(title="t", interpretation="Concatenating untrusted input into a query permits statement manipulation.", groundedness=GroundingVerdict.SUPPORTED)
    assert a.observed_evidence == []
    assert a.limitations == []


@pytest.mark.parametrize("blank", ["", "   ", "\n\t"])
def test_evidence_rejects_blank_required_strings(blank):
    with pytest.raises(ValidationError):
        EvidenceAssessment(title=blank, interpretation="Concatenating untrusted input into a query permits statement manipulation.", groundedness=GroundingVerdict.SUPPORTED)
    with pytest.raises(ValidationError):
        EvidenceAssessment(title="t", interpretation=blank, groundedness=GroundingVerdict.SUPPORTED)


def test_evidence_rejects_blank_list_entries():
    with pytest.raises(ValidationError):
        EvidenceAssessment(
            title="t", interpretation="Concatenating untrusted input into a query permits statement manipulation.", observed_evidence=["  "], groundedness=GroundingVerdict.SUPPORTED
        )


def test_evidence_rejects_unknown_fields():
    # A model that invents a field has misunderstood the task; that must fail loudly
    # rather than be silently dropped.
    with pytest.raises(ValidationError):
        EvidenceAssessment(
            title="t",
            interpretation="i",
            groundedness=GroundingVerdict.SUPPORTED,
            confidence=0.9,
        )


def test_evidence_rejects_an_invalid_groundedness_value():
    with pytest.raises(ValidationError):
        EvidenceAssessment(title="t", interpretation="Concatenating untrusted input into a query permits statement manipulation.", groundedness="probably fine")


def test_evidence_separates_observations_from_limitations():
    # The contract-level distinction the whole pipeline depends on: what the
    # evidence shows is a different field from what it cannot show.
    a = EvidenceAssessment(
        title="t",
        interpretation="Concatenating untrusted input into a query permits statement manipulation.",
        observed_evidence=["concatenation is present"],
        limitations=["attacker control is unknown"],
        groundedness=GroundingVerdict.PARTIALLY_SUPPORTED,
    )
    assert "attacker control is unknown" not in a.observed_evidence
    assert "concatenation is present" not in a.limitations


def test_evidence_is_immutable():
    a = EvidenceAssessment(title="t", interpretation="Concatenating untrusted input into a query permits statement manipulation.", groundedness=GroundingVerdict.SUPPORTED)
    with pytest.raises(ValidationError):
        a.title = "changed"


# --- ExploitAssessment -----------------------------------------------------------------------------------------------


def test_exploit_allows_null_path_and_impact():
    # Declining is a schema-legal answer; it is how the model avoids inventing one.
    e = ExploitAssessment()
    assert e.exploit_path is None
    assert e.impact is None
    assert e.required_assumptions == []


def test_exploit_rejects_blank_path_when_present():
    # Null means "none established"; "" would render as an empty report section.
    with pytest.raises(ValidationError):
        ExploitAssessment(exploit_path="   ")


def test_exploit_keeps_assumptions_separate_from_the_path():
    e = ExploitAssessment(
        exploit_path="A crafted value alters the query.",
        required_assumptions=["The value is attacker-controlled."],
    )
    assert e.required_assumptions == ["The value is attacker-controlled."]
    assert "attacker-controlled" not in e.exploit_path


def test_exploit_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        ExploitAssessment(exploit_path="p", likelihood="high")


# --- RemediationPlan -------------------------------------------------------------------------------------------------


def test_remediation_allows_null_patch_suggestion():
    plan = RemediationPlan(remediation="Use parameterized queries.")
    assert plan.patch_suggestion is None
    assert plan.validation_steps == []


def test_remediation_requires_a_non_blank_remediation():
    with pytest.raises(ValidationError):
        RemediationPlan(remediation="   ")


def test_remediation_rejects_blank_patch_when_present():
    with pytest.raises(ValidationError):
        RemediationPlan(remediation="fix it", patch_suggestion="  ")


# --- ExploitRemediationAssessment (the combined call-2 envelope) -------------------------------------------------------


def test_combined_envelope_accepts_both_sections():
    combined = ExploitRemediationAssessment(
        exploit=ExploitAssessment(exploit_path="p"),
        remediation=RemediationPlan(remediation="r"),
    )
    assert combined.exploit.exploit_path == "p"
    assert combined.remediation.remediation == "r"


def test_combined_envelope_requires_both_sections():
    with pytest.raises(ValidationError):
        ExploitRemediationAssessment(exploit=ExploitAssessment())


def test_combined_envelope_validates_nested_sections():
    with pytest.raises(ValidationError):
        ExploitRemediationAssessment.model_validate(
            {"exploit": {"exploit_path": "p"}, "remediation": {"remediation": "  "}}
        )


def test_combined_envelope_exposes_a_json_schema_with_nested_defs():
    # Sent to Ollama in the `format` field; nested models must survive the round trip.
    schema = ExploitRemediationAssessment.model_json_schema()
    assert "$defs" in schema
    assert {"exploit", "remediation"} <= set(schema["properties"])


# --- CriticAssessment ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "verdict",
    [
        GroundingVerdict.SUPPORTED,
        GroundingVerdict.PARTIALLY_SUPPORTED,
        GroundingVerdict.INSUFFICIENT_EVIDENCE,
    ],
)
def test_critic_supports_every_verdict(verdict):
    # Non-supported verdicts must name what is unsupported - see the invariant
    # tests at the end of this module.
    claims = (
        []
        if verdict is GroundingVerdict.SUPPORTED
        else ["The analysis assumes attacker control that the evidence does not establish."]
    )
    assert CriticAssessment(verdict=verdict, unsupported_claims=claims).verdict is verdict


def test_critic_supported_claims_default_to_empty():
    # supported_claims is optional; unsupported_claims is required so a constrained
    # decoder cannot skip it (see the field-order/required note on the model).
    c = CriticAssessment(verdict=GroundingVerdict.SUPPORTED, unsupported_claims=[])
    assert c.supported_claims == []


def test_critic_requires_unsupported_claims_to_be_stated_explicitly():
    with pytest.raises(ValidationError):
        CriticAssessment(verdict=GroundingVerdict.SUPPORTED)


def test_critic_keeps_supported_and_unsupported_claims_separate():
    c = CriticAssessment(
        supported_claims=["concatenation is present"],
        unsupported_claims=["The claim that the input is attacker-controlled is not evidenced."],
        verdict=GroundingVerdict.PARTIALLY_SUPPORTED,
    )
    assert c.supported_claims != c.unsupported_claims


def test_critic_rejects_an_invalid_verdict():
    with pytest.raises(ValidationError):
        CriticAssessment(verdict="looks_fine", unsupported_claims=[])


def test_critic_requires_a_verdict():
    with pytest.raises(ValidationError):
        CriticAssessment(supported_claims=["a"], unsupported_claims=[])


# --- every agent schema is usable as an Ollama `format` payload ---------------------------------------------------------


@pytest.mark.parametrize(
    "model", [EvidenceAssessment, ExploitRemediationAssessment, CriticAssessment]
)
def test_every_agent_schema_exposes_a_json_schema(model):
    import json

    schema = model.model_json_schema()
    assert schema["type"] == "object"
    assert schema.get("additionalProperties") is False  # extra="forbid" survives into the schema
    json.dumps(schema)  # must be JSON-serializable to travel in the request body


# --- critic verdict/claim-list invariant --------------------------------------------------------------------------------
#
# A live run produced a partially_supported or insufficient_evidence verdict on
# 17/17 findings with an empty unsupported_claims list every time, which renders
# as a verdict badge with no explanation under it. The invariant is enforced by
# the model validator so that shape cannot reach synthesis.

CLAIM = "The analysis asserts the input is attacker-controlled, which the evidence does not show."


def test_supported_verdict_may_have_no_claims_at_all():
    c = CriticAssessment(verdict=GroundingVerdict.SUPPORTED, unsupported_claims=[])
    assert c.unsupported_claims == []


def test_supported_verdict_may_list_supported_claims():
    c = CriticAssessment(
        verdict=GroundingVerdict.SUPPORTED,
        unsupported_claims=[],
        supported_claims=["The query is concatenated."],
    )
    assert c.supported_claims


def test_supported_verdict_rejects_unsupported_claims():
    # If something is unsupported the verdict is not 'supported'.
    with pytest.raises(ValidationError):
        CriticAssessment(verdict=GroundingVerdict.SUPPORTED, unsupported_claims=[CLAIM])


@pytest.mark.parametrize(
    "verdict", [GroundingVerdict.PARTIALLY_SUPPORTED, GroundingVerdict.INSUFFICIENT_EVIDENCE]
)
def test_non_supported_verdict_requires_an_unsupported_claim(verdict):
    with pytest.raises(ValidationError) as excinfo:
        CriticAssessment(verdict=verdict, unsupported_claims=[])

    assert "unsupported_claims" in str(excinfo.value)


@pytest.mark.parametrize(
    "verdict", [GroundingVerdict.PARTIALLY_SUPPORTED, GroundingVerdict.INSUFFICIENT_EVIDENCE]
)
def test_non_supported_verdict_accepts_a_substantive_claim(verdict):
    c = CriticAssessment(verdict=verdict, unsupported_claims=[CLAIM])
    assert c.unsupported_claims == [CLAIM]


@pytest.mark.parametrize(
    "verdict", [GroundingVerdict.PARTIALLY_SUPPORTED, GroundingVerdict.INSUFFICIENT_EVIDENCE]
)
def test_supported_claims_are_never_required(verdict):
    # A critic may find nothing separately worth listing as supported.
    c = CriticAssessment(verdict=verdict, unsupported_claims=[CLAIM])
    assert c.supported_claims == []


@pytest.mark.parametrize(
    "degenerate",
    ["app/db.py:7", "B324", "SENT-001", "hashlib", "CWE-327", "high", "/Users/x/app/tasks.py:13"],
)
def test_unsupported_claims_reject_identifiers_and_locations(degenerate):
    with pytest.raises(ValidationError):
        CriticAssessment(
            verdict=GroundingVerdict.PARTIALLY_SUPPORTED, unsupported_claims=[degenerate]
        )


def test_critic_declares_verdict_before_the_claim_lists():
    """Field order is a functional requirement of constrained decoding, not style.

    Ollama generates fields in schema order and cannot revise an earlier field.
    With the claim lists first, llama3.1:8b committed to empty lists before
    choosing a verdict and then selected partially_supported - the unexplained
    verdict the model validator rejects. Measured on an identical prompt: 0
    unsupported claims with claims-first, 2 with verdict-first.
    """
    order = list(CriticAssessment.model_json_schema()["properties"])

    assert order.index("verdict") < order.index("unsupported_claims")
    assert order.index("verdict") < order.index("supported_claims")
