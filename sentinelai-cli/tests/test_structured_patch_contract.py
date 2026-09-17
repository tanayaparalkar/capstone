"""
The StructuredPatch data contract (Phase 1.1).

This phase adds a shape and nothing else. There is deliberately no diff
parsing, no applicability check, and no application here - those are
deterministic steps that belong to later phases, and the tests below assert
the *absence* of behaviour as carefully as its presence, because the value of
this contract is that it is the clean handoff point between what a model
generates and what deterministic code consumes.

Backward compatibility is the property most at risk and is tested from both
directions: a report written before this field existed must still load, and a
report written with it must round-trip through JSON unchanged.
"""
import json

import pytest
from pydantic import ValidationError

from sentinelai.ai.agents.schemas import ExploitRemediationAssessment, RemediationPlan
from sentinelai.ai.agents.synthesizer import build_verifier_input, synthesize
from sentinelai.contracts import (
    AIEnrichedFinding,
    ConfidenceLabel,
    ScannerFinding,
    Severity,
    StructuredPatch,
    VerificationStatus,
)
from sentinelai.reporting import to_json
from sentinelai.reporting.loader import load_scan_result
from sentinelai.statistics import calculate_statistics
from sentinelai.reporting.models import REPORT_SCHEMA_VERSION

_DIFF = (
    "--- a/app/db.py\n"
    "+++ b/app/db.py\n"
    "@@ -5,3 +5,3 @@\n"
    "-    cursor.execute(f\"SELECT * FROM users WHERE name = '{name}'\")\n"
    "+    cursor.execute(\"SELECT * FROM users WHERE name = %s\", (name,))\n"
)


def _patch(**overrides) -> StructuredPatch:
    defaults = dict(diff=_DIFF, file="app/db.py")
    defaults.update(overrides)
    return StructuredPatch(**defaults)


# --- the model itself -------------------------------------------------------------------------------------------------


def test_diff_and_file_are_required():
    """A patch naming neither the change nor its target is not applicable by any means."""
    with pytest.raises(ValidationError):
        StructuredPatch(file="app/db.py")
    with pytest.raises(ValidationError):
        StructuredPatch(diff=_DIFF)


def test_line_span_and_replacement_are_optional():
    """A unified diff already encodes its own ranges; the redundant view may be absent."""
    patch = _patch()
    assert patch.start_line is None
    assert patch.end_line is None
    assert patch.replacement is None


def test_optional_fields_are_carried_when_supplied():
    patch = _patch(start_line=5, end_line=7, replacement="cursor.execute(sql, (name,))")
    assert (patch.start_line, patch.end_line) == (5, 7)
    assert patch.replacement == "cursor.execute(sql, (name,))"


def test_model_is_frozen():
    with pytest.raises(ValidationError):
        _patch().diff = "mutated"


def test_unexpected_keys_are_rejected():
    """extra='forbid', matching _STRICT: this is raw model output."""
    with pytest.raises(ValidationError):
        StructuredPatch(diff=_DIFF, file="app/db.py", applied=True)


@pytest.mark.parametrize("blank", ["", "   ", "\n"])
def test_blank_strings_are_rejected(blank):
    """NonBlankStr semantics: a model with nothing to say must use null, not ''."""
    with pytest.raises(ValidationError):
        StructuredPatch(diff=blank, file="app/db.py")
    with pytest.raises(ValidationError):
        StructuredPatch(diff=_DIFF, file=blank)


@pytest.mark.parametrize("bad_line", [0, -1])
def test_line_numbers_below_one_are_rejected(bad_line):
    """ge=1, matching ScannerFinding.line_start."""
    with pytest.raises(ValidationError):
        _patch(start_line=bad_line)


def test_no_cross_field_validation_in_this_phase():
    """end_line < start_line is accepted: consistency checking is deterministic work for a later phase."""
    assert _patch(start_line=9, end_line=2).end_line == 2


# --- backward compatibility -------------------------------------------------------------------------------------------


def test_remediation_plan_defaults_to_no_structured_patch():
    """The field is additive; an agent response that omits it stays valid."""
    plan = RemediationPlan(remediation="Use a parameterized query API.")
    assert plan.structured_patch is None
    assert plan.patch_suggestion is None


def test_patch_suggestion_is_retained_alongside_the_structured_form():
    """Both coexist: prose for a human, structure for a machine."""
    plan = RemediationPlan(
        remediation="Use a parameterized query API.",
        patch_suggestion="cursor.execute(sql, (name,))",
        structured_patch=_patch(),
    )
    assert plan.patch_suggestion == "cursor.execute(sql, (name,))"
    assert plan.structured_patch.file == "app/db.py"


def test_ai_enriched_finding_defaults_to_none():
    finding = AIEnrichedFinding(
        finding_id="SENT-001",
        title="t",
        severity=Severity.HIGH,
        explanation="e",
        remediation="r",
        confidence_score=0.5,
        confidence_label=ConfidenceLabel.MEDIUM,
        verification_status=VerificationStatus.UNVERIFIED,
    )
    assert finding.structured_patch is None


def test_report_schema_version_is_not_bumped():
    """loader.py compares with strict equality - a bump would orphan every saved report."""
    assert REPORT_SCHEMA_VERSION == "1.0"


# --- serialization ----------------------------------------------------------------------------------------------------


def _scan_result(ai_findings):
    from datetime import datetime, timezone

    from sentinelai.contracts import RepositoryInfo, ScanMetadata, ScanMode, ScanResult

    scanner_finding = ScannerFinding(
        finding_id="SENT-001",
        scanner="semgrep",
        category="sql-injection",
        severity=Severity.CRITICAL,
        file="app/db.py",
        line_start=5,
        rule_id="semgrep.python.sql-injection",
        message="m",
    )
    return ScanResult(
        repository=RepositoryInfo(name="demo", path="/tmp/demo"),
        metadata=ScanMetadata(
            timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
            mode=ScanMode.STANDARD,
            duration_seconds=1.0,
        ),
        scanner_findings=[scanner_finding],
        ai_findings=ai_findings,
    )


def _to_json(result) -> str:
    """Serialize exactly as the CLI does - statistics recomputed from the result."""
    return to_json(result, calculate_statistics(result))


def _enriched(**overrides) -> AIEnrichedFinding:
    defaults = dict(
        finding_id="SENT-001",
        title="SQL injection in db.py",
        severity=Severity.CRITICAL,
        explanation="e",
        remediation="r",
        patch_suggestion="cursor.execute(sql, (name,))",
        confidence_score=0.8,
        confidence_label=ConfidenceLabel.HIGH,
        verification_status=VerificationStatus.VERIFIED,
    )
    defaults.update(overrides)
    return AIEnrichedFinding(**defaults)


def test_structured_patch_round_trips_through_a_saved_report(tmp_path):
    ai = _enriched(structured_patch=_patch(start_line=5, end_line=5))
    path = tmp_path / "report.json"
    path.write_text(_to_json(_scan_result([ai])), encoding="utf-8")

    loaded, _ = load_scan_result(path)

    assert loaded.ai_findings[0].structured_patch == ai.structured_patch
    assert loaded.ai_findings[0].patch_suggestion == ai.patch_suggestion


def test_a_report_written_before_this_field_existed_still_loads(tmp_path):
    """The exact backward-compatibility guarantee: strip the key, it must still load."""
    path = tmp_path / "legacy.json"
    data = json.loads(_to_json(_scan_result([_enriched()])))
    for entry in data["findings"]["ai_enriched"]:
        entry.pop("structured_patch", None)
    assert "structured_patch" not in data["findings"]["ai_enriched"][0]
    path.write_text(json.dumps(data), encoding="utf-8")

    loaded, _ = load_scan_result(path)

    assert loaded.ai_findings[0].structured_patch is None
    assert loaded.ai_findings[0].patch_suggestion == "cursor.execute(sql, (name,))"


def test_serialized_json_carries_null_when_absent(tmp_path):
    """Existing JSON stays shape-compatible: the key appears, valued null."""
    data = json.loads(_to_json(_scan_result([_enriched()])))
    assert data["findings"]["ai_enriched"][0]["structured_patch"] is None


# --- deterministic/LLM separation -------------------------------------------------------------------------------------


def _assessment(structured_patch=None) -> ExploitRemediationAssessment:
    return ExploitRemediationAssessment.model_validate(
        {
            "exploit": {"exploit_path": None, "impact": None, "required_assumptions": []},
            "remediation": {
                "remediation": "Use a parameterized query API.",
                "patch_suggestion": "cursor.execute(sql, (name,))",
                "validation_steps": [],
                "structured_patch": structured_patch.model_dump() if structured_patch else None,
            },
        }
    )


def _synthesize(assessment):
    from sentinelai.ai.agents.schemas import CriticAssessment, EvidenceAssessment
    from sentinelai.ai.repository_context import RepositoryContext

    evidence = EvidenceAssessment.model_validate(
        {
            "title": "SQL injection",
            "observed_evidence": [],
            "interpretation": "A query string is built by concatenating a parameter value.",
            "limitations": [],
            "groundedness": "supported",
        }
    )
    critic = CriticAssessment(verdict="supported", unsupported_claims=[], supported_claims=[])
    finding = ScannerFinding(
        finding_id="SENT-001",
        scanner="semgrep",
        category="sql-injection",
        severity=Severity.CRITICAL,
        file="app/db.py",
        line_start=5,
        rule_id="r",
        message="m",
    )
    return synthesize(
        finding=finding,
        evidence=evidence,
        assessment=assessment,
        critic=critic,
        retrieved_chunks=[],
        confidence_score=0.8,
        confidence_label=ConfidenceLabel.HIGH,
        verification_status=VerificationStatus.VERIFIED,
        repository_context=RepositoryContext(summary="demo"),
    )


def test_synthesizer_passes_the_patch_through_unchanged():
    """Deterministic pass-through: the object is carried, not rebuilt or re-derived."""
    patch = _patch(start_line=5)
    result = _synthesize(_assessment(patch))
    assert result.structured_patch == patch


def test_synthesizer_carries_none_when_the_model_supplied_none():
    assert _synthesize(_assessment(None)).structured_patch is None


def test_verifier_input_is_unchanged_by_this_field():
    """ai/verifier.py is deterministic and its input shape must not shift under it."""
    from sentinelai.ai.agents.schemas import EvidenceAssessment

    evidence = EvidenceAssessment.model_validate(
        {
            "title": "SQL injection",
            "observed_evidence": [],
            "interpretation": "A query string is built by concatenating a parameter value.",
            "limitations": [],
            "groundedness": "supported",
        }
    )
    with_patch = build_verifier_input(evidence, _assessment(_patch()))
    without = build_verifier_input(evidence, _assessment(None))

    assert with_patch == without
    assert not hasattr(with_patch, "structured_patch")
