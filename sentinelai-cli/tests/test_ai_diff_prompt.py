"""
Prompt instructions for reproducible unified diffs (Phase 1.2).

These assert the *prompt*, not model behaviour. A local model's output varies
run to run and cannot be asserted in a unit test; what can be asserted is that
every instruction the diff depends on is actually present in the text sent to
it. Each of the requirements below caused a specific, foreseeable failure mode
if omitted - a diff wrapped in markdown fences, a hunk header numbered from the
excerpt instead of the file, display markers leaking into the diff body - so
each is guarded individually rather than by one assertion over the whole block.

The separation this phase must not blur is also asserted here: generation is
the only step that involves a model, and the prompt is the last point at which
one is involved at all.
"""
import pytest

from sentinelai.ai.agents.exploit_remediation_analyst import (
    build_exploit_remediation_prompt,
)
from sentinelai.ai.agents.schemas import EvidenceAssessment
from sentinelai.ai.repository_context import RepositoryContext
from sentinelai.contracts import ScannerFinding, Severity

from agent_fakes import VALID_EVIDENCE


def _prompt() -> str:
    finding = ScannerFinding(
        finding_id="SENT-001",
        scanner="semgrep",
        category="sql-injection",
        severity=Severity.CRITICAL,
        file="app/db.py",
        line_start=7,
        rule_id="semgrep.python.sql-injection",
        message="User input concatenated into a SQL query.",
        raw_evidence='query = f"SELECT * FROM users WHERE name = \'{name}\'"',
        cwe="CWE-89",
    )
    return build_exploit_remediation_prompt(
        finding,
        EvidenceAssessment.model_validate(VALID_EVIDENCE),
        RepositoryContext(summary="Repository 'demo'."),
        [],
    )


# --- the required instructions ----------------------------------------------------------------------------------------


def test_unified_diff_shape_is_shown_literally():
    """Describing the format in prose is weaker than showing it; both markers must appear."""
    prompt = _prompt()
    assert "--- a/<path>" in prompt
    assert "+++ b/<path>" in prompt
    assert "@@ -<old_start>,<old_count> +<new_start>,<new_count> @@" in prompt


def test_surrounding_context_is_required():
    assert "at least three unchanged context lines" in _prompt()


def test_indentation_must_be_preserved_exactly():
    prompt = _prompt()
    assert "Reproduce indentation exactly" in prompt
    assert "tabs or spaces" in prompt


def test_inventing_surrounding_code_is_forbidden():
    prompt = _prompt()
    assert "Never invent surrounding code" in prompt
    assert "set `structured_patch` to null rather than guessing" in prompt


def test_changes_are_limited_to_what_the_evidence_supports():
    prompt = _prompt()
    assert "Change only what the evidence and the shown code support" in prompt
    assert "Do not reformat, reorder, rename" in prompt


def test_markdown_fences_are_forbidden():
    """A fenced diff is the single most common way an LLM makes a patch unapplicable."""
    assert "No markdown fences" in _prompt()


def test_commentary_inside_the_diff_is_forbidden():
    prompt = _prompt()
    assert "no explanatory comments inserted into the diff body" in prompt
    assert "Explanation belongs in `remediation.remediation`" in prompt


def test_output_is_required_to_be_deterministic():
    assert "smallest edit that fixes the finding, so the same input yields the same diff" in _prompt()


def test_line_numbers_are_absolute_not_excerpt_offsets():
    """The Source context block is an excerpt; a hunk numbered from it would not apply."""
    assert "absolute positions in the file, not offsets into the excerpt" in _prompt()


def test_display_markers_must_be_stripped():
    """'> ' and 'NNN | ' are how evidence is rendered, not file content."""
    prompt = _prompt()
    assert "line-number prefix" in prompt
    assert "must never appear inside the diff" in prompt


# --- schema coverage and backward compatibility -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field",
    [
        "remediation.remediation",
        "remediation.patch_suggestion",
        "remediation.structured_patch",
        "remediation.validation_steps",
        "structured_patch.file",
        "structured_patch.start_line",
        "structured_patch.replacement",
    ],
)
def test_every_populated_field_has_a_rule(field):
    assert f"`{field}`" in _prompt()


def test_null_structured_patch_is_explicitly_allowed():
    """Backward compatibility: a model that declines must have a schema-legal way to say so."""
    prompt = _prompt()
    assert "Set it to null otherwise" in prompt
    assert "a null patch is the correct, expected answer rather than a failure" in prompt


def test_patch_suggestion_is_retained_and_tied_to_the_structured_form():
    prompt = _prompt()
    assert "`remediation.patch_suggestion` is the human-readable form" in prompt
    assert "both must describe the SAME fix" in prompt


def test_structured_patch_is_conditional_on_source_context_being_present():
    assert "only when a `Source context` section appears above" in _prompt()


# --- architecture -------------------------------------------------------------------------------------------------------


def test_prompt_still_carries_the_prior_evidence_assessment():
    """The stage dependency is unchanged: this agent still works from call 1's validated output."""
    assert "Prior evidence assessment:" in _prompt()


def test_no_validation_or_application_is_requested_of_the_model():
    """Everything after generation is deterministic; the prompt must not ask the model to do it."""
    prompt = _prompt().lower()
    for forbidden in ("apply the patch", "write the file", "run the tests yourself", "verify the patch applies"):
        assert forbidden not in prompt
