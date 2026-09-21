"""
Typed output contracts for the multi-agent triage pipeline.

One model per LLM call, not one per logical agent: the pipeline makes
exactly three generation requests per finding, and the second carries the
Exploit and Remediation analysts as two sections of a single
ExploitRemediationAssessment. Splitting those into two requests would
double the dominant cost of the pipeline for no additional grounding.

Every model here sets extra="forbid" and frozen=True, matching
ai/llm_response.py's already-established reasoning: this input is raw
model output, so an unexpected key is far more likely to mean the model
misunderstood the task than that the schema is stale, and rejecting it
immediately is safer than silently dropping it.

NonBlankStr is used for every user-facing string. A model that has
nothing to say is required to say so through the schema - by returning
null for an optional field, or an empty list - rather than by emitting
`""`, `"N/A"`, or `" "`, which would render as blank sections in a
report and read as missing analysis rather than as an absent finding.
Optional strings may be null, but must be non-blank when present.

The separation between observed facts and inference is carried in the
schema rather than left to prose, because it is the property this
pipeline exists to preserve:

    EvidenceAssessment.observed_evidence   what the supplied evidence shows
    EvidenceAssessment.limitations         what it cannot show
    ExploitAssessment.required_assumptions what must additionally hold
    CriticAssessment.unsupported_claims    what was asserted without support

Those four lists are deliberately distinct fields on distinct models. A
downstream consumer can therefore tell an observation from an assumption
without parsing English, and ai/agents/synthesizer.py labels each one
explicitly when it folds them into the report.

GroundingVerdict is imported from sentinelai.contracts rather than
defined here: it is part of the AIEnrichedFinding contract that the CLI
and every renderer consume, and contracts/ is a leaf that must not
import from ai/. Defining it here would invert that dependency.
"""
from typing import Annotated, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from sentinelai.contracts import GroundingVerdict, StructuredPatch

# Strips surrounding whitespace, then requires at least one character - so
# "   " is rejected rather than silently becoming an empty section.
NonBlankStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

_STRICT = ConfigDict(extra="forbid", frozen=True)

# Minimum words for a field that must read as prose rather than as an identifier.
# Not an arbitrary quality bar: it is calibrated against the degenerate values a
# live llama3.1:8b run actually produced for `interpretation` before the prompt
# named that field explicitly -
#     "SENT-001"                                    (1 word)
#     "SENT-002: hardcoded_sql_expressions (B608)"  (3 words)
#     "/Users/.../app/tasks.py:13"                  (1 word)
# - all of which validated cleanly under the original schema and reached a
# rendered report as the entire explanation. Real interpretations from the same
# model run 20-40 words, so this threshold sits well clear of legitimate output.
_MIN_PROSE_WORDS = 6


# A claim a reader must be able to act on. Same floor as _require_prose, applied
# per list entry rather than to a single field - an "unsupported claim" reading
# "app/db.py:7" or "B324" tells a reviewer nothing about what was overclaimed.
ProseClaim = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1)
]


def _require_prose(value: str, field_name: str) -> str:
    """Reject a value that is an identifier or location rather than a sentence.

    Deliberately narrow. This catches the demonstrated failure mode - a model
    echoing the nearest salient token instead of writing analysis - and nothing
    more. It cannot judge whether prose is *good*, and a fluent but shallow
    sentence passes. The prompt is what asks for quality; this is the floor that
    stops an unreadable report from being emitted as though it were analysis.
    """
    if len(value.split()) < _MIN_PROSE_WORDS:
        raise ValueError(
            f"{field_name} must be prose explaining the finding, not an identifier or location "
            f"(got {value!r}). Fewer than {_MIN_PROSE_WORDS} words is a restated id/rule/path, "
            "which renders as an empty explanation."
        )
    return value


class EvidenceAssessment(BaseModel):
    """Output of the Evidence Analyst - LLM call 1 of 3."""

    model_config = _STRICT

    title: NonBlankStr = Field(
        ...,
        description=(
            "Short label for this finding, grounded in the scanner's own category, rule id, and "
            "evidence. Not a free-form headline: the prompt constrains it to restate what the "
            "scanner reported, so the report's title never asserts more than the scanner did."
        ),
    )
    observed_evidence: list[NonBlankStr] = Field(
        default_factory=list,
        description="Facts read directly from the supplied evidence. Nothing inferred, assumed, or recalled.",
    )
    interpretation: NonBlankStr = Field(
        ..., description="Why the observed evidence is security-relevant."
    )
    limitations: list[NonBlankStr] = Field(
        default_factory=list,
        description="What cannot be determined from the supplied evidence alone.",
    )
    groundedness: GroundingVerdict = Field(
        ..., description="How well the interpretation is supported by the observed evidence."
    )

    @field_validator("interpretation")
    @classmethod
    def _interpretation_must_be_prose(cls, value: str) -> str:
        """Reject an interpretation that is merely the finding id, location, category, or rule id.

        `interpretation` becomes the explanation in every rendered report, so an
        identifier here produces a finding with no explanation at all. A live run
        did exactly that on every finding before the prompt named this field, and
        the schema accepted all of it - see _require_prose.
        """
        return _require_prose(value, "interpretation")


class ExploitAssessment(BaseModel):
    """Exploit section of LLM call 2 of 3."""

    model_config = _STRICT

    exploit_path: Optional[NonBlankStr] = Field(
        None,
        description=(
            "How the finding could be exploited, or null when the supplied evidence does not "
            "support a concrete path. Null is the correct answer for a finding whose "
            "reachability cannot be established from one file's evidence."
        ),
    )
    impact: Optional[NonBlankStr] = Field(
        None, description="What an attacker would gain if the exploit path holds, or null."
    )
    required_assumptions: list[NonBlankStr] = Field(
        default_factory=list,
        description=(
            "Conditions that must hold for the exploit path to be real but that the evidence does "
            "not establish - for example reachability from untrusted input. Kept separate from "
            "EvidenceAssessment.observed_evidence so an assumption is never mistaken for a fact."
        ),
    )


class RemediationPlan(BaseModel):
    """Remediation section of LLM call 2 of 3."""

    model_config = _STRICT

    remediation: NonBlankStr = Field(..., description="How to fix the finding.")
    patch_suggestion: Optional[NonBlankStr] = Field(
        None,
        description=(
            "Concrete code change, or null when one cannot responsibly be proposed from the "
            "supplied evidence. Can be a unified diff or replacement snippet."
        ),
    )
    validation_steps: list[NonBlankStr] = Field(
        default_factory=list, description="How to confirm the fix worked."
    )
    structured_patch: Optional[StructuredPatch] = Field(
        None,
        description=(
            "Machine-applicable form of the same fix, when the model can express one. Null "
            "otherwise, and null is the correct answer whenever a patch cannot be grounded in "
            "the evidence shown - the same bar patch_suggestion is already held to."
        ),
    )


class ExploitRemediationAssessment(BaseModel):
    """Envelope for LLM call 2 of 3, carrying both analysts' sections in one request."""

    model_config = _STRICT

    exploit: ExploitAssessment
    remediation: RemediationPlan


class CriticAssessment(BaseModel):
    """Output of the Critic - LLM call 3 of 3.

    A review signal, not a proof system: the critic re-reads the other agents'
    claims against the same evidence and reports which it can support. It cannot
    establish factual correctness, and its verdict is never converted into
    verification_status, which is produced deterministically and independently.
    """

    model_config = _STRICT

    # Two schema-level properties here are load-bearing for constrained decoding,
    # not cosmetic, and both were established by measurement against llama3.1:8b
    # on an identical prompt:
    #
    # 1. Field ORDER. Ollama generates fields in schema order and cannot revise an
    #    earlier field. With the claim lists declared first, the model committed to
    #    empty lists *before* choosing a verdict, then selected
    #    `partially_supported` - the unexplained-verdict shape
    #    _verdict_matches_claims rejects. Declaring `verdict` first makes it a
    #    commitment the claim lists are then written to justify.
    # 2. unsupported_claims is REQUIRED (Field(...), no default). A field with a
    #    default is omitted from the JSON Schema's `required` array, which lets a
    #    constrained decoder skip it entirely.
    #
    # Measured: claims-first + optional -> 0 unsupported claims; verdict-first +
    # required -> 2. Neither change alone was sufficient. Do not reorder these
    # fields, and do not give unsupported_claims a default.
    verdict: GroundingVerdict = Field(
        ..., description="Overall judgement of how well the narrative is grounded in the evidence."
    )
    unsupported_claims: list[ProseClaim] = Field(
        ...,
        description=(
            "Claims asserted without support in the supplied evidence, or supported only under "
            "assumptions. Distinct from EvidenceAssessment.observed_evidence by construction. "
            "Required to be non-empty for a partially_supported or insufficient_evidence verdict - "
            "see _verdict_matches_claims."
        ),
    )
    supported_claims: list[NonBlankStr] = Field(
        default_factory=list, description="Claims the supplied evidence supports."
    )

    @field_validator("unsupported_claims")
    @classmethod
    def _claims_must_be_readable(cls, values: list[str]) -> list[str]:
        """Each unsupported claim must restate what was claimed, not point at a location."""
        for value in values:
            _require_prose(value, "unsupported_claims entry")
        return values

    @model_validator(mode="after")
    def _verdict_matches_claims(self) -> "CriticAssessment":
        """Enforce that a non-supported verdict says what is unsupported.

        A `partially_supported` or `insufficient_evidence` verdict with an empty
        unsupported_claims list is internally meaningless: it asserts the analysis
        overreaches without naming anything that overreaches, and it renders as a
        verdict badge with no explanation beneath it. A live run produced exactly
        that on 17/17 findings - every verdict was partially_supported or
        insufficient_evidence, and every claim list was empty - which is why this
        is enforced by the contract rather than left to the prompt.

        supported_claims is deliberately NOT required to be non-empty. A critic may
        legitimately find nothing separately worth listing as supported; that is not
        a contradiction the way an unexplained partial verdict is.

        A `supported` verdict must have no unsupported claims, which is the same
        invariant read from the other direction: if something is unsupported, the
        verdict is not `supported`.
        """
        if self.verdict is GroundingVerdict.SUPPORTED:
            if self.unsupported_claims:
                raise ValueError(
                    "verdict 'supported' cannot list unsupported_claims - use "
                    "'partially_supported' if any claim is unsupported"
                )
            return self

        if not self.unsupported_claims:
            raise ValueError(
                f"verdict '{self.verdict.value}' requires at least one entry in "
                "unsupported_claims explaining what the evidence does not support; an "
                "unexplained non-supported verdict is meaningless in a report"
            )
        return self
