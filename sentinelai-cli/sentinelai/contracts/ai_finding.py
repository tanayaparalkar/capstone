"""
AI-enriched finding contract.

This is Tanaya's RAG/agentic-reasoning layer's structured output: it takes
one or more ScannerFinding objects plus repository context and produces an
explanation, exploit narrative, remediation, and a confidence score. Viraj's
CLI only ever consumes this shape - it does not calculate confidence, run
verification, or assume anything about how the underlying agents work.

`finding_id` identifies the primary ScannerFinding this enrichment is
about; `related_findings` carries the finding_ids of any other findings
involved in the same multi-step exploit chain.
"""
from enum import Enum
from typing import Annotated, Optional

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from .common import Severity


class ConfidenceLabel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class VerificationStatus(str, Enum):
    UNVERIFIED = "unverified"
    VERIFIED = "verified"
    REJECTED = "rejected"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class GroundingVerdict(str, Enum):
    """How well a critic agent judged the generated narrative to be supported by evidence.

    Deliberately a separate axis from VerificationStatus, never a replacement for
    it. VerificationStatus is produced by ai/verifier.py, a deterministic,
    network-free check that the generated text is topically consistent with the
    scanner's category. GroundingVerdict is produced by an LLM critic reviewing
    the other agents' claims against the supplied evidence. They answer different
    questions by different means, they can legitimately disagree, and neither is
    derived from the other - see ai/agents/synthesizer.py, which is forbidden from
    converting one into the other.

    This is a review signal, not calibrated verification and not a proof of
    factual correctness.
    """

    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


# Same definition as ai/agents/schemas.py's NonBlankStr and _STRICT, written out
# here rather than imported. StructuredPatch is part of the AIEnrichedFinding
# contract that the CLI and every renderer consume, so it has to live in
# contracts/ - a leaf that must not import from ai/ - for the same reason
# GroundingVerdict does. Importing ai/'s copies would invert that dependency;
# two duplicated lines is the cheaper price. Keep the two definitions identical.
_NonBlankStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
_STRICT_PATCH = ConfigDict(extra="forbid", frozen=True)


class StructuredPatch(BaseModel):
    """A machine-applicable form of a proposed fix. A data contract only.

    Nothing in this phase parses, validates, or applies one. The separation the
    rest of this package keeps between generated and deterministic work applies
    here in full: a model may *produce* this shape, and every step afterwards -
    parsing the diff, checking it applies, backing the file up, writing it,
    rolling back - is deterministic code that makes no model call. This class is
    the handoff point between those two halves, which is why it carries data and
    no behaviour.

    `diff` and `file` are required because a patch that names neither what to
    change nor where is not applicable by any means. The remaining three are
    optional: a unified diff already encodes its own line ranges in its hunk
    headers, so start_line/end_line/replacement are a redundant, easier-to-
    consume view for a consumer that would rather not parse a diff - present
    when the model can state them, absent when it cannot. Their absence is not
    an error, and no consistency check between them and `diff` exists yet.

    frozen/extra="forbid" match ai/agents/schemas.py's _STRICT rather than
    AIEnrichedFinding's own lenient config: this value is raw model output, so
    an unexpected key means the model misunderstood the task. The leniency
    AIEnrichedFinding needs is for *report* compatibility across versions, and
    no report has ever contained this field, so there is no legacy shape to
    tolerate.
    """

    model_config = _STRICT_PATCH

    diff: _NonBlankStr = Field(
        ...,
        description=(
            "Unified diff text for this fix. Advisory in this phase - nothing parses or "
            "applies it. Required: a structured patch with no diff carries no change."
        ),
    )
    file: _NonBlankStr = Field(
        ...,
        description=(
            "Path of the file the diff applies to. Named `file` to match ScannerFinding.file, "
            "so both sides of a finding use one spelling for the same concept."
        ),
    )
    start_line: Optional[int] = Field(
        None,
        ge=1,
        description=(
            "First line the change affects, if the model states it. ge=1 matches "
            "ScannerFinding.line_start; no relationship to `end_line` or to the diff's own "
            "hunk headers is enforced in this phase."
        ),
    )
    end_line: Optional[int] = Field(
        None, ge=1, description="Last line the change affects, if the model states it."
    )
    replacement: Optional[_NonBlankStr] = Field(
        None,
        description=(
            "The replacement block as literal source text, for a consumer that prefers a "
            "span-and-text edit over a diff. Null when the model supplies only a diff."
        ),
    )


class AIEnrichedFinding(BaseModel):
    finding_id: str = Field(..., description="finding_id of the primary ScannerFinding this enrichment is about")
    title: str = Field(..., description="Short human-readable title for the finding")
    severity: Severity
    scanner_sources: list[str] = Field(
        default_factory=list, description="Scanners that contributed evidence, e.g. ['semgrep', 'bandit']"
    )
    evidence: Optional[str] = Field(None, description="Evidence excerpt supporting the explanation")
    repository_context: Optional[str] = Field(None, description="Relevant repository context used during reasoning")
    explanation: str = Field(..., description="LLM-generated explanation of the vulnerability")
    exploit_path: Optional[str] = Field(None, description="Narrative of how this finding could be exploited, if applicable")
    impact: Optional[str] = Field(None, description="Description of the potential impact if exploited")
    remediation: str = Field(..., description="Suggested fix")
    patch_suggestion: Optional[str] = Field(None, description="Concrete patch/diff suggestion, if generated")
    structured_patch: Optional[StructuredPatch] = Field(
        None,
        description=(
            "Machine-applicable form of the same fix patch_suggestion describes in prose. "
            "Optional and defaulted so every report written before this field existed still "
            "loads unchanged - the same additive pattern used for correlation_id and the "
            "critic signals below. patch_suggestion is deliberately retained rather than "
            "replaced: existing reports carry it, and it stays the human-readable form."
        ),
    )
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Numeric confidence, 0.0-1.0")
    confidence_label: ConfidenceLabel
    verification_status: VerificationStatus
    related_findings: list[str] = Field(
        default_factory=list, description="finding_ids of other findings involved in the same exploit chain"
    )
    references: list[str] = Field(default_factory=list, description="External references, e.g. CWE/OWASP links")

    # Correlation linkage. finding_id above deliberately keeps its original meaning -
    # the canonical *raw* ScannerFinding this enrichment is about - so every existing
    # raw-finding -> AI join keeps working untouched. These two fields add the group
    # view alongside it rather than redefining it.
    correlation_id: Optional[str] = Field(
        None,
        description=(
            "CorrelatedFinding.correlation_id of the group this enrichment covers. None when "
            "the scan predates correlation. Renderers use it to resolve enrichment for the "
            "non-canonical raw findings in a group."
        ),
    )
    source_finding_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Every raw finding_id in the correlated group, including the canonical one. Empty "
            "for reports written before correlation existed."
        ),
    )

    # Critic-agent review signals. All three are optional with defaults so that a
    # report written before the multi-agent pipeline existed still loads: a saved
    # JSON report with none of these keys validates and simply carries None/[],
    # which every renderer treats as "no critic ran" rather than as an error.
    grounding_verdict: Optional[GroundingVerdict] = Field(
        None,
        description=(
            "Critic agent's judgement of how well this finding's narrative is supported by the "
            "supplied evidence. None when no critic ran. Never derived from, and never converted "
            "into, verification_status - the two are independent signals."
        ),
    )
    supported_claims: list[str] = Field(
        default_factory=list, description="Claims the critic judged to be supported by the supplied evidence"
    )
    unsupported_claims: list[str] = Field(
        default_factory=list,
        description="Claims the critic judged unsupported or uncertain given the supplied evidence",
    )
