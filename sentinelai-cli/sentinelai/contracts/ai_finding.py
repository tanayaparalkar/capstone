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
from typing import Optional

from pydantic import BaseModel, Field

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
