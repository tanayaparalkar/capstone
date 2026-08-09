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
