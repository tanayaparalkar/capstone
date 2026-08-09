"""
Statistics data model.

Pure structured output of the statistics engine - no Rich, no formatting
concerns. Any consumer (terminal summary, JSON/Markdown/HTML reports, a
future report viewer, or a research/evaluation script) works from the
same ScanStatistics object, so numbers can never drift between output
formats.

Severity and confidence each get explicit named fields because they're
closed, contract-owned enums (Severity, ConfidenceLabel); scanner and
category counts stay open dict[str, int] distributions because scanner
names and vulnerability categories are not a fixed set this engine may
assume - a new scanner or category must show up correctly with zero
changes to this model.
"""
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class AIEnrichmentStatus(str, Enum):
    """Whether AI enrichment covers none, some, or all of this scan's findings."""

    UNAVAILABLE = "unavailable"
    PARTIAL = "partial"
    AVAILABLE = "available"


class ConfidenceStatistics(BaseModel):
    """Only meaningful when ai_findings is non-empty; all-zero/None otherwise."""

    enriched_count: int = 0
    high_count: int = 0
    medium_count: int = 0
    low_count: int = 0
    average_score: Optional[float] = None
    min_score: Optional[float] = None
    max_score: Optional[float] = None


class ScanStatistics(BaseModel):
    # Finding counts
    total_findings: int
    critical_findings: int
    high_findings: int
    medium_findings: int
    low_findings: int

    # Open-ended distributions - never hardcode scanner/category names
    scanner_counts: dict[str, int] = Field(default_factory=dict)
    category_counts: dict[str, int] = Field(default_factory=dict)

    # AI enrichment: matched_ai_findings counts scanner_findings that have a
    # corresponding AIEnrichedFinding (correlated by finding_id); confidence
    # and verification_counts are computed over all ai_findings as produced,
    # including any that don't correlate to a scanner finding in this result.
    ai_enrichment_status: AIEnrichmentStatus
    matched_ai_findings: int = 0
    confidence: ConfidenceStatistics = Field(default_factory=ConfidenceStatistics)
    verification_counts: dict[str, int] = Field(default_factory=dict)

    # Repository / scan metadata - only what ScanResult actually provides
    repository_name: str
    scan_mode: str
    timestamp: datetime
    duration_seconds: Optional[float] = None
    files_analyzed: Optional[int] = Field(
        None, description="Not provided by any current contract - always None until Nithanth's layer exposes it."
    )
