"""
Scan-level envelope.

Wraps repository metadata, scan metadata, and the two finding layers
(scanner-level and AI-enriched) into the single object the reporting
layer consumes. New optional fields can be added to any model here later
(e.g. tool version, scan_id) without breaking existing reporting code,
since Pydantic models degrade gracefully to their defaults.
"""
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from .ai_finding import AIEnrichedFinding
from .correlated_finding import CorrelatedFinding
from .scanner_finding import ScannerFinding


class ScanMode(str, Enum):
    QUICK = "quick"
    STANDARD = "standard"
    FULL = "full"


class RepositoryInfo(BaseModel):
    name: str = Field(..., description="Repository name, e.g. derived from the path or the git remote")
    path: str = Field(..., description="Local path or URL the repository was scanned from")
    commit_hash: Optional[str] = Field(None, description="Commit hash scanned, if the repo is a git checkout")
    branch: Optional[str] = Field(None, description="Branch scanned, if known")
    languages: list[str] = Field(default_factory=list, description="Programming languages detected in the repository")


class ScanMetadata(BaseModel):
    timestamp: datetime = Field(..., description="When the scan was run (UTC)")
    mode: ScanMode = Field(..., description="Scan mode requested via the CLI")
    duration_seconds: Optional[float] = Field(None, ge=0, description="Total scan duration, once known")


class ScanResult(BaseModel):
    repository: RepositoryInfo
    metadata: ScanMetadata
    scanner_findings: list[ScannerFinding] = Field(default_factory=list)
    ai_findings: list[AIEnrichedFinding] = Field(default_factory=list)
    correlated_findings: list[CorrelatedFinding] = Field(
        default_factory=list,
        description=(
            "Raw findings grouped into logical issues by sentinelai.correlation. An index over "
            "scanner_findings, never a replacement: every raw finding is preserved above and "
            "appears in exactly one group here. Empty for reports written before correlation "
            "existed, which is what keeps those reports loadable."
        ),
    )
