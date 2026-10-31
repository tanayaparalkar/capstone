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
from typing import Literal, Optional, Sequence

from pydantic import BaseModel, Field

from .ai import AIFault, AIRoleProvenance, RedactionReport
from .ai_finding import AIEnrichedFinding
from .correlated_finding import CorrelatedFinding
from .scanner_finding import ScannerFinding


class ScanMode(str, Enum):
    QUICK = "quick"
    STANDARD = "standard"
    FULL = "full"


class ScannerTier(str, Enum):
    """Which set of scanners a scan ran - an axis orthogonal to ScanMode.

    ScanMode is a depth control over the checks a scanner performs;
    ScannerTier is which scanners run at all. They are deliberately not
    merged: folding dependency scanning into ScanMode.FULL would silently
    change what an existing `--full` scan means and what its findings
    count, whereas a separate axis leaves every existing invocation
    producing exactly what it produced before.

    CORE is Semgrep + Bandit + GitLeaks, the set every scan has always
    run and the set the project's frozen performance baseline is measured
    against. EXTENDED adds the dependency scanners (Trivy, OSV-Scanner)
    and is opt-in, never a default.
    """

    CORE = "core"
    EXTENDED = "extended"


class RepositoryInfo(BaseModel):
    name: str = Field(..., description="Repository name, e.g. derived from the path or the git remote")
    path: str = Field(..., description="Local path or URL the repository was scanned from")
    commit_hash: Optional[str] = Field(None, description="Commit hash scanned, if the repo is a git checkout")
    branch: Optional[str] = Field(None, description="Branch scanned, if known")
    languages: list[str] = Field(default_factory=list, description="Programming languages detected in the repository")


RetentionStatus = Literal["retained", "not_retained", "unverified"]


def retention_status_from_provider(
    retains_data: Optional[bool],
    *,
    is_local: Optional[bool] = None,
    account_verified: bool = False,
) -> RetentionStatus:
    """Map provider metadata to ScanMetadata.retention_status.

    Reflects configured provider metadata only - not an independent verification of vendor terms.
    - `None` is preserved as 'unverified', never reinterpreted as False ('not_retained').
    - `True` maps to 'retained'.
    - `False` maps to 'not_retained' only when local (`is_local` is True or unspecified for legacy
      tri-state callers) or when `account_verified` is True. A remote provider (`is_local=False`)
      whose class merely defines `retains_data=False` without runtime verification is represented
      as 'unverified'.
    """
    if retains_data is True:
        return "retained"
    if retains_data is False:
        if is_local is False and not account_verified:
            return "unverified"
        return "not_retained"
    return "unverified"


def aggregate_retention_status(statuses: Sequence[Optional[str]]) -> RetentionStatus:
    """Conservatively aggregate retention assessments across all providers that processed scan data.

    Precedence: 'retained' > 'unverified' (including None/unknown) > 'not_retained'.
    """
    if not statuses:
        return "unverified"
    if any(s == "retained" for s in statuses):
        return "retained"
    if any(s is None or s == "unverified" for s in statuses):
        return "unverified"
    return "not_retained"


class ScanMetadata(BaseModel):
    timestamp: datetime = Field(..., description="When the scan was run (UTC)")
    mode: ScanMode = Field(..., description="Scan mode requested via the CLI")
    duration_seconds: Optional[float] = Field(None, ge=0, description="Total scan duration, once known")
    scanner_tier: ScannerTier = Field(
        ScannerTier.CORE,
        description=(
            "Which scanner set produced this result. Defaults to CORE so a report written "
            "before the extended tier existed still loads, and so a finding count can always "
            "be read against the scanner set that produced it."
        ),
    )
    # AI provider provenance. All optional and None by default so reports written
    # before multi-provider support existed - and scans with AI disabled - still load.
    ai_provider: Optional[str] = Field(None, description="LLM provider used for AI enrichment, if any.")
    ai_model: Optional[str] = Field(None, description="Model used for AI enrichment, if any.")
    egress_policy: Optional[str] = Field(None, description="Egress policy in force for AI enrichment ('allow'/'deny').")
    retention_status: Optional[RetentionStatus] = Field(
        None,
        description=(
            "Data-retention status reported by configured AI provider metadata "
            "('retained', 'not_retained', or 'unverified'). Reflects configured provider "
            "metadata only, not an independent verification of vendor retention policies. "
            "None for scanner-only runs and reports written before this field existed."
        ),
    )
    redaction: Optional[RedactionReport] = Field(
        None, description="Summary of secret redaction applied before any prompt was built. Never contains secret values."
    )
    role_provenance: list[AIRoleProvenance] = Field(
        default_factory=list,
        description=(
            "Per-role AI provider provenance (discovery, remediation, verification). Empty for "
            "single-provider runs and reports written before multi-role support existed."
        ),
    )
    ai_faults: list[AIFault] = Field(
        default_factory=list,
        description=(
            "Operational faults from AI enrichment (provider errors, egress refusals). Recorded here, never as "
            "findings: the affected findings are simply left unenriched. Empty for scanner-only runs and for "
            "reports written before this field existed."
        ),
    )


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
