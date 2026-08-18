"""
Canonical JSON report envelope.

Wraps the existing ScanResult data (repository/scan metadata + both
finding layers) and the existing ScanStatistics into one stable,
versioned structure - this is a composition of existing contracts, not a
second definition of finding fields. ScannerFinding, AIEnrichedFinding,
RepositoryInfo, and ScanMetadata remain the only source of truth for
their own fields; this model only arranges them for the report.

schema_version is a plain string (not tied to the package version) so the
report format can be versioned independently of sentinelai's own release
number - a report-shape change bumps this, a CLI bugfix release doesn't.

`findings.correlated` was added without bumping schema_version, which is
deliberate and not an oversight. reporting/loader.py rejects any report
whose schema_version is not exactly REPORT_SCHEMA_VERSION, so bumping to
"1.1" would make every previously-saved "1.0" report unloadable - it
would break backward compatibility rather than express it. The addition
is compatible in both directions without a bump: a new reader supplies
the default [] for an old report that has no `correlated` key, and an
older reader ignores the extra key (this model does not set
extra="forbid"). A bump is warranted when a change would actually make
an old report unreadable - renaming or removing a field, or changing a
field's meaning - which this does not.
"""
from pydantic import BaseModel, Field

from ..contracts import (
    AIEnrichedFinding,
    CorrelatedFinding,
    RepositoryInfo,
    ScanMetadata,
    ScannerFinding,
)
from ..statistics import ScanStatistics

REPORT_SCHEMA_VERSION = "1.0"


class ReportFindings(BaseModel):
    scanner: list[ScannerFinding] = Field(default_factory=list)
    ai_enriched: list[AIEnrichedFinding] = Field(default_factory=list)
    correlated: list[CorrelatedFinding] = Field(
        default_factory=list,
        description=(
            "Cross-scanner correlation groups. An index over `scanner`, never a substitute for "
            "it: every raw finding above is still present and unmodified. Defaults to [] so a "
            "report written before correlation existed still loads."
        ),
    )


class JSONReport(BaseModel):
    schema_version: str = REPORT_SCHEMA_VERSION
    repository: RepositoryInfo
    scan: ScanMetadata
    statistics: ScanStatistics
    findings: ReportFindings
