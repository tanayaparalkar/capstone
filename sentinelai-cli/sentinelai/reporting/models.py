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
"""
from pydantic import BaseModel, Field

from ..contracts import AIEnrichedFinding, RepositoryInfo, ScanMetadata, ScannerFinding
from ..statistics import ScanStatistics

REPORT_SCHEMA_VERSION = "1.0"


class ReportFindings(BaseModel):
    scanner: list[ScannerFinding] = Field(default_factory=list)
    ai_enriched: list[AIEnrichedFinding] = Field(default_factory=list)


class JSONReport(BaseModel):
    schema_version: str = REPORT_SCHEMA_VERSION
    repository: RepositoryInfo
    scan: ScanMetadata
    statistics: ScanStatistics
    findings: ReportFindings
