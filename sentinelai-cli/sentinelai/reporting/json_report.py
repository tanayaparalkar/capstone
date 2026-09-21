"""
JSON report generator.

Builds the canonical JSON report envelope from an existing ScanResult and
ScanStatistics - it does not recalculate statistics (that's
sentinelai.statistics.calculate_statistics's job, called once by the
CLI) and does not redefine finding fields (ScannerFinding /
AIEnrichedFinding remain the only source of truth). This is the report
format the future Markdown/HTML/SARIF generators, CI tooling, and
research workflows will all consume.

Findings are sorted by finding_id before serialization so the report is
deterministic regardless of the order a provider happens to return them
in - a pure presentational reorder, not a change in meaning.
"""
from ..contracts import ScanResult
from ..statistics import ScanStatistics
from typing import Optional

from .models import (
    REPORT_SCHEMA_VERSION,
    JSONReport,
    PatchApplicationReport,
    ReportFindings,
)


def build_json_report(result: ScanResult, statistics: ScanStatistics, patch_application: Optional[PatchApplicationReport] = None) -> JSONReport:
    return JSONReport(
        schema_version=REPORT_SCHEMA_VERSION,
        repository=result.repository,
        scan=result.metadata,
        statistics=statistics,
        findings=ReportFindings(
            scanner=sorted(result.scanner_findings, key=lambda f: f.finding_id),
            ai_enriched=sorted(result.ai_findings, key=lambda f: f.finding_id),
            # Sorted by correlation_id, which the correlator already assigns from a
            # content-based ordering - so the serialized report is deterministic for
            # the same scan regardless of the order scanners ran in.
            correlated=sorted(result.correlated_findings, key=lambda c: c.correlation_id),
        ),
        # Deliberately a sibling of `findings`, not a member of it: patch
        # application is a separate deterministic stage, and nesting its outcomes
        # under a finding would blur which part of the pipeline produced what.
        patch_application=patch_application,
    )


def to_json(result: ScanResult, statistics: ScanStatistics, *, indent: int = 2,
            patch_application: Optional[PatchApplicationReport] = None) -> str:
    """Pretty-printed (indented) JSON by default - this is the human/CI-facing report."""
    return build_json_report(result, statistics, patch_application).model_dump_json(indent=indent)
