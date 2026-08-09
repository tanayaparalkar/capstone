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
from .models import REPORT_SCHEMA_VERSION, JSONReport, ReportFindings


def build_json_report(result: ScanResult, statistics: ScanStatistics) -> JSONReport:
    return JSONReport(
        schema_version=REPORT_SCHEMA_VERSION,
        repository=result.repository,
        scan=result.metadata,
        statistics=statistics,
        findings=ReportFindings(
            scanner=sorted(result.scanner_findings, key=lambda f: f.finding_id),
            ai_enriched=sorted(result.ai_findings, key=lambda f: f.finding_id),
        ),
    )


def to_json(result: ScanResult, statistics: ScanStatistics, *, indent: int = 2) -> str:
    """Pretty-printed (indented) JSON by default - this is the human/CI-facing report."""
    return build_json_report(result, statistics).model_dump_json(indent=indent)
