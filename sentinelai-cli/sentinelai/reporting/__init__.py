"""Report generation: JSON, Markdown, HTML, SARIF - plus loading a saved report back for `sentinelai report`."""
from .html_report import to_html
from .json_report import build_json_report, to_json
from .loader import ReportLoadError, load_patch_application, load_scan_result
from .markdown_report import to_markdown
from .models import (
    REPORT_SCHEMA_VERSION,
    JSONReport,
    PatchApplicationReport,
    PatchApplicationSummary,
    PatchAttemptReport,
    ReportFindings,
)
from .patch_section import build_patch_application_report
from .sarif_report import SARIF_VERSION, build_sarif_report, to_sarif

__all__ = [
    "build_json_report",
    "to_json",
    "to_markdown",
    "to_html",
    "build_sarif_report",
    "to_sarif",
    "SARIF_VERSION",
    "JSONReport",
    "ReportFindings",
    "build_patch_application_report",
    "PatchAttemptReport",
    "PatchApplicationSummary",
    "PatchApplicationReport",
    "load_patch_application",
    "REPORT_SCHEMA_VERSION",
    "load_scan_result",
    "ReportLoadError",
]
