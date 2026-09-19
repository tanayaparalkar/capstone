"""Terminal presentation layer: Rich rendering, used only for --format terminal."""
from .console import get_console, get_error_console, print_error, print_traceback
from .findings import render_finding_detail, render_findings_table
from .header import render_scan_header
from .narrator import prompt_user_for_fix, render_remediation_summary, render_vulnerability_narration
from .progress import ScanProgress
from .summary import render_summary

__all__ = [
    "get_console",
    "get_error_console",
    "print_error",
    "print_traceback",
    "render_finding_detail",
    "render_findings_table",
    "render_scan_header",
    "render_summary",
    "render_vulnerability_narration",
    "prompt_user_for_fix",
    "render_remediation_summary",
    "ScanProgress",
]

