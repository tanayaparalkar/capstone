"""
Shared display formatting for ScannerFinding data.

format_location() was previously duplicated verbatim across
presentation/findings.py, reporting/markdown_report.py, and
reporting/html_report.py. Consolidated here as the one place both
packages depend on (see core/severity.py for the same rationale).

Fixed while consolidating: the duplicated versions only ever showed
`file:line_start`, silently dropping `line_end` even when it differed
from `line_start` - so a multi-line finding's span was visible in SARIF
(which does include endLine) but not in the terminal table, Markdown, or
HTML output. format_location() now includes the end line when present
and different, matching SARIF's behavior. This is a no-op for the
current mock dataset, where every finding has line_end == line_start.
"""
from typing import Optional

from ..contracts import ScannerFinding


def format_location(f: ScannerFinding) -> str:
    if not f.file:
        return "n/a"
    if not f.line_start:
        return f.file
    if f.line_end and f.line_end != f.line_start:
        return f"{f.file}:{f.line_start}-{f.line_end}"
    return f"{f.file}:{f.line_start}"


def normalize_file_uri(file: Optional[str]) -> Optional[str]:
    """
    Normalize a ScannerFinding.file path into a valid URI path.

    SARIF requires artifactLocation.uri to be a URI-reference, which
    uses forward slashes. The current mock provider always supplies
    forward-slash paths already (this is a no-op for it), but a future
    provider running on Windows could plausibly supply OS-native
    backslash paths - normalizing here keeps SARIF output spec-valid
    regardless of what separator the provider used.
    """
    if file is None:
        return None
    return file.replace("\\", "/")
