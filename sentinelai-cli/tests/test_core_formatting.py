"""
Tests for sentinelai/core/formatting.py - the shared format_location() /
normalize_file_uri() helpers consolidated during the Milestone 12
architecture audit (previously duplicated verbatim across
presentation/findings.py, reporting/markdown_report.py, and
reporting/html_report.py).
"""
from sentinelai.contracts import ScannerFinding, Severity
from sentinelai.core import format_location, normalize_file_uri


def _finding(**overrides) -> ScannerFinding:
    defaults = dict(
        finding_id="SENT-001",
        scanner="semgrep",
        category="sql-injection",
        severity=Severity.CRITICAL,
        rule_id="rule",
        message="message",
    )
    defaults.update(overrides)
    return ScannerFinding(**defaults)


# --- format_location -----------------------------------------------------------------


def test_format_location_no_file():
    assert format_location(_finding(file=None)) == "n/a"


def test_format_location_file_no_line():
    assert format_location(_finding(file="requirements.txt")) == "requirements.txt"


def test_format_location_single_line():
    finding = _finding(file="app.py", line_start=10, line_end=10)
    assert format_location(finding) == "app.py:10"


def test_format_location_file_with_line_start_only():
    finding = _finding(file="app.py", line_start=10)
    assert format_location(finding) == "app.py:10"


def test_format_location_shows_line_range_when_end_differs():
    # This was the bug found during the Milestone 12 audit: the previous
    # (triplicated) helper always showed only line_start, silently
    # dropping line_end even when it differed - so a multi-line finding's
    # span was visible in SARIF but not in the terminal/Markdown/HTML.
    finding = _finding(file="app.py", line_start=10, line_end=15)
    assert format_location(finding) == "app.py:10-15"


# --- normalize_file_uri -----------------------------------------------------------------


def test_normalize_file_uri_none():
    assert normalize_file_uri(None) is None


def test_normalize_file_uri_already_forward_slashes_is_a_no_op():
    assert normalize_file_uri("config/settings.py") == "config/settings.py"


def test_normalize_file_uri_converts_backslashes():
    assert normalize_file_uri("config\\settings.py") == "config/settings.py"


def test_normalize_file_uri_mixed_separators():
    assert normalize_file_uri("app\\db\\queries.py") == "app/db/queries.py"
