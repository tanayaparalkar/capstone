"""
Unit tests for the terminal presentation layer (sentinelai/presentation/).

Each render function takes an explicit Console, so tests construct a
Console writing to an in-memory buffer instead of going through the full
CLI - this keeps these tests fast and lets width/color be controlled
directly per scenario (narrow terminal, non-interactive, etc.).
"""
from datetime import datetime, timezone
from io import StringIO

from rich.console import Console

from sentinelai.contracts import (
    AIEnrichedFinding,
    ConfidenceLabel,
    RepositoryInfo,
    ScanMetadata,
    ScanMode,
    ScanResult,
    ScannerFinding,
    Severity,
    VerificationStatus,
)
from sentinelai.presentation import (
    ScanProgress,
    render_finding_detail,
    render_findings_table,
    render_scan_header,
    render_summary,
)
from sentinelai.statistics import calculate_statistics


def _buffer_console(width: int = 100, no_color: bool = False) -> Console:
    return Console(file=StringIO(), width=width, force_terminal=False, no_color=no_color)


def _output(console: Console) -> str:
    return console.file.getvalue()


def _scanner_finding(**overrides) -> ScannerFinding:
    defaults = dict(
        finding_id="SENT-001",
        scanner="semgrep",
        category="sql-injection",
        severity=Severity.CRITICAL,
        file="app/db/queries.py",
        line_start=47,
        rule_id="semgrep.python.sql-injection.string-concat",
        message="User-supplied input concatenated directly into a SQL query string.",
        raw_evidence="query = f\"SELECT * FROM users WHERE username = '{username}'\"",
        cwe="CWE-89",
    )
    defaults.update(overrides)
    return ScannerFinding(**defaults)


def _ai_finding(**overrides) -> AIEnrichedFinding:
    defaults = dict(
        finding_id="SENT-001",
        title="SQL injection via unparameterized query",
        severity=Severity.CRITICAL,
        scanner_sources=["semgrep"],
        explanation="The query is built via string concatenation, allowing arbitrary SQL to be injected.",
        exploit_path="Submitting \"' OR '1'='1' --\" bypasses authentication.",
        impact="Full authentication bypass and potential database compromise.",
        remediation="Use parameterized queries.",
        confidence_score=0.95,
        confidence_label=ConfidenceLabel.HIGH,
        verification_status=VerificationStatus.VERIFIED,
    )
    defaults.update(overrides)
    return AIEnrichedFinding(**defaults)


def _scan_result(scanner_findings=None, ai_findings=None) -> ScanResult:
    return ScanResult(
        repository=RepositoryInfo(name="demo-app", path="/tmp/demo-app"),
        metadata=ScanMetadata(timestamp="2026-08-08T12:00:00Z", mode=ScanMode.STANDARD),
        scanner_findings=scanner_findings or [],
        ai_findings=ai_findings or [],
    )


# --- header -----------------------------------------------------------------


def test_render_scan_header_contains_key_fields():
    console = _buffer_console()
    render_scan_header(
        console,
        repository="demo-app",
        mode="standard",
        version="0.1.0",
        started_at=datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc),
    )
    out = _output(console)
    assert "SentinelAI Security Scan" in out
    assert "demo-app" in out
    assert "standard" in out
    assert "0.1.0" in out


# --- progress -----------------------------------------------------------------


def test_scan_progress_stage_prints_start_and_completion():
    console = _buffer_console()
    with ScanProgress(console).stage("Retrieving scan results (MockFindingsProvider)"):
        pass
    out = _output(console)
    assert "Retrieving scan results (MockFindingsProvider)..." in out
    assert "Retrieving scan results (MockFindingsProvider) - done" in out


def test_scan_progress_stage_reports_failure_and_reraises():
    console = _buffer_console()
    try:
        with ScanProgress(console).stage("Retrieving scan results"):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    else:
        assert False, "expected RuntimeError to propagate"
    out = _output(console)
    assert "Retrieving scan results - failed" in out


def test_scan_progress_does_not_fake_unrun_stages():
    # No provider today reports "static analysis" or "AI reasoning" as real
    # events, so nothing in the presentation layer should ever print them.
    console = _buffer_console()
    with ScanProgress(console).stage("Retrieving scan results (MockFindingsProvider)"):
        pass
    out = _output(console).lower()
    for fake_stage in ("semgrep completed", "ai reasoning completed", "static analysis completed", "verification completed"):
        assert fake_stage not in out


# --- summary -----------------------------------------------------------------


def test_render_summary_empty_findings():
    console = _buffer_console()
    render_summary(console, calculate_statistics(_scan_result()))
    out = _output(console)
    assert "Total findings" in out
    assert "0" in out


def test_render_summary_counts_by_severity():
    findings = [
        _scanner_finding(finding_id="SENT-001", severity=Severity.CRITICAL),
        _scanner_finding(finding_id="SENT-002", severity=Severity.HIGH),
        _scanner_finding(finding_id="SENT-003", severity=Severity.MEDIUM),
        _scanner_finding(finding_id="SENT-004", severity=Severity.LOW),
    ]
    console = _buffer_console()
    render_summary(console, calculate_statistics(_scan_result(scanner_findings=findings)))
    out = _output(console)
    assert "CRITICAL" in out
    assert "HIGH" in out
    assert "MEDIUM" in out
    assert "LOW" in out


def test_render_summary_without_ai_enrichment_says_not_available():
    console = _buffer_console()
    render_summary(console, calculate_statistics(_scan_result(scanner_findings=[_scanner_finding()])))
    out = _output(console)
    assert "not yet available" in out
    # Must not fabricate a confidence/verification breakdown that doesn't exist.
    assert "Confidence" not in out
    assert "Verification" not in out


def test_render_summary_with_ai_enrichment_shows_real_counts():
    console = _buffer_console()
    result = _scan_result(scanner_findings=[_scanner_finding()], ai_findings=[_ai_finding()])
    render_summary(console, calculate_statistics(result))
    out = _output(console)
    assert "1/1 findings enriched" in out
    assert "high" in out.lower()
    assert "verified" in out.lower()


# --- findings table -----------------------------------------------------------------


def test_render_findings_table_empty():
    console = _buffer_console()
    render_findings_table(console, _scan_result())
    out = _output(console)
    assert "SentinelAI Findings" in out


def test_render_findings_table_one_finding():
    console = _buffer_console()
    render_findings_table(console, _scan_result(scanner_findings=[_scanner_finding()]))
    out = _output(console)
    assert "SENT-001" in out
    assert "CRITICAL" in out
    assert "semgrep" in out


def test_render_findings_table_multi_line_finding_shows_line_range():
    # Regression test for a Milestone 12 audit finding: line_end was
    # silently dropped in the terminal table (and Markdown/HTML) even
    # when it differed from line_start, while SARIF always showed it.
    console = _buffer_console(width=100)
    finding = _scanner_finding(file="app.py", line_start=10, line_end=15)
    render_findings_table(console, _scan_result(scanner_findings=[finding]))
    out = _output(console)
    assert "app.py:10-15" in out


def test_render_findings_table_multiple_findings_all_severities():
    findings = [
        _scanner_finding(finding_id="SENT-001", severity=Severity.CRITICAL, category="secret-exposure"),
        _scanner_finding(finding_id="SENT-002", severity=Severity.HIGH, category="command-injection"),
        _scanner_finding(finding_id="SENT-003", severity=Severity.MEDIUM, category="weak-cryptography"),
        _scanner_finding(finding_id="SENT-004", severity=Severity.LOW, category="misconfiguration"),
    ]
    console = _buffer_console()
    render_findings_table(console, _scan_result(scanner_findings=findings))
    out = _output(console)
    for finding_id in ("SENT-001", "SENT-002", "SENT-003", "SENT-004"):
        assert finding_id in out
    for sev in ("CRITICAL", "HIGH", "MEDIUM", "LOW"):
        assert sev in out


def test_render_findings_table_long_file_path_does_not_crash():
    long_path = "app/" + "very_long_nested_directory_name/" * 6 + "module.py"
    console = _buffer_console(width=100)
    render_findings_table(console, _scan_result(scanner_findings=[_scanner_finding(file=long_path)]))
    out = _output(console)
    assert "SENT-001" in out


def test_render_findings_table_pending_confidence_without_ai_enrichment():
    console = _buffer_console()
    render_findings_table(console, _scan_result(scanner_findings=[_scanner_finding()]))
    out = _output(console)
    assert "PENDING" in out


def test_render_findings_table_shows_confidence_with_ai_enrichment():
    console = _buffer_console()
    result = _scan_result(scanner_findings=[_scanner_finding()], ai_findings=[_ai_finding()])
    render_findings_table(console, result)
    out = _output(console)
    assert "HIGH" in out  # confidence label, in addition to severity
    assert "PENDING" not in out


def test_render_findings_table_narrow_terminal_width_does_not_crash():
    console = _buffer_console(width=40)
    findings = [_scanner_finding(finding_id=f"SENT-{i:03d}") for i in range(1, 4)]
    render_findings_table(console, _scan_result(scanner_findings=findings))
    out = _output(console)
    assert "SENT-001" in out


def test_render_findings_table_no_color_mode():
    console = _buffer_console(no_color=True)
    render_findings_table(console, _scan_result(scanner_findings=[_scanner_finding()]))
    out = _output(console)
    assert "\x1b[" not in out
    assert "SENT-001" in out
    assert "CRITICAL" in out


# --- finding detail -----------------------------------------------------------------


def test_render_finding_detail_without_ai_enrichment():
    console = _buffer_console()
    render_finding_detail(console, _scanner_finding(), None)
    out = _output(console)
    assert "SENT-001" in out
    assert "sql-injection" in out
    assert "CWE-89" in out
    assert "AI enrichment not yet available" in out
    # Must not fabricate any AI content when none exists.
    assert "Remediation" not in out
    assert "Confidence" not in out


def test_render_finding_detail_with_ai_enrichment():
    console = _buffer_console()
    render_finding_detail(console, _scanner_finding(), _ai_finding())
    out = _output(console)
    assert "AI Explanation" in out
    assert "Exploit Path" in out
    assert "Remediation" in out
    assert "verified" in out.lower()


def test_render_finding_detail_long_description_does_not_crash():
    long_message = "A" * 500
    console = _buffer_console(width=80)
    render_finding_detail(console, _scanner_finding(message=long_message), None)
    out = _output(console)
    assert "SENT-001" in out


def test_render_finding_detail_missing_optional_fields():
    finding = _scanner_finding(raw_evidence=None, cwe=None)
    ai = _ai_finding(exploit_path=None, impact=None)
    console = _buffer_console()
    render_finding_detail(console, finding, ai)
    out = _output(console)
    assert "Evidence" not in out
    assert "Exploit Path" not in out
    assert "Impact" not in out
    assert "Remediation" in out
