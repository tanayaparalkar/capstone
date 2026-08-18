"""
Tests for the Markdown report generator (sentinelai/reporting/markdown_report.py).

to_markdown() is a pure function of an existing ScanResult + a
pre-computed ScanStatistics - no scanner/AI logic, no statistics
recalculation, no second data model. These tests exercise that
composition directly (test_cli.py covers CLI-integration: stdout purity,
file writing, terminal-format non-regression).
"""
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
from sentinelai.reporting import to_markdown
from sentinelai.statistics import calculate_statistics


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
        impact="Full authentication bypass.",
        remediation="Use parameterized queries.",
        confidence_score=0.92,
        confidence_label=ConfidenceLabel.HIGH,
        verification_status=VerificationStatus.VERIFIED,
    )
    defaults.update(overrides)
    return AIEnrichedFinding(**defaults)


def _scan_result(scanner_findings=None, ai_findings=None, **metadata_overrides) -> ScanResult:
    metadata_defaults = dict(timestamp="2026-08-09T12:00:00Z", mode=ScanMode.STANDARD, duration_seconds=1.5)
    metadata_defaults.update(metadata_overrides)
    return ScanResult(
        repository=RepositoryInfo(name="demo-app", path="/tmp/demo-app", commit_hash="abc123", branch="main"),
        metadata=ScanMetadata(**metadata_defaults),
        scanner_findings=scanner_findings or [],
        ai_findings=ai_findings or [],
    )


def _render(result: ScanResult) -> str:
    return to_markdown(result, calculate_statistics(result))


# --- basic generation -----------------------------------------------------------------


def test_valid_markdown_generation():
    md = _render(_scan_result(scanner_findings=[_scanner_finding()]))
    assert md.startswith("# SentinelAI Security Report")


def test_empty_findings():
    md = _render(_scan_result())
    assert "Total Findings:** 0" in md
    assert "No findings were reported" in md
    assert "## Findings Overview" not in md
    assert "## Detailed Findings" not in md


def test_one_scanner_finding():
    md = _render(_scan_result(scanner_findings=[_scanner_finding()]))
    assert "SENT-001" in md
    assert "sql-injection" in md
    assert "app/db/queries.py:47" in md


def test_multi_line_finding_shows_line_range():
    # Regression test for a Milestone 12 audit finding: line_end was
    # silently dropped in Markdown (and HTML/terminal) even when it
    # differed from line_start, while SARIF always showed it correctly.
    finding = _scanner_finding(file="app.py", line_start=10, line_end=15)
    md = _render(_scan_result(scanner_findings=[finding]))
    assert "app.py:10-15" in md


def test_multiple_findings():
    findings = [_scanner_finding(finding_id=f"SENT-{i:03d}") for i in range(1, 6)]
    md = _render(_scan_result(scanner_findings=findings))
    for i in range(1, 6):
        assert f"SENT-{i:03d}" in md


# --- statistics rendering -----------------------------------------------------------------


def test_severity_summary_from_statistics_not_recalculated():
    findings = [
        _scanner_finding(finding_id="SENT-001", severity=Severity.CRITICAL),
        _scanner_finding(finding_id="SENT-002", severity=Severity.CRITICAL),
        _scanner_finding(finding_id="SENT-003", severity=Severity.HIGH),
    ]
    result = _scan_result(scanner_findings=findings)
    stats = calculate_statistics(result)
    md = to_markdown(result, stats)
    assert f"| Critical | {stats.critical_findings} |" in md
    assert f"| High | {stats.high_findings} |" in md
    assert "| Critical | 2 |" in md
    assert "| High | 1 |" in md


def test_scanner_distribution_rendered():
    findings = [
        _scanner_finding(finding_id="SENT-001", scanner="semgrep"),
        _scanner_finding(finding_id="SENT-002", scanner="bandit"),
    ]
    md = _render(_scan_result(scanner_findings=findings))
    assert "Findings by Scanner" in md
    assert "| semgrep | 1 |" in md
    assert "| bandit | 1 |" in md


def test_category_distribution_rendered():
    findings = [
        _scanner_finding(finding_id="SENT-001", category="sql-injection"),
        _scanner_finding(finding_id="SENT-002", category="secret-exposure"),
    ]
    md = _render(_scan_result(scanner_findings=findings))
    assert "Findings by Category" in md
    assert "| sql-injection | 1 |" in md
    assert "| secret-exposure | 1 |" in md


def test_statistics_values_match_calculator_exactly():
    findings = [_scanner_finding(finding_id=f"SENT-{i:03d}") for i in range(1, 4)]
    result = _scan_result(scanner_findings=findings)
    stats = calculate_statistics(result)
    md = to_markdown(result, stats)
    assert f"Total Findings:** {stats.total_findings}" in md
    assert f"AI Enrichment:** {stats.ai_enrichment_status.value}" in md


# --- correlation -----------------------------------------------------------------


def test_scanner_only_finding_has_no_ai_section():
    md = _render(_scan_result(scanner_findings=[_scanner_finding()]))
    assert "AI enrichment not yet available for this finding" in md
    assert "AI Explanation" not in md


def test_ai_enriched_finding_shows_ai_section():
    result = _scan_result(scanner_findings=[_scanner_finding()], ai_findings=[_ai_finding()])
    md = _render(result)
    assert "**AI Explanation**" in md
    assert "**Exploit Path**" in md
    assert "**Remediation**" in md
    assert "Confidence:** HIGH (0.92)" in md
    assert "Verification:** verified" in md


def test_mixed_scanner_and_ai_findings():
    result = _scan_result(
        scanner_findings=[_scanner_finding(finding_id="SENT-001"), _scanner_finding(finding_id="SENT-002")],
        ai_findings=[_ai_finding(finding_id="SENT-001")],
    )
    md = _render(result)
    assert md.count("### SENT-") == 2
    assert "AI enrichment not yet available for this finding" in md
    assert "**AI Explanation**" in md


def test_correlation_by_finding_id_with_mismatched_lengths():
    # scanner: SENT-001, SENT-002, SENT-003 / AI: SENT-002, SENT-003 (per spec example)
    scanner_findings = [
        _scanner_finding(finding_id="SENT-001"),
        _scanner_finding(finding_id="SENT-002"),
        _scanner_finding(finding_id="SENT-003"),
    ]
    ai_findings = [_ai_finding(finding_id="SENT-002"), _ai_finding(finding_id="SENT-003")]
    result = _scan_result(scanner_findings=scanner_findings, ai_findings=ai_findings)
    md = _render(result)

    sections = md.split("### SENT-")
    sent_001_section = next(s for s in sections if s.startswith("001"))
    sent_002_section = next(s for s in sections if s.startswith("002"))
    sent_003_section = next(s for s in sections if s.startswith("003"))

    assert "AI enrichment not yet available" in sent_001_section
    assert "**AI Explanation**" in sent_002_section
    assert "**AI Explanation**" in sent_003_section


def test_ai_finding_correlation_ignores_list_order():
    scanner_findings = [_scanner_finding(finding_id="SENT-001"), _scanner_finding(finding_id="SENT-002")]
    ai_findings = [_ai_finding(finding_id="SENT-002"), _ai_finding(finding_id="SENT-001")]  # reversed
    md = _render(_scan_result(scanner_findings=scanner_findings, ai_findings=ai_findings))
    assert md.count("**AI Explanation**") == 2


# --- optional fields -----------------------------------------------------------------


def test_missing_optional_fields_omitted_cleanly():
    finding = _scanner_finding(cwe=None, raw_evidence=None)
    md = _render(_scan_result(scanner_findings=[finding]))
    assert "**CWE:**" not in md
    assert "**Evidence**" not in md


def test_cwe_rendered_when_present():
    finding = _scanner_finding(cwe="CWE-798")
    md = _render(_scan_result(scanner_findings=[finding]))
    assert "**CWE:** CWE-798" in md


def test_ai_optional_fields_omitted_when_absent():
    ai = _ai_finding(exploit_path=None, impact=None, patch_suggestion=None, related_findings=[], references=[])
    md = _render(_scan_result(scanner_findings=[_scanner_finding()], ai_findings=[ai]))
    assert "**Exploit Path**" not in md
    assert "**Impact**" not in md
    assert "**Patch Suggestion**" not in md
    assert "Related Findings" not in md
    assert "References" not in md


# --- safe rendering of evidence -----------------------------------------------------------------


def test_raw_evidence_with_markdown_characters_does_not_break_structure():
    evidence = "# This looks like a heading\n<script>alert(1)</script>\n**bold** and _italic_"
    finding = _scanner_finding(raw_evidence=evidence)
    md = _render(_scan_result(scanner_findings=[finding]))
    # The evidence must appear verbatim, wrapped in a fence - not interpreted
    # as an actual heading/HTML/bold text by anything that renders this file.
    assert "```\n" + evidence + "\n```" in md


def test_raw_evidence_with_triple_backticks_uses_a_longer_fence():
    evidence = "some text\n```\nnested fenced block\n```\nmore text"
    finding = _scanner_finding(raw_evidence=evidence)
    md = _render(_scan_result(scanner_findings=[finding]))
    assert evidence in md
    # The wrapping fence must be longer than the 3-backtick run inside the evidence.
    assert "````\n" + evidence + "\n````" in md


def test_raw_evidence_with_single_backticks():
    evidence = "call `dangerous_function()` here"
    finding = _scanner_finding(raw_evidence=evidence)
    md = _render(_scan_result(scanner_findings=[finding]))
    assert "```\n" + evidence + "\n```" in md


def test_patch_suggestion_uses_fenced_code():
    ai = _ai_finding(patch_suggestion="-query = f\"...\"\n+query = \"SELECT * FROM users WHERE username = %s\"")
    md = _render(_scan_result(scanner_findings=[_scanner_finding()], ai_findings=[ai]))
    assert "**Patch Suggestion**" in md
    assert "```\n" + ai.patch_suggestion + "\n```" in md


# --- unicode -----------------------------------------------------------------


def test_unicode_content_preserved():
    finding = _scanner_finding(message="Ce message contient des caractères non-ASCII: 中文 ✓")
    md = _render(_scan_result(scanner_findings=[finding]))
    assert finding.message in md


# --- determinism -----------------------------------------------------------------


def test_deterministic_ordering_regardless_of_input_order():
    findings = [
        _scanner_finding(finding_id="SENT-003"),
        _scanner_finding(finding_id="SENT-001"),
        _scanner_finding(finding_id="SENT-002"),
    ]
    md = _render(_scan_result(scanner_findings=findings))
    positions = [md.index(f"SENT-{i:03d}") for i in (1, 2, 3)]
    assert positions == sorted(positions)


def test_deterministic_output_for_same_input():
    result = _scan_result(scanner_findings=[_scanner_finding()])
    stats = calculate_statistics(result)
    assert to_markdown(result, stats) == to_markdown(result, stats)
