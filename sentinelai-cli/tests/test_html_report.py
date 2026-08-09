"""
Tests for the HTML report generator (sentinelai/reporting/html_report.py).

to_html() is a pure function of an existing ScanResult + a pre-computed
ScanStatistics - no scanner/AI logic, no statistics recalculation, no
second data model. Security (autoescaping) is the primary concern here:
finding/AI content may eventually include LLM-generated text or embedded
code snippets, so every injection scenario must render as inert text.

HTML validity is checked with lightweight substring/structural
assertions rather than a full parser dependency - this project avoids
adding a dependency purely for testing.
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
from sentinelai.reporting import to_html
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
        confidence_score=0.917,
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
    return to_html(result, calculate_statistics(result))


# --- generation / structure -----------------------------------------------------------------


def test_html_generation():
    html = _render(_scan_result(scanner_findings=[_scanner_finding()]))
    assert isinstance(html, str)
    assert len(html) > 0


def test_valid_html_structure():
    html = _render(_scan_result(scanner_findings=[_scanner_finding()]))
    assert html.strip().startswith("<!DOCTYPE html>")
    assert "<html" in html
    assert "<head>" in html and "</head>" in html
    assert "<body>" in html and "</body>" in html
    assert html.rstrip().endswith("</html>")
    assert html.index("<head>") < html.index("<body>") < html.index("</html>")


def test_empty_findings():
    html = _render(_scan_result())
    assert "No findings were reported" in html
    assert "<article" not in html  # no finding cards rendered, only the CSS class definition exists


def test_one_finding():
    html = _render(_scan_result(scanner_findings=[_scanner_finding()]))
    assert "SENT-001" in html
    assert "sql-injection" in html
    assert 'id="finding-SENT-001"' in html


def test_multi_line_finding_shows_line_range():
    # Regression test for a Milestone 12 audit finding: line_end was
    # silently dropped in HTML (and Markdown/terminal) even when it
    # differed from line_start, while SARIF always showed it correctly.
    finding = _scanner_finding(file="app.py", line_start=10, line_end=15)
    html = _render(_scan_result(scanner_findings=[finding]))
    assert "app.py:10-15" in html


def test_multiple_findings():
    findings = [_scanner_finding(finding_id=f"SENT-{i:03d}") for i in range(1, 6)]
    html = _render(_scan_result(scanner_findings=findings))
    for i in range(1, 6):
        assert f"SENT-{i:03d}" in html


def test_all_severity_levels():
    findings = [
        _scanner_finding(finding_id="SENT-001", severity=Severity.CRITICAL),
        _scanner_finding(finding_id="SENT-002", severity=Severity.HIGH),
        _scanner_finding(finding_id="SENT-003", severity=Severity.MEDIUM),
        _scanner_finding(finding_id="SENT-004", severity=Severity.LOW),
    ]
    html = _render(_scan_result(scanner_findings=findings))
    for badge in ("badge critical", "badge high", "badge medium", "badge low"):
        assert badge in html


# --- statistics rendering -----------------------------------------------------------------


def test_scanner_distribution_rendered():
    findings = [
        _scanner_finding(finding_id="SENT-001", scanner="semgrep"),
        _scanner_finding(finding_id="SENT-002", scanner="bandit"),
    ]
    html = _render(_scan_result(scanner_findings=findings))
    assert "Findings by Scanner" in html
    assert "semgrep" in html
    assert "bandit" in html


def test_category_distribution_rendered():
    findings = [
        _scanner_finding(finding_id="SENT-001", category="sql-injection"),
        _scanner_finding(finding_id="SENT-002", category="secret-exposure"),
    ]
    html = _render(_scan_result(scanner_findings=findings))
    assert "Findings by Category" in html
    assert "sql-injection" in html
    assert "secret-exposure" in html


def test_statistics_values_match_calculator_not_recalculated():
    findings = [_scanner_finding(finding_id=f"SENT-{i:03d}") for i in range(1, 4)]
    result = _scan_result(scanner_findings=findings)
    stats = calculate_statistics(result)
    html = to_html(result, stats)
    assert f">{stats.total_findings}<" in html


# --- AI enrichment -----------------------------------------------------------------


def test_ai_unavailable_state():
    html = _render(_scan_result(scanner_findings=[_scanner_finding()]))
    assert "AI enrichment unavailable for this finding" in html
    assert "badge pending" in html
    # Must not fabricate any AI content when none exists.
    assert "AI Explanation" not in html
    assert "Exploit Path" not in html


def test_ai_enriched_finding():
    result = _scan_result(scanner_findings=[_scanner_finding()], ai_findings=[_ai_finding()])
    html = _render(result)
    assert "AI Explanation" in html
    assert "Exploit Path" in html
    assert "Impact" in html
    assert "Remediation" in html
    assert "AI enrichment unavailable for this finding" not in html


def test_confidence_rendering():
    result = _scan_result(scanner_findings=[_scanner_finding()], ai_findings=[_ai_finding(confidence_score=0.917)])
    html = _render(result)
    assert "92%" in html  # round(0.917 * 100) == 92
    assert "badge high" in html
    assert "confidence-fill" in html


def test_verification_rendering():
    result = _scan_result(
        scanner_findings=[_scanner_finding()],
        ai_findings=[_ai_finding(verification_status=VerificationStatus.REJECTED)],
    )
    html = _render(result)
    assert "badge rejected" in html
    assert "REJECTED" in html
    assert "rejected" in html.lower()  # textual status alongside the visual badge


def test_related_findings_linked_when_present_in_scan():
    scanner_findings = [
        _scanner_finding(finding_id="SENT-002"),
        _scanner_finding(finding_id="SENT-005"),
        _scanner_finding(finding_id="SENT-007"),
    ]
    ai_findings = [_ai_finding(finding_id="SENT-002", related_findings=["SENT-005", "SENT-007"])]
    html = _render(_scan_result(scanner_findings=scanner_findings, ai_findings=ai_findings))
    assert 'href="#finding-SENT-005"' in html
    assert 'href="#finding-SENT-007"' in html


def test_related_finding_not_in_scan_shown_as_text_not_invented_link():
    result = _scan_result(
        scanner_findings=[_scanner_finding(finding_id="SENT-001")],
        ai_findings=[_ai_finding(finding_id="SENT-001", related_findings=["SENT-999"])],
    )
    html = _render(result)
    assert "SENT-999" in html
    assert 'href="#finding-SENT-999"' not in html  # no fabricated link to a nonexistent finding


def test_references_rendered():
    result = _scan_result(
        scanner_findings=[_scanner_finding()],
        ai_findings=[_ai_finding(references=["CWE-89", "https://owasp.org/www-community/attacks/SQL_Injection"])],
    )
    html = _render(result)
    assert "CWE-89" in html
    assert 'href="https://owasp.org/www-community/attacks/SQL_Injection"' in html
    assert 'target="_blank"' in html
    assert 'rel="noopener noreferrer"' in html


# --- evidence / code -----------------------------------------------------------------


def test_raw_evidence_rendered_in_pre_block():
    finding = _scanner_finding(raw_evidence="obj = pickle.loads(cache.get(key))")
    html = _render(_scan_result(scanner_findings=[finding]))
    assert "<pre><code>obj = pickle.loads(cache.get(key))</code></pre>" in html


def test_patch_suggestion_rendered_in_pre_block():
    ai = _ai_finding(patch_suggestion="-old_line\n+new_line")
    html = _render(_scan_result(scanner_findings=[_scanner_finding()], ai_findings=[ai]))
    assert "<pre><code>-old_line\n+new_line</code></pre>" in html


# --- unicode -----------------------------------------------------------------


def test_unicode_content_preserved():
    finding = _scanner_finding(message="Ce message contient des caractères non-ASCII: 中文 ✓")
    html = _render(_scan_result(scanner_findings=[finding]))
    assert finding.message in html


# --- security: injection -----------------------------------------------------------------


def test_script_tag_injection_is_escaped_not_executed():
    payload = "<script>alert(1)</script>"
    finding = _scanner_finding(message=payload, raw_evidence=payload)
    html = _render(_scan_result(scanner_findings=[finding]))
    assert payload not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html


def test_html_tag_injection_is_escaped():
    payload = "<img src=x onerror=alert(1)>"
    finding = _scanner_finding(category=payload)
    html = _render(_scan_result(scanner_findings=[finding]))
    assert payload not in html
    assert "&lt;img" in html


def test_ai_content_injection_is_escaped():
    result = _scan_result(
        scanner_findings=[_scanner_finding()],
        ai_findings=[
            _ai_finding(
                explanation="<script>steal(document.cookie)</script>",
                exploit_path="<b>bold exploit</b>",
                remediation="<img src=x onerror=alert(2)>",
                patch_suggestion="<script>evil()</script>",
            )
        ],
    )
    html = _render(result)
    for payload in (
        "<script>steal(document.cookie)</script>",
        "<b>bold exploit</b>",
        "<img src=x onerror=alert(2)>",
        "<script>evil()</script>",
    ):
        assert payload not in html


def test_quotes_and_special_characters_escaped():
    finding = _scanner_finding(message='He said "hello" & <b>bold</b> & used \'quotes\'')
    html = _render(_scan_result(scanner_findings=[finding]))
    assert '"hello"' not in html or "&#34;hello&#34;" in html
    assert "<b>bold</b>" not in html


def test_markdown_syntax_rendered_as_literal_text():
    payload = "# Heading\n**bold** and `code`"
    finding = _scanner_finding(raw_evidence=payload)
    html = _render(_scan_result(scanner_findings=[finding]))
    assert "<pre><code># Heading\n**bold** and `code`</code></pre>" in html


# --- determinism / anchors -----------------------------------------------------------------


def test_deterministic_finding_ordering():
    findings = [
        _scanner_finding(finding_id="SENT-003"),
        _scanner_finding(finding_id="SENT-001"),
        _scanner_finding(finding_id="SENT-002"),
    ]
    html = _render(_scan_result(scanner_findings=findings))
    positions = [html.index(f'id="finding-SENT-{i:03d}"') for i in (1, 2, 3)]
    assert positions == sorted(positions)


def test_deterministic_output_for_same_input():
    result = _scan_result(scanner_findings=[_scanner_finding()])
    stats = calculate_statistics(result)
    assert to_html(result, stats) == to_html(result, stats)


def test_stable_finding_anchors():
    html = _render(_scan_result(scanner_findings=[_scanner_finding(finding_id="SENT-042")]))
    assert 'id="finding-SENT-042"' in html
    assert 'href="#finding-SENT-042"' in html


# --- offline / self-contained -----------------------------------------------------------------


def test_no_external_resource_dependency():
    html = _render(_scan_result(scanner_findings=[_scanner_finding()]))
    assert "<script src=" not in html
    assert '<link rel="stylesheet"' not in html
    assert "fonts.googleapis.com" not in html
    assert "cdn." not in html.lower()
    assert "<style>" in html  # CSS is embedded, not linked


def test_no_javascript_in_report():
    html = _render(_scan_result(scanner_findings=[_scanner_finding()], ai_findings=[_ai_finding()]))
    assert "<script>" not in html
