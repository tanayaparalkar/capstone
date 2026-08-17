"""
Rendering of the critic review signals across every output format.

grounding_verdict, supported_claims, and unsupported_claims were added to
AIEnrichedFinding as optional fields with defaults. Two properties need
holding down:

- when a critic ran, the verdict is visible in JSON, Markdown, HTML,
  SARIF, and the terminal detail view, and is presented as a *separate*
  signal from the deterministic verification_status rather than replacing
  it;
- when no critic ran - scanner-only mode, or a report written before the
  multi-agent pipeline existed - nothing is emitted, nothing crashes, and
  a saved report still loads.

The "old report" cases matter because reports are persisted artifacts:
`sentinelai report` is expected to render a JSON file produced by an
earlier version of the tool.
"""
import json

from rich.console import Console

from sentinelai.contracts import (
    AIEnrichedFinding,
    ConfidenceLabel,
    GroundingVerdict,
    RepositoryInfo,
    ScanMetadata,
    ScanMode,
    ScanResult,
    ScannerFinding,
    Severity,
    VerificationStatus,
)
from sentinelai.presentation.findings import render_finding_detail
from sentinelai.reporting import to_html, to_json, to_markdown, to_sarif
from sentinelai.reporting.loader import load_scan_result
from sentinelai.statistics import calculate_statistics

UNSUPPORTED = "That the parameter is attacker-controlled."
SUPPORTED = "The query is built by concatenation."


def _scanner_finding() -> ScannerFinding:
    return ScannerFinding(
        finding_id="SENT-001",
        scanner="semgrep",
        category="sql-injection",
        severity=Severity.HIGH,
        file="app/db.py",
        line_start=6,
        rule_id="python.sql.concat",
        message="Concatenated query.",
        raw_evidence="query = 'SELECT ' + x",
    )


def _ai_finding(**overrides) -> AIEnrichedFinding:
    base = dict(
        finding_id="SENT-001",
        title="sql-injection in db.py",
        severity=Severity.HIGH,
        explanation="Concatenation permits statement manipulation.",
        remediation="Use parameterized queries.",
        confidence_score=0.87,
        confidence_label=ConfidenceLabel.HIGH,
        verification_status=VerificationStatus.VERIFIED,
        grounding_verdict=GroundingVerdict.PARTIALLY_SUPPORTED,
        supported_claims=[SUPPORTED],
        unsupported_claims=[UNSUPPORTED],
    )
    base.update(overrides)
    return AIEnrichedFinding(**base)


def _result(ai_findings) -> ScanResult:
    return ScanResult(
        repository=RepositoryInfo(name="demo", path="/tmp/demo"),
        metadata=ScanMetadata(timestamp="2026-08-18T12:00:00Z", mode=ScanMode.STANDARD),
        scanner_findings=[_scanner_finding()],
        ai_findings=ai_findings,
    )


def _render(result, renderer):
    return renderer(result, calculate_statistics(result))


def _terminal(ai) -> str:
    console = Console(record=True, width=200)
    render_finding_detail(console, _scanner_finding(), ai)
    return console.export_text()


# --- with a critic verdict -------------------------------------------------------------------------------------------


def test_json_exposes_the_critic_fields():
    data = json.loads(_render(_result([_ai_finding()]), to_json))
    ai = data["findings"]["ai_enriched"][0]

    assert ai["grounding_verdict"] == "partially_supported"
    assert ai["supported_claims"] == [SUPPORTED]
    assert ai["unsupported_claims"] == [UNSUPPORTED]


def test_markdown_shows_grounding_and_unsupported_claims():
    md = _render(_result([_ai_finding()]), to_markdown)

    assert "**Grounding (critic):** partially_supported" in md
    assert UNSUPPORTED in md
    # Presented separately from the deterministic check, not instead of it.
    assert "**Verification:** verified" in md


def test_html_shows_grounding_and_unsupported_claims():
    html = _render(_result([_ai_finding()]), to_html)

    assert "Grounding (critic review)" in html
    assert "PARTIALLY_SUPPORTED" in html
    assert "Unsupported or Uncertain Claims" in html
    assert UNSUPPORTED in html


def test_html_styles_every_grounding_verdict_badge():
    # An unstyled badge renders white-on-nothing; each verdict needs a CSS rule.
    html = _render(_result([_ai_finding()]), to_html)
    for verdict in ("supported", "partially_supported", "insufficient_evidence"):
        assert f".badge.{verdict}" in html or f".badge.{verdict}," in html


def test_sarif_exposes_the_critic_fields():
    sarif = json.loads(_render(_result([_ai_finding()]), to_sarif))
    props = sarif["runs"][0]["results"][0]["properties"]["sentinelai"]["ai"]

    assert props["groundingVerdict"] == "partially_supported"
    assert props["supportedClaims"] == [SUPPORTED]
    assert props["unsupportedClaims"] == [UNSUPPORTED]
    # Distinct key from the deterministic signal.
    assert props["verificationStatus"] == "verified"


def test_terminal_detail_shows_grounding_and_claims():
    text = _terminal(_ai_finding())

    assert "Grounding: partially_supported" in text
    assert "Verification: verified" in text
    assert "Unsupported or Uncertain Claims" in text


# --- without a critic (scanner-only, or a pre-multi-agent report) -------------------------------------------------------


def test_renderers_omit_the_sections_when_no_critic_ran():
    ai = _ai_finding(grounding_verdict=None, supported_claims=[], unsupported_claims=[])
    result = _result([ai])

    md = _render(result, to_markdown)
    html = _render(result, to_html)
    sarif = json.loads(_render(result, to_sarif))
    props = sarif["runs"][0]["results"][0]["properties"]["sentinelai"]["ai"]

    assert "Grounding (critic)" not in md
    assert "Unsupported or Uncertain Claims" not in md
    assert "Grounding (critic review)" not in html
    assert "groundingVerdict" not in props
    assert "unsupportedClaims" not in props
    assert "Grounding:" not in _terminal(ai)


def test_empty_unsupported_claims_render_no_section_even_with_a_verdict():
    ai = _ai_finding(unsupported_claims=[], supported_claims=[])

    md = _render(_result([ai]), to_markdown)

    assert "**Grounding (critic):** partially_supported" in md
    assert "Unsupported or Uncertain Claims" not in md


def test_scanner_only_reports_still_render():
    result = _result([])

    assert "No findings" not in _render(result, to_markdown)
    assert _render(result, to_html)
    assert json.loads(_render(result, to_sarif))


# --- persisted reports -----------------------------------------------------------------------------------------------


def test_a_report_written_before_the_critic_existed_still_loads(tmp_path):
    # Exactly the shape `sentinelai report` must keep accepting.
    old = json.loads(_render(_result([_ai_finding()]), to_json))
    for field in ("grounding_verdict", "supported_claims", "unsupported_claims"):
        del old["findings"]["ai_enriched"][0][field]
    path = tmp_path / "old-report.json"
    path.write_text(json.dumps(old), encoding="utf-8")

    result, _ = load_scan_result(path)

    ai = result.ai_findings[0]
    assert ai.grounding_verdict is None
    assert ai.supported_claims == []
    assert ai.unsupported_claims == []


def test_a_report_with_critic_fields_round_trips(tmp_path):
    path = tmp_path / "report.json"
    path.write_text(_render(_result([_ai_finding()]), to_json), encoding="utf-8")

    result, _ = load_scan_result(path)

    ai = result.ai_findings[0]
    assert ai.grounding_verdict is GroundingVerdict.PARTIALLY_SUPPORTED
    assert ai.unsupported_claims == [UNSUPPORTED]
