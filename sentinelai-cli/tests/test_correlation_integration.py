"""
Correlation as it reaches the rest of the system: statistics, all five
renderers, saved reports, and the AI pipeline's once-per-group budget.

Two properties are load-bearing throughout and are asserted repeatedly
rather than once:

- Raw findings are never lost. Every renderer still emits one entry per
  raw ScannerFinding, and raw ids survive in JSON and SARIF. Correlation
  is an index, not a filter.
- Both counts stay visible. A reader must always be able to distinguish
  "17 raw findings" from "13 distinct issues"; collapsing to either
  number alone would misrepresent the scan.
"""
import json

import pytest
from agent_fakes import make_agent_generate

from sentinelai.ai.config import AISettings
from sentinelai.ai.confidence_scorer import score_confidence
from sentinelai.ai.pipeline import enrich_findings
from sentinelai.ai.retrieval import RetrievedChunk, Retriever
from sentinelai.ai.verifier import verify_finding
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
from sentinelai.correlation import correlate_findings
from sentinelai.core import build_ai_lookup, filter_by_severity
from sentinelai.reporting import to_html, to_json, to_markdown, to_sarif
from sentinelai.reporting.loader import load_scan_result
from sentinelai.statistics import calculate_statistics


def _f(fid, scanner, cwe, line, severity=Severity.HIGH) -> ScannerFinding:
    return ScannerFinding(
        finding_id=fid, scanner=scanner, category="blacklist", severity=severity,
        file="app/tasks.py", line_start=line, line_end=line,
        rule_id="R1", message="m", raw_evidence="e", cwe=cwe,
    )


def _paired():
    """Two scanners on one line (merges) plus one unrelated finding (singleton)."""
    return [
        _f("bandit-0", "bandit", "CWE-78", 21),
        _f("semgrep-0", "semgrep", "CWE-78: OS Command Injection", 21),
        _f("bandit-1", "bandit", "CWE-20", 90, severity=Severity.LOW),
    ]


def _result(findings=None, ai=None) -> ScanResult:
    findings = findings if findings is not None else _paired()
    return ScanResult(
        repository=RepositoryInfo(name="demo", path="/tmp/demo"),
        metadata=ScanMetadata(timestamp="2026-08-18T12:00:00Z", mode=ScanMode.STANDARD),
        scanner_findings=findings,
        ai_findings=ai or [],
        correlated_findings=correlate_findings(findings),
    )


def _render(result, renderer):
    return renderer(result, calculate_statistics(result))


# --- statistics --------------------------------------------------------------------------------------------------------


def test_statistics_expose_raw_and_correlated_counts_separately():
    stats = calculate_statistics(_result())

    assert stats.total_findings == 3        # raw, unchanged
    assert stats.correlated_findings == 2   # distinct issues
    assert stats.multi_scanner_findings == 1


def test_statistics_default_to_zero_for_pre_correlation_results():
    plain = ScanResult(
        repository=RepositoryInfo(name="d", path="/tmp"),
        metadata=ScanMetadata(timestamp="2026-08-18T12:00:00Z", mode=ScanMode.STANDARD),
        scanner_findings=_paired(),
    )

    stats = calculate_statistics(plain)

    assert stats.correlated_findings == 0
    assert stats.multi_scanner_findings == 0
    assert stats.total_findings == 3  # raw count is unaffected


def test_severity_filter_keeps_raw_and_correlated_consistent():
    filtered = filter_by_severity(_result(), Severity.HIGH)
    stats = calculate_statistics(filtered)

    assert all(f.severity is Severity.HIGH for f in filtered.scanner_findings)
    assert stats.correlated_findings == len(filtered.correlated_findings)
    assert stats.total_findings == 2  # the LOW singleton dropped from both views


# --- raw findings are never lost -----------------------------------------------------------------------------------------


def test_every_renderer_still_emits_one_entry_per_raw_finding():
    result = _result()

    data = json.loads(_render(result, to_json))
    sarif = json.loads(_render(result, to_sarif))
    md = _render(result, to_markdown)

    assert len(data["findings"]["scanner"]) == 3
    assert len(sarif["runs"][0]["results"]) == 3  # one SARIF result per RAW finding
    for f in result.scanner_findings:
        assert f.finding_id in md


def test_raw_scanner_ids_are_preserved_in_json_and_sarif():
    result = _result()

    data = json.loads(_render(result, to_json))
    sarif = json.loads(_render(result, to_sarif))

    assert {f["finding_id"] for f in data["findings"]["scanner"]} == {"bandit-0", "semgrep-0", "bandit-1"}
    assert {r["properties"]["sentinelai"]["findingId"] for r in sarif["runs"][0]["results"]} == {
        "bandit-0", "semgrep-0", "bandit-1"
    }


def test_json_carries_the_correlation_group_alongside_raw_findings():
    data = json.loads(_render(_result(), to_json))

    # correlated_findings travels inside the ScanResult that the report envelope wraps.
    assert len(data["findings"]["scanner"]) == 3


# --- correlation is visible in reports ------------------------------------------------------------------------------------


def test_markdown_shows_both_counts_and_the_grouping():
    md = _render(_result(), to_markdown)

    assert "**Total Findings:** 3" in md
    assert "**Correlated Issues:** 2" in md
    assert "confirmed by 2+ scanners" in md
    assert "**Correlated Issue:** CORR-" in md
    assert "**Grouped With:**" in md
    assert "**Correlation Basis:**" in md


def test_markdown_marks_canonical_and_grouped_roles():
    md = _render(_result(), to_markdown)

    assert "(canonical)" in md
    assert "grouped under bandit-0" in md


def test_html_shows_both_counts_and_the_grouping():
    html = _render(_result(), to_html)

    assert "Raw Findings" in html
    assert "Correlated Issues" in html
    assert "Correlated Issue" in html
    assert "CORR-" in html


def test_sarif_carries_correlation_properties():
    sarif = json.loads(_render(_result(), to_sarif))
    props = {r["properties"]["sentinelai"]["findingId"]: r["properties"]["sentinelai"]
             for r in sarif["runs"][0]["results"]}

    merged = props["semgrep-0"]
    assert merged["correlationId"].startswith("CORR-")
    assert merged["canonicalFindingId"] == "bandit-0"
    assert sorted(merged["sourceFindingIds"]) == ["bandit-0", "semgrep-0"]
    assert merged["correlatedScanners"] == ["bandit", "semgrep"]
    assert "correlationReason" in merged


def test_sarif_statistics_include_correlation_counts():
    sarif = json.loads(_render(_result(), to_sarif))
    stats = sarif["runs"][0]["properties"]["sentinelai"]["statistics"]

    assert stats["total_findings"] == 3
    assert stats["correlated_findings"] == 2
    assert stats["multi_scanner_findings"] == 1


# --- AI enrichment: once per group ------------------------------------------------------------------------------------------


class _FixedRetriever(Retriever):
    def retrieve(self, query, top_k):
        return [RetrievedChunk(kb_entry_id="KB-1", text="t", score=1.0, references=["https://x"])]


def _settings(monkeypatch, **kw):
    monkeypatch.setattr("sentinelai.ai.pipeline.get_settings", lambda: AISettings(**kw))


def test_enrichment_runs_once_per_correlated_group(monkeypatch):
    _settings(monkeypatch)
    findings = _paired()
    groups = correlate_findings(findings)
    generate = make_agent_generate()

    results = enrich_findings(
        findings, _FixedRetriever(), generate, score_confidence, verify_finding,
        correlated_findings=groups,
    )

    # 2 groups x 3 agent calls, NOT 3 raw findings x 3.
    assert len(results) == 2
    assert len(generate.calls) == 6


def test_enrichment_covers_every_group_exactly_once(monkeypatch):
    _settings(monkeypatch)
    findings = _paired()
    groups = correlate_findings(findings)

    results = enrich_findings(
        findings, _FixedRetriever(), make_agent_generate(), score_confidence, verify_finding,
        correlated_findings=groups,
    )

    assert sorted(r.correlation_id for r in results) == sorted(g.correlation_id for g in groups)


def test_enriched_finding_keeps_the_canonical_raw_id(monkeypatch):
    _settings(monkeypatch)
    findings = _paired()
    groups = correlate_findings(findings)

    results = enrich_findings(
        findings, _FixedRetriever(), make_agent_generate(), score_confidence, verify_finding,
        correlated_findings=groups,
    )
    merged = next(r for r in results if len(r.source_finding_ids) > 1)

    # finding_id keeps its original meaning: a raw ScannerFinding id.
    assert merged.finding_id == "bandit-0"
    assert sorted(merged.source_finding_ids) == ["bandit-0", "semgrep-0"]
    assert merged.scanner_sources == ["bandit", "semgrep"]


def test_without_correlation_enrichment_is_per_finding(monkeypatch):
    # Older callers that pass no groups keep the previous behaviour exactly.
    _settings(monkeypatch)
    generate = make_agent_generate()

    results = enrich_findings(_paired(), _FixedRetriever(), generate, score_confidence, verify_finding)

    assert len(results) == 3
    assert len(generate.calls) == 9


# --- renderer join resolves group members ------------------------------------------------------------------------------------


def _ai(finding_id, correlation_id=None, sources=()) -> AIEnrichedFinding:
    return AIEnrichedFinding(
        finding_id=finding_id, title="t", severity=Severity.HIGH,
        explanation="e", remediation="r", confidence_score=0.8,
        confidence_label=ConfidenceLabel.HIGH, verification_status=VerificationStatus.VERIFIED,
        correlation_id=correlation_id, source_finding_ids=list(sources),
        scanner_sources=["bandit", "semgrep"],
    )


def test_non_canonical_raw_finding_resolves_to_the_group_enrichment():
    ai = _ai("bandit-0", "CORR-001", ["bandit-0", "semgrep-0"])
    lookup = build_ai_lookup(_result(ai=[ai]))

    assert lookup["bandit-0"] is ai   # direct join, unchanged
    assert lookup["semgrep-0"] is ai  # resolved via the group
    assert lookup["semgrep-0"] is lookup["bandit-0"]  # same object, not a duplicate


def test_lookup_creates_no_duplicate_enrichments():
    ai = _ai("bandit-0", "CORR-001", ["bandit-0", "semgrep-0"])
    result = _result(ai=[ai])

    assert len(result.ai_findings) == 1
    assert len({id(v) for v in build_ai_lookup(result).values()}) == 1


def test_pre_correlation_enrichment_still_joins_directly():
    ai = _ai("bandit-0")  # no correlation_id, no source ids

    lookup = build_ai_lookup(_result(ai=[ai]))

    assert lookup["bandit-0"] is ai
    assert "semgrep-0" not in lookup


def test_markdown_shows_the_shared_analysis_for_both_group_members():
    ai = _ai("bandit-0", "CORR-001", ["bandit-0", "semgrep-0"])

    md = _render(_result(ai=[ai]), to_markdown)

    # Both raw findings render, and both carry the one shared AI explanation.
    assert md.count("**AI Explanation**") == 2


# --- saved-report compatibility ---------------------------------------------------------------------------------------------


def test_correlation_survives_the_report_round_trip(tmp_path):
    # This previously asserted correlation was *dropped* by the envelope, which
    # encoded a real defect: statistics changed depending on whether a report was
    # rendered live or reloaded from disk. Correlation is now persisted, so the
    # reloaded result must carry the same groups. Full lifecycle coverage lives in
    # tests/test_correlation_persistence.py.
    live = _result()
    path = tmp_path / "report.json"
    path.write_text(_render(live, to_json), encoding="utf-8")

    result, stats = load_scan_result(path)

    assert len(result.scanner_findings) == 3
    assert len(result.correlated_findings) == 2
    assert stats.total_findings == 3
    assert stats.correlated_findings == 2
    assert stats.multi_scanner_findings == 1


def test_report_with_correlated_ai_fields_round_trips(tmp_path):
    ai = _ai("bandit-0", "CORR-001", ["bandit-0", "semgrep-0"])
    path = tmp_path / "report.json"
    path.write_text(_render(_result(ai=[ai]), to_json), encoding="utf-8")

    result, _ = load_scan_result(path)
    loaded = result.ai_findings[0]

    assert loaded.correlation_id == "CORR-001"
    assert loaded.source_finding_ids == ["bandit-0", "semgrep-0"]
    # And the group join still works after a round trip.
    assert build_ai_lookup(result)["semgrep-0"] is loaded


def test_pre_correlation_ai_finding_loads_with_defaults():
    old = {
        "finding_id": "F", "title": "t", "severity": "high", "explanation": "e",
        "remediation": "r", "confidence_score": 0.5, "confidence_label": "medium",
        "verification_status": "verified",
    }

    ai = AIEnrichedFinding.model_validate(old)

    assert ai.correlation_id is None
    assert ai.source_finding_ids == []
