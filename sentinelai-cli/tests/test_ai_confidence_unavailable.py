"""
Regression tests for unavailable confidence scoring when retrieval is unavailable
or returns no chunks.

Covers:
1. Unset embedding model (_NoRetrieval) or empty retrieved_chunks produces
   (None, None) rather than a fabricated 0.5 / MEDIUM (or 0.0 / LOW) score.
2. Non-empty retrieved_chunks preserve exact existing heuristic scores and labels.
3. JSON (including round-trip load), Markdown, HTML, SARIF, and terminal renderers
   never emit a fabricated numeric confidence when confidence is unavailable.
4. Scanner findings, AI narrative fields, deterministic verification, critic
   grounding signals, and AIEnrichmentStatus remain unchanged apart from the
   unavailable-confidence representation.
"""
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from agent_fakes import make_agent_generate
from pydantic import ValidationError
from rich.console import Console

from sentinelai.ai.config import AISettings
from sentinelai.ai.confidence_scorer import score_confidence
from sentinelai.ai.factory import create_retriever_for_scan
from sentinelai.ai.pipeline import enrich_findings
from sentinelai.ai.repository_context import RepositoryContext
from sentinelai.ai.retrieval import RetrievedChunk, Retriever
from sentinelai.ai.verifier import verify_finding
from sentinelai.contracts import (
    AIEnrichedFinding,
    ConfidenceLabel,
    RepositoryInfo,
    ScanMetadata,
    ScanMode,
    ScannerFinding,
    ScanResult,
    Severity,
    VerificationStatus,
)
from sentinelai.presentation import render_finding_detail, render_findings_table, render_summary
from sentinelai.presentation.narrator import _confidence_badge, render_vulnerability_narration
from sentinelai.reporting import to_html, to_json, to_markdown, to_sarif
from sentinelai.reporting.loader import load_scan_result
from sentinelai.statistics import AIEnrichmentStatus, calculate_statistics


def _finding(
    finding_id: str = "SENT-001",
    raw_evidence: str | None = "cursor.execute('SELECT * FROM users WHERE id=' + uid)",
    **overrides,
) -> ScannerFinding:
    base = dict(
        finding_id=finding_id,
        scanner="semgrep",
        category="sql-injection",
        severity=Severity.HIGH,
        file="app/db.py",
        line_start=12,
        rule_id="python.sql.concat",
        message="User input concatenated into SQL query.",
        raw_evidence=raw_evidence,
        cwe="CWE-89",
    )
    base.update(overrides)
    return ScannerFinding(**base)


class _EmptyRetriever(Retriever):
    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        return []


class _StaticRetriever(Retriever):
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self._chunks = list(chunks)

    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        return list(self._chunks[:top_k])


# --- 1. Unavailable retrieval -> (None, None) & contract pairing -----------------


@pytest.mark.parametrize(
    "raw_evidence",
    [
        "cursor.execute('SELECT * FROM users WHERE id=' + uid)",
        None,
    ],
)
def test_score_confidence_returns_none_when_no_chunks_retrieved(raw_evidence):
    finding = _finding(raw_evidence=raw_evidence)

    score, label = score_confidence(finding, [])

    assert score is None
    assert label is None


def test_no_embedding_model_retriever_produces_unavailable_confidence(monkeypatch):
    monkeypatch.setattr(
        "sentinelai.ai.factory.get_settings",
        lambda: AISettings(embedding_model=None),
    )
    monkeypatch.setattr(
        "sentinelai.ai.pipeline.get_settings",
        lambda: AISettings(embedding_model=None),
    )

    retriever = create_retriever_for_scan()
    finding = _finding()
    chunks = retriever.retrieve("sql-injection", top_k=3)
    assert chunks == []

    enriched = enrich_findings(
        [finding],
        retriever,
        make_agent_generate(),
        score_confidence,
        verify_finding,
    )

    assert len(enriched) == 1
    assert enriched[0].confidence_score is None
    assert enriched[0].confidence_label is None


def test_contract_enforces_confidence_score_and_label_pairing():
    base_kwargs = dict(
        finding_id="SENT-001",
        title="SQL injection via string concatenation",
        severity=Severity.HIGH,
        scanner_sources=["semgrep"],
        explanation="Untrusted input reaches SQL query.",
        remediation="Use parameterized queries.",
        verification_status=VerificationStatus.VERIFIED,
    )

    # Both omitted -> defaults to (None, None)
    unscored = AIEnrichedFinding(**base_kwargs)
    assert unscored.confidence_score is None
    assert unscored.confidence_label is None

    # Score without label -> rejected
    with pytest.raises(ValidationError, match="confidence_score and confidence_label"):
        AIEnrichedFinding(**base_kwargs, confidence_score=0.5, confidence_label=None)

    # Label without score -> rejected
    with pytest.raises(ValidationError, match="confidence_score and confidence_label"):
        AIEnrichedFinding(
            **base_kwargs,
            confidence_score=None,
            confidence_label=ConfidenceLabel.MEDIUM,
        )


# --- 2. Available retrieval -> unchanged scoring ---------------------------------


@pytest.mark.parametrize(
    "chunk_scores,raw_evidence,expected_score,expected_label",
    [
        ([0.90], "x = 1", 0.95, ConfidenceLabel.HIGH),
        ([0.20, 0.80], "x = 1", 0.90, ConfidenceLabel.HIGH),
        ([0.30], "x = 1", 0.65, ConfidenceLabel.MEDIUM),
        ([-0.25], "x = 1", 0.50, ConfidenceLabel.MEDIUM),
        ([0.40], None, 0.20, ConfidenceLabel.LOW),
        ([-0.10], None, 0.00, ConfidenceLabel.LOW),
    ],
)
def test_score_confidence_unchanged_when_chunks_present(
    monkeypatch, chunk_scores, raw_evidence, expected_score, expected_label
):
    monkeypatch.setattr("sentinelai.ai.confidence_scorer.get_settings", lambda: AISettings())
    chunks = [
        RetrievedChunk(kb_entry_id=f"KB-{idx}", text="kb entry", score=s, references=[])
        for idx, s in enumerate(chunk_scores)
    ]
    finding = _finding(raw_evidence=raw_evidence)

    score, label = score_confidence(finding, chunks)

    assert score == pytest.approx(expected_score)
    assert label == expected_label


# --- 3. JSON, Markdown, HTML, SARIF, and terminal outputs ------------------------


def _build_unscored_scan_result(monkeypatch) -> tuple[ScanResult, object]:
    monkeypatch.setattr("sentinelai.ai.pipeline.get_settings", lambda: AISettings(embedding_model=None))
    finding = _finding()
    enriched = enrich_findings(
        [finding],
        _EmptyRetriever(),
        make_agent_generate(),
        score_confidence,
        verify_finding,
        repository_context=RepositoryContext(summary="Flask app with raw SQL helper"),
    )
    result = ScanResult(
        repository=RepositoryInfo(name="demo-repo", path="/tmp/demo-repo"),
        metadata=ScanMetadata(
            timestamp=datetime(2026, 8, 8, 12, 0, 0, tzinfo=timezone.utc),
            mode=ScanMode.STANDARD,
            duration_seconds=1.25,
            ai_provider="ollama",
            ai_model="qwen2.5-coder:7b",
        ),
        scanner_findings=[finding],
        ai_findings=enriched,
    )
    stats = calculate_statistics(result)
    return result, stats


def test_json_report_and_loader_round_trip_with_unavailable_confidence(monkeypatch, tmp_path: Path):
    result, stats = _build_unscored_scan_result(monkeypatch)

    assert stats.ai_enrichment_status == AIEnrichmentStatus.AVAILABLE
    assert stats.confidence.enriched_count == 1
    assert stats.confidence.high_count == 0
    assert stats.confidence.medium_count == 0
    assert stats.confidence.low_count == 0
    assert stats.confidence.average_score is None
    assert stats.confidence.min_score is None
    assert stats.confidence.max_score is None

    raw_json = to_json(result, stats)
    parsed = json.loads(raw_json)
    ai_entry = parsed["findings"]["ai_enriched"][0]
    assert ai_entry["confidence_score"] is None
    assert ai_entry["confidence_label"] is None
    assert parsed["statistics"]["confidence"]["average_score"] is None
    assert parsed["statistics"]["confidence"]["medium_count"] == 0

    report_path = tmp_path / "report.json"
    report_path.write_text(raw_json, encoding="utf-8")
    loaded_result, loaded_stats = load_scan_result(report_path)
    assert loaded_result.ai_findings[0].confidence_score is None
    assert loaded_result.ai_findings[0].confidence_label is None
    assert loaded_stats.confidence == stats.confidence


def test_markdown_and_html_reports_do_not_fabricate_numeric_confidence(monkeypatch):
    result, stats = _build_unscored_scan_result(monkeypatch)

    md = to_markdown(result, stats)
    assert "| SENT-001 | HIGH | semgrep | sql-injection | app/db.py:12 | `python.sql.concat` | unavailable |" in md
    assert "- **Confidence:** unavailable" in md
    assert "0.50" not in md
    assert "MEDIUM (0.50)" not in md

    html = to_html(result, stats)
    assert '<span class="badge unavailable">UNAVAILABLE</span>' in html
    assert '<div class="confidence-meter">' not in html
    assert '<div class="confidence-fill"' not in html
    assert "50%" not in html


def test_sarif_omits_unavailable_confidence_and_conforms_to_schema(monkeypatch):
    result, stats = _build_unscored_scan_result(monkeypatch)

    raw_sarif = to_sarif(result, stats)
    sarif_doc = json.loads(raw_sarif)
    ai_props = sarif_doc["runs"][0]["results"][0]["properties"]["sentinelai"]["ai"]
    assert "confidenceScore" not in ai_props
    assert "confidenceLabel" not in ai_props
    assert ai_props["verificationStatus"] == result.ai_findings[0].verification_status.value
    assert ai_props["explanation"] == result.ai_findings[0].explanation

    jsonschema = pytest.importorskip("jsonschema", reason="dev-only dependency")
    schema_path = Path(__file__).parent / "data" / "sarif-schema-2.1.0.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.validate(instance=sarif_doc, schema=schema)


def test_terminal_renderers_display_unavailable_without_crashing(monkeypatch):
    result, stats = _build_unscored_scan_result(monkeypatch)
    finding = result.scanner_findings[0]
    ai = result.ai_findings[0]

    assert _confidence_badge(ai) == "[dim]UNAVAILABLE[/dim]"

    buf = io.StringIO()
    console = Console(file=buf, width=120, force_terminal=False, color_system=None)
    render_summary(console, stats, metadata=result.metadata)
    render_findings_table(console, result)
    render_finding_detail(console, finding, ai)
    render_vulnerability_narration(console, finding, ai, patch=None)

    rendered = buf.getvalue()
    assert "UNAVAILABLE" in rendered
    assert "Confidence: UNAVAILABLE" in rendered
    assert "0.50" not in rendered
    assert "50%" not in rendered


# --- 4. Scanner findings & AI enrichment unchanged apart from confidence ---------


def test_enrichment_fields_unchanged_except_confidence_when_retrieval_unavailable(monkeypatch):
    monkeypatch.setattr("sentinelai.ai.pipeline.get_settings", lambda: AISettings())
    monkeypatch.setattr("sentinelai.ai.confidence_scorer.get_settings", lambda: AISettings())

    finding = _finding()
    repo_ctx = RepositoryContext(summary="Flask app")

    unscored = enrich_findings(
        [finding],
        _EmptyRetriever(),
        make_agent_generate(),
        score_confidence,
        verify_finding,
        repository_context=repo_ctx,
    )[0]

    scored = enrich_findings(
        [finding],
        _StaticRetriever([RetrievedChunk(kb_entry_id="KB-1", text="sql injection", score=0.8, references=[])]),
        make_agent_generate(),
        score_confidence,
        verify_finding,
        repository_context=repo_ctx,
    )[0]

    assert unscored.confidence_score is None
    assert unscored.confidence_label is None
    assert scored.confidence_score == pytest.approx(0.90)
    assert scored.confidence_label == ConfidenceLabel.HIGH

    assert unscored.model_dump(exclude={"confidence_score", "confidence_label"}) == scored.model_dump(
        exclude={"confidence_score", "confidence_label"}
    )


def test_mixed_scored_and_unscored_statistics_only_average_scored_findings():
    f1 = _finding("SENT-001")
    f2 = _finding("SENT-002")
    ai_scored = AIEnrichedFinding(
        finding_id="SENT-001",
        title="Scored SQLi",
        severity=Severity.HIGH,
        scanner_sources=["semgrep"],
        explanation="Parameterized query missing.",
        remediation="Use bind parameters.",
        confidence_score=0.90,
        confidence_label=ConfidenceLabel.HIGH,
        verification_status=VerificationStatus.VERIFIED,
    )
    ai_unscored = AIEnrichedFinding(
        finding_id="SENT-002",
        title="Unscored SQLi",
        severity=Severity.HIGH,
        scanner_sources=["semgrep"],
        explanation="Parameterized query missing.",
        remediation="Use bind parameters.",
        confidence_score=None,
        confidence_label=None,
        verification_status=VerificationStatus.VERIFIED,
    )
    result = ScanResult(
        repository=RepositoryInfo(name="demo", path="/tmp/demo"),
        metadata=ScanMetadata(
            timestamp=datetime(2026, 8, 8, 12, 0, 0, tzinfo=timezone.utc),
            mode=ScanMode.STANDARD,
        ),
        scanner_findings=[f1, f2],
        ai_findings=[ai_scored, ai_unscored],
    )

    stats = calculate_statistics(result)
    assert stats.ai_enrichment_status == AIEnrichmentStatus.AVAILABLE
    assert stats.matched_ai_findings == 2
    assert stats.confidence.enriched_count == 2
    assert stats.confidence.high_count == 1
    assert stats.confidence.medium_count == 0
    assert stats.confidence.low_count == 0
    assert stats.confidence.average_score == pytest.approx(0.90)
    assert stats.confidence.min_score == pytest.approx(0.90)
    assert stats.confidence.max_score == pytest.approx(0.90)
