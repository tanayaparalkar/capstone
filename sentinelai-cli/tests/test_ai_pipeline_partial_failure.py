"""
Tests for per-finding failure handling in ai/pipeline.py.

enrich_findings() previously re-raised on the first failing finding, so a
single malformed LLM response discarded every successful enrichment in
the same scan. It now skips the failing finding and continues, while
still raising when *every* finding fails - so an unreachable Ollama
server is reported as an error rather than as an exit-0 scan that merely
happens to contain no AI content.

The tests below pin both halves of that, plus the downstream statistics
consequences that motivated omitting failed findings rather than giving
them placeholder AIEnrichedFindings: PARTIAL only becomes reachable when
matched_ai_findings < total_findings, and ConfidenceStatistics must be
computed over successful enrichments alone.

All stages are injected callables, so no Ollama server, no knowledge
base, and no network are involved.
"""
import json

import pytest

from sentinelai.ai.config import AISettings
from sentinelai.ai.confidence_scorer import score_confidence
from sentinelai.ai.pipeline import enrich_findings
from sentinelai.ai.retrieval import RetrievedChunk, Retriever
from sentinelai.ai.verifier import verify_finding
from sentinelai.contracts import (
    RepositoryInfo,
    ScanMetadata,
    ScanMode,
    ScanResult,
    ScannerFinding,
    Severity,
)
from sentinelai.core.errors import AIEnrichmentError
from sentinelai.statistics import calculate_statistics
from sentinelai.statistics.models import AIEnrichmentStatus

_VALID = {
    "title": "A finding",
    "explanation": "sql injection explanation.",
    "exploit_path": None,
    "impact": None,
    "remediation": "A sql injection remediation.",
    "patch_suggestion": None,
}


def _finding(finding_id: str) -> ScannerFinding:
    return ScannerFinding(
        finding_id=finding_id,
        scanner="semgrep",
        category="sql-injection",
        severity=Severity.HIGH,
        rule_id="dummy.rule",
        message="test finding",
        raw_evidence="query = 'SELECT ' + x",
    )


class _FixedRetriever(Retriever):
    def __init__(self, score: float = 1.0):
        self._score = score

    def retrieve(self, query, top_k):
        return [RetrievedChunk(kb_entry_id="KB-1", text="sql injection", score=self._score)]


def _generate_failing_on(*failing_ids: str):
    """An LLMFn that raises for the named findings and returns valid JSON otherwise.

    The prompt embeds `Finding: <id>`, so the finding under enrichment is
    identifiable without threading extra state through the pipeline.
    """

    def generate(prompt: str) -> str:
        for finding_id in failing_ids:
            if f"Finding: {finding_id}\n" in prompt or prompt.rstrip().endswith(f"Finding: {finding_id}"):
                raise ConnectionRefusedError(f"simulated failure for {finding_id}")
        return json.dumps(_VALID)

    return generate


def _settings(monkeypatch, **overrides):
    settings = AISettings(**overrides)
    monkeypatch.setattr("sentinelai.ai.pipeline.get_settings", lambda: settings)


def _scan_result(scanner_findings, ai_findings) -> ScanResult:
    return ScanResult(
        repository=RepositoryInfo(name="demo", path="/tmp/demo"),
        metadata=ScanMetadata(timestamp="2026-08-17T12:00:00Z", mode=ScanMode.STANDARD),
        scanner_findings=scanner_findings,
        ai_findings=ai_findings,
    )


# --- partial failure is tolerated --------------------------------------------------------------------------------


def test_one_failure_does_not_discard_the_other_findings(monkeypatch):
    _settings(monkeypatch)
    findings = [_finding("SENT-001"), _finding("SENT-002"), _finding("SENT-003")]

    results = enrich_findings(
        findings, _FixedRetriever(), _generate_failing_on("SENT-002"), score_confidence, verify_finding
    )

    assert [r.finding_id for r in results] == ["SENT-001", "SENT-003"]


def test_failed_finding_is_omitted_not_placeholdered(monkeypatch):
    _settings(monkeypatch)
    findings = [_finding("SENT-001"), _finding("SENT-002")]

    results = enrich_findings(
        findings, _FixedRetriever(), _generate_failing_on("SENT-002"), score_confidence, verify_finding
    )

    assert "SENT-002" not in {r.finding_id for r in results}
    assert len(results) == 1


def test_enrichment_continues_past_multiple_failures(monkeypatch):
    _settings(monkeypatch)
    findings = [_finding(f"SENT-00{i}") for i in range(1, 6)]

    results = enrich_findings(
        findings,
        _FixedRetriever(),
        _generate_failing_on("SENT-002", "SENT-004"),
        score_confidence,
        verify_finding,
    )

    assert [r.finding_id for r in results] == ["SENT-001", "SENT-003", "SENT-005"]


def test_failure_is_logged_as_a_warning_naming_the_finding(monkeypatch, caplog):
    import logging

    _settings(monkeypatch)
    findings = [_finding("SENT-001"), _finding("SENT-002")]

    with caplog.at_level(logging.WARNING, logger="sentinelai"):
        enrich_findings(
            findings, _FixedRetriever(), _generate_failing_on("SENT-002"), score_confidence, verify_finding
        )

    assert "SENT-002" in caplog.text
    assert "enriched 1 of 2 findings" in caplog.text


# --- total failure is still an error -------------------------------------------------------------------------------


def test_all_findings_failing_raises(monkeypatch):
    _settings(monkeypatch)
    findings = [_finding("SENT-001"), _finding("SENT-002")]

    with pytest.raises(AIEnrichmentError) as excinfo:
        enrich_findings(
            findings,
            _FixedRetriever(),
            _generate_failing_on("SENT-001", "SENT-002"),
            score_confidence,
            verify_finding,
        )

    message = str(excinfo.value)
    assert "all 2 finding(s) failed" in message
    assert "SENT-001" in message
    assert "simulated failure" in message  # the underlying reason survives


def test_total_failure_chains_the_original_exception(monkeypatch):
    _settings(monkeypatch)

    with pytest.raises(AIEnrichmentError) as excinfo:
        enrich_findings(
            [_finding("SENT-001")],
            _FixedRetriever(),
            _generate_failing_on("SENT-001"),
            score_confidence,
            verify_finding,
        )

    assert isinstance(excinfo.value.__cause__, ConnectionRefusedError)


def test_empty_input_returns_empty_without_raising(monkeypatch):
    _settings(monkeypatch)

    assert enrich_findings([], _FixedRetriever(), _generate_failing_on(), score_confidence, verify_finding) == []


# --- downstream statistics -----------------------------------------------------------------------------------------


def test_partial_failure_reports_partial_status(monkeypatch):
    # The reason failed findings are omitted rather than placeholdered.
    _settings(monkeypatch)
    findings = [_finding("SENT-001"), _finding("SENT-002"), _finding("SENT-003")]

    results = enrich_findings(
        findings, _FixedRetriever(), _generate_failing_on("SENT-002"), score_confidence, verify_finding
    )
    stats = calculate_statistics(_scan_result(findings, results))

    assert stats.ai_enrichment_status == AIEnrichmentStatus.PARTIAL
    assert stats.matched_ai_findings == 2
    assert stats.total_findings - stats.matched_ai_findings == 1  # the failure count, already derivable


def test_full_success_still_reports_available(monkeypatch):
    _settings(monkeypatch)
    findings = [_finding("SENT-001"), _finding("SENT-002")]

    results = enrich_findings(
        findings, _FixedRetriever(), _generate_failing_on(), score_confidence, verify_finding
    )
    stats = calculate_statistics(_scan_result(findings, results))

    assert stats.ai_enrichment_status == AIEnrichmentStatus.AVAILABLE
    assert stats.matched_ai_findings == 2


def test_confidence_statistics_exclude_failed_findings(monkeypatch):
    # A placeholder at confidence 0.0 would drag these values down; omission keeps
    # them computed over successful enrichments only.
    _settings(monkeypatch)
    findings = [_finding("SENT-001"), _finding("SENT-002")]

    results = enrich_findings(
        findings, _FixedRetriever(score=1.0), _generate_failing_on("SENT-002"), score_confidence, verify_finding
    )
    stats = calculate_statistics(_scan_result(findings, results))

    assert stats.confidence.enriched_count == 1
    assert stats.confidence.min_score == 1.0  # not diluted by a fabricated 0.0
    assert stats.confidence.average_score == 1.0
