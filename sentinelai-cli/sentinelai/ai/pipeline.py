"""
The public entry point: orchestrates the per-finding sequence
(retrieve -> build prompt -> reason -> score confidence -> verify ->
assemble) without implementing any stage itself.

Confidence scoring and verification are injected callables, not
imports, the same way generate/embed already are elsewhere in this
package - matching ai/confidence_scorer.py's
score_confidence(finding, retrieved_chunks) -> (float, ConfidenceLabel)
and ai/verifier.py's verify_finding(finding, llm_response) ->
VerificationStatus exactly.

AIEnrichedFinding.references is populated by collecting and
deduplicating RetrievedChunk.references across a finding's retrieved
chunks (_collect_references below) - references are KB-sourced, per
ai/llm_response.py's docstring, to avoid the LLM inventing URLs.
RetrievedChunk carries references through from its source
KnowledgeBaseEntry, so this file gets everything it needs from
retrieval's output alone, with no security_kb import and no second
lookup against the knowledge base. Deduplication matters because
nothing guarantees a retrieved chunk's source entry is unique
(security_kb/repository.py documents the same non-uniqueness
assumption), so two chunks could carry overlapping references.

No embedding-provider or JSON-parsing knowledge: retriever arrives
pre-constructed (this file imports Retriever only for the type hint,
never InMemoryRetriever/EmbeddingFn/security_kb), and generate's output
is already validated into LLMResponse by explain() before this file
ever sees it.

Failure handling is per-finding, not per-scan. A finding whose
enrichment fails - a malformed LLM response, a schema violation, a
transient error on one call - is logged at WARNING and skipped, and the
remaining findings are still enriched. Previously the first failure
re-raised and aborted the whole run, so one bad response discarded every
successful enrichment alongside it.

Skipped findings are omitted from the returned list rather than being
given a placeholder AIEnrichedFinding, and that is a deliberate choice
with three consequences, all of them the point:

- statistics/calculator.py's _enrichment_status() reports PARTIAL
  precisely when matched_ai_findings < total_findings, so omission is
  what makes PARTIAL reachable. A placeholder carrying the same
  finding_id would push matched back up to total and report AVAILABLE -
  the opposite of the truth.
- The failure count is already derivable from existing statistics
  (total_findings - matched_ai_findings) whenever the status is PARTIAL,
  so no new contract field is needed to express it.
- Placeholders would also have to invent a confidence_score, which
  _tally_ai_findings() averages into ConfidenceStatistics; a fabricated
  0.0 would silently corrupt the reported confidence distribution for
  every successful finding in the same scan.

The scanner finding itself is never lost - it stays in
ScanResult.scanner_findings and every renderer already has a path for a
scanner finding with no matching enrichment ("AI enrichment not yet
available for this finding"), which is the same, already-tested path
scanner-only mode uses.

Total failure remains an error: if every finding failed, the run raises
AIEnrichmentError rather than returning an empty list. Silently
completing with no enrichment would hide exactly the case the AI error
handling in sentinelai/main.py was built to surface - an unreachable
Ollama server or a misconfigured embedding model - reporting it as a
clean PROVIDER_ERROR instead of an exit-0 scan that merely happens to
contain no AI content. An empty input list is not a failure and returns
[] without raising.

repository_context defaults to RepositoryContext() when not supplied -
the one construction this file does perform, safe because
RepositoryContext is this module's own minimal, AI-layer-owned type,
not a provider choice. settings = get_settings() is read here, not in any
stage function, because the orchestrator is the correct layer for
configuration to enter the system - retrieval_top_k and
enable_verification are orchestration decisions (how much to retrieve,
whether to run a stage at all), not stage-internal logic.
"""
import logging
from typing import Callable, Optional

from sentinelai.contracts import AIEnrichedFinding, ConfidenceLabel, ScannerFinding, VerificationStatus
from sentinelai.core.errors import AIEnrichmentError

from .agents.explainer import LLMFn, explain
from .config import get_settings
from .llm_response import LLMResponse
from .prompt_builder import build_prompt
from .repository_context import RepositoryContext
from .retrieval import RetrievedChunk, Retriever, retrieve_context

# Silent by default (no handler configured here), same convention and
# logger name as sentinelai.main / sentinelai.providers.live_provider.
logger = logging.getLogger("sentinelai")

ConfidenceScorerFn = Callable[[ScannerFinding, list[RetrievedChunk]], tuple[float, ConfidenceLabel]]
VerifierFn = Callable[[ScannerFinding, LLMResponse], VerificationStatus]


def _collect_references(retrieved_chunks: list[RetrievedChunk]) -> list[str]:
    """Dedupe references across chunks, preserving first-seen order. Reads what retrieval already returned - no lookup."""
    seen: set[str] = set()
    references: list[str] = []
    for chunk in retrieved_chunks:
        for reference in chunk.references:
            if reference not in seen:
                seen.add(reference)
                references.append(reference)
    return references


def _assemble(
    finding: ScannerFinding,
    retrieved_chunks: list[RetrievedChunk],
    llm_response: LLMResponse,
    confidence_score: float,
    confidence_label: ConfidenceLabel,
    verification_status: VerificationStatus,
    repository_context: RepositoryContext,
) -> AIEnrichedFinding:
    return AIEnrichedFinding(
        finding_id=finding.finding_id,
        title=llm_response.title,
        severity=finding.severity,
        scanner_sources=[finding.scanner],
        evidence=finding.raw_evidence,
        repository_context=repository_context.summary,
        explanation=llm_response.explanation,
        exploit_path=llm_response.exploit_path,
        impact=llm_response.impact,
        remediation=llm_response.remediation,
        patch_suggestion=llm_response.patch_suggestion,
        confidence_score=confidence_score,
        confidence_label=confidence_label,
        verification_status=verification_status,
        references=_collect_references(retrieved_chunks),
    )


def enrich_findings(
    scanner_findings: list[ScannerFinding],
    retriever: Retriever,
    generate: LLMFn,
    score_confidence: ConfidenceScorerFn,
    verify_finding: VerifierFn,
    repository_context: Optional[RepositoryContext] = None,
) -> list[AIEnrichedFinding]:
    ctx = repository_context or RepositoryContext()
    settings = get_settings()
    logger.info(
        "AI pipeline: enriching %d findings (verification=%s)",
        len(scanner_findings),
        "enabled" if settings.enable_verification else "disabled",
    )

    results: list[AIEnrichedFinding] = []
    failures: list[tuple[str, Exception]] = []
    for finding in scanner_findings:
        try:
            chunks = retrieve_context(finding, ctx, retriever, settings.retrieval_top_k)
            prompt = build_prompt(finding, ctx, chunks)
            llm_response = explain(prompt, generate)
            confidence_score, confidence_label = score_confidence(finding, chunks)
            if settings.enable_verification:
                verification_status = verify_finding(finding, llm_response)
            else:
                verification_status = VerificationStatus.UNVERIFIED
            results.append(
                _assemble(finding, chunks, llm_response, confidence_score, confidence_label, verification_status, ctx)
            )
        except Exception as exc:
            # WARNING, not ERROR: the run continues and the outcome is reported.
            # With no logging handler configured (this project's default), Python's
            # logging.lastResort prints WARNING and above to stderr, so a skipped
            # finding is visible to an operator rather than silently absent.
            logger.warning("AI pipeline: enrichment failed on finding_id=%s: %s", finding.finding_id, exc)
            failures.append((finding.finding_id, exc))

    if failures and not results:
        first_finding_id, first_exc = failures[0]
        logger.error("AI pipeline: all %d findings failed enrichment", len(failures))
        raise AIEnrichmentError(
            f"all {len(failures)} finding(s) failed AI enrichment; "
            f"first failure on '{first_finding_id}': {first_exc}"
        ) from first_exc

    if failures:
        logger.warning(
            "AI pipeline: enriched %d of %d findings (%d failed)",
            len(results),
            len(scanner_findings),
            len(failures),
        )
    return results
