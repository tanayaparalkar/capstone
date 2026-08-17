"""
The public entry point: orchestrates the per-finding multi-agent
sequence without implementing any stage itself.

    retrieve
      -> Evidence Analyst          [LLM call 1]
      -> Exploit + Remediation     [LLM call 2, two typed sections]
      -> Critic                    [LLM call 3]
      -> score confidence          (deterministic)
      -> deterministic verify      (deterministic)
      -> synthesize                (deterministic, no model call)

Exactly three generation calls per finding. The Exploit and Remediation
analysts share one request because they take identical inputs and neither
depends on the other; keeping them separate would double the pipeline's
dominant cost for no additional grounding. Report synthesis is ordinary
Python - see ai/agents/synthesizer.py for why nothing there may be
delegated to a model.

The stage order is a data dependency: each stage receives the previous
stages' *validated* typed output, so a claim is always traceable to the
agent that made it and to the evidence that agent was shown.

Confidence scoring and verification are injected callables, not
imports, the same way generate/embed already are elsewhere in this
package - matching ai/confidence_scorer.py's
score_confidence(finding, retrieved_chunks) -> (float, ConfidenceLabel)
and ai/verifier.py's verify_finding(finding, llm_response) ->
VerificationStatus exactly.

Assembly of AIEnrichedFinding - including KB-sourced references and the
scanner-owned fields - belongs to ai/agents/synthesizer.py, not here.
This file orchestrates; it does not build the contract.

No embedding-provider or JSON-parsing knowledge: retriever arrives
pre-constructed (this file imports Retriever only for the type hint,
never InMemoryRetriever/EmbeddingFn/security_kb), and each agent's
output is already validated into its typed schema by run_agent() before
this file ever sees it. LLMResponse survives here only as the parameter
type of VerifierFn, since ai/verifier.py's deterministic check still
consumes that shape - ai/agents/synthesizer.py's build_verifier_input()
adapts the agent outputs into it.

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

from sentinelai.contracts import (
    AIEnrichedFinding,
    ConfidenceLabel,
    CorrelatedFinding,
    ScannerFinding,
    VerificationStatus,
)
from sentinelai.core.errors import AIEnrichmentError

from .agents.critic import critique
from .agents.evidence_analyst import assess_evidence
from .agents.exploit_remediation_analyst import assess_exploit_and_remediation
from .agents.explainer import LLMFn
from .agents.synthesizer import build_verifier_input, synthesize
from .config import get_settings
from .llm_response import LLMResponse
from .repository_context import RepositoryContext
from .retrieval import RetrievedChunk, Retriever, retrieve_context

# Silent by default (no handler configured here), same convention and
# logger name as sentinelai.main / sentinelai.providers.live_provider.
logger = logging.getLogger("sentinelai")

ConfidenceScorerFn = Callable[[ScannerFinding, list[RetrievedChunk]], tuple[float, ConfidenceLabel]]
VerifierFn = Callable[[ScannerFinding, LLMResponse], VerificationStatus]


def _enrich_one(
    finding: ScannerFinding,
    ctx: RepositoryContext,
    retriever: Retriever,
    generate: LLMFn,
    score_confidence: ConfidenceScorerFn,
    verify_finding: VerifierFn,
    settings,
) -> AIEnrichedFinding:
    """Run the three agent stages plus deterministic synthesis for one finding.

    Stage order is a real data dependency, not a convention: the exploit and
    remediation analysts receive the Evidence Analyst's validated output, and the
    critic reviews both prior outputs. Exactly three generate() calls happen
    here - one per stage - and synthesis makes none.
    """
    chunks = retrieve_context(finding, ctx, retriever, settings.retrieval_top_k)

    # LLM call 1 of 3.
    evidence = assess_evidence(finding, ctx, chunks, generate)
    # LLM call 2 of 3 - both analysts in one request.
    assessment = assess_exploit_and_remediation(finding, evidence, ctx, chunks, generate)
    # LLM call 3 of 3.
    critic = critique(finding, evidence, assessment, ctx, chunks, generate)

    # Deterministic from here down - no further model calls.
    confidence_score, confidence_label = score_confidence(finding, chunks)
    if settings.enable_verification:
        verification_status = verify_finding(finding, build_verifier_input(evidence, assessment))
    else:
        verification_status = VerificationStatus.UNVERIFIED

    return synthesize(
        finding=finding,
        evidence=evidence,
        assessment=assessment,
        critic=critic,
        retrieved_chunks=chunks,
        confidence_score=confidence_score,
        confidence_label=confidence_label,
        verification_status=verification_status,
        repository_context=ctx,
    )


def enrich_findings(
    scanner_findings: list[ScannerFinding],
    retriever: Retriever,
    generate: LLMFn,
    score_confidence: ConfidenceScorerFn,
    verify_finding: VerifierFn,
    repository_context: Optional[RepositoryContext] = None,
    correlated_findings: Optional[list[CorrelatedFinding]] = None,
) -> list[AIEnrichedFinding]:
    ctx = repository_context or RepositoryContext()

    # Enrich once per correlated issue, not once per raw scanner hit. Two scanners
    # reporting one defect is one thing to explain, and paying three model calls
    # per duplicate would be waste, not thoroughness. When no correlation is
    # supplied - an older caller, or a provider that predates it - every finding
    # is its own group, which reproduces the previous per-finding behaviour exactly.
    groups = correlated_findings if correlated_findings is not None else []
    canonical_ids = {g.canonical_finding_id for g in groups}
    group_by_canonical = {g.canonical_finding_id: g for g in groups}
    if groups:
        scanner_findings = [f for f in scanner_findings if f.finding_id in canonical_ids]
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
            enriched = _enrich_one(
                finding, ctx, retriever, generate, score_confidence, verify_finding, settings
            )
            group = group_by_canonical.get(finding.finding_id)
            if group is not None:
                # finding_id deliberately stays the canonical *raw* id so existing
                # raw-finding -> AI joins keep working; the group view is additive.
                enriched = enriched.model_copy(
                    update={
                        "correlation_id": group.correlation_id,
                        "source_finding_ids": list(group.source_finding_ids),
                        "scanner_sources": list(group.scanners),
                    }
                )
            results.append(enriched)
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
