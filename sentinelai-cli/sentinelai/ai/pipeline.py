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

No exception handling: every stage's failure propagates out of
enrich_findings() uncaught. Deciding what an acceptable partial failure
looks like, what to log, or what a caller should be told is a policy
decision "orchestrator only, no business logic" leaves to whoever calls
this function - and this file has no logging of its own to make a
swallowed failure visible in the first place.

repository_context defaults to RepositoryContext() when not supplied -
the one construction this file does perform, safe because
RepositoryContext is this module's own interim placeholder type, not a
provider choice. settings = get_settings() is read here, not in any
stage function, because the orchestrator is the correct layer for
configuration to enter the system - retrieval_top_k and
enable_verification are orchestration decisions (how much to retrieve,
whether to run a stage at all), not stage-internal logic.
"""
from typing import Callable, Optional

from sentinelai.contracts import AIEnrichedFinding, ConfidenceLabel, ScannerFinding, VerificationStatus

from .agents.explainer import LLMFn, explain
from .config import get_settings
from .llm_response import LLMResponse
from .prompt_builder import build_prompt
from .repository_context import RepositoryContext
from .retrieval import RetrievedChunk, Retriever, retrieve_context

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

    results = []
    for finding in scanner_findings:
        chunks = retrieve_context(finding, ctx, retriever, settings.retrieval_top_k)
        prompt = build_prompt(finding, ctx, chunks)
        llm_response = explain(prompt, generate)
        confidence_score, confidence_label = score_confidence(finding, chunks)
        if settings.enable_verification:
            verification_status = verify_finding(finding, llm_response)
        else:
            verification_status = VerificationStatus.UNVERIFIED
        results.append(_assemble(finding, chunks, llm_response, confidence_score, confidence_label, verification_status, ctx))
    return results
