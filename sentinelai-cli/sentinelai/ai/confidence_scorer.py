"""
Heuristic confidence scoring: combines retrieval relevance and evidence
strength into (confidence_score, confidence_label).

No dedicated signals model (no confidence_models.py): every signal used
here already exists on an existing type - RetrievedChunk.score
(retrieval relevance) and ScannerFinding.raw_evidence (evidence
strength) - and each is read/derived, not stored, by exactly one
function, this one. A model is justified when a shape crosses a
boundary with more than one producer or consumer (RetrievedChunk:
produced by Retriever, consumed by both prompt_builder.py and this
file); an intermediate "ConfidenceSignals" here would have exactly one
producer and one consumer, both inside this function - not a shared
type, just local computation.

Two signals only, deliberately not more:
- scanner_reliability (a per-scanner weight table) is rejected outright,
  not just omitted as a field - there is no real data to justify any
  particular set of weights, and inventing one would be exactly the
  speculation this design review was meant to catch.
- llm_self_rating does not exist anywhere in the pipeline: LLMResponse
  (ai/llm_response.py) deliberately has no self-rating field, for the
  same reason - LLM confidence self-reports are poorly calibrated.

Combined as an unweighted average of exactly these two signals, not a
tuned formula - there is no data here to justify any particular
weighting either. Written as sum(signals) / len(signals) over a list
rather than inline arithmetic, so a future justified third signal is a
one-line addition to the list, not a rewrite of the expression. This
keeps the *contract* (score_confidence(...) -> (float, ConfidenceLabel))
stable while the formula itself is free to evolve, per the project's
own requirement that confidence scoring can change without breaking
interfaces.

retrieval_relevance uses the maximum chunk score, not the average: one
strongly relevant match should not be diluted by weaker additional
matches. Clamped to >= 0.0 since RetrievedChunk.score is deliberately
unbounded (a negative cosine similarity should not subtract from
confidence). Reads get_settings() for the two threshold values -
already documented as this file's dependency in ai/config.py's own
docstring, not a new one.

No llm_response parameter: an earlier draft of this signature included
one, carried over from pipeline.py's (rejected) provisional interface,
but neither signal here reads it. Keeping an unused parameter "in case
the formula wants it later" would be the same speculation being
rejected for scanner_reliability above - dropped so the signature
reflects exactly what this heuristic actually uses.
"""
import logging

from sentinelai.contracts import ConfidenceLabel, ScannerFinding

from .config import get_settings
from .retrieval import RetrievedChunk


logger = logging.getLogger("sentinelai")


def _retrieval_relevance(retrieved_chunks: list[RetrievedChunk]) -> float:
    best = max((chunk.score for chunk in retrieved_chunks), default=0.0)
    return max(best, 0.0)


def _evidence_strength(finding: ScannerFinding) -> float:
    return 1.0 if finding.raw_evidence else 0.0


def score_confidence(
    finding: ScannerFinding,
    retrieved_chunks: list[RetrievedChunk],
) -> tuple[float, ConfidenceLabel]:
    signals = [_retrieval_relevance(retrieved_chunks), _evidence_strength(finding)]
    score = sum(signals) / len(signals)

    settings = get_settings()
    if score >= settings.confidence_high_threshold:
        label = ConfidenceLabel.HIGH
    elif score >= settings.confidence_medium_threshold:
        label = ConfidenceLabel.MEDIUM
    else:
        label = ConfidenceLabel.LOW

    logger.debug(
        "confidence: finding_id=%s score=%.3f label=%s signals=%d",
        finding.finding_id,
        score,
        label.value,
        len(signals),
    )
    return score, label
