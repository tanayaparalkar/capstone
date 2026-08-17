"""
Report synthesis - deterministic Python, zero LLM calls.

Assembles the three validated agent outputs into the existing
AIEnrichedFinding contract. Every value here is either copied from an
already-validated object or from the ScannerFinding; nothing is
generated, inferred, or asked of a model. Running this twice on the same
inputs produces identical output.

Provenance rules this module exists to enforce:

- Scanner-owned fields come from the ScannerFinding and are never
  overwritten by agent output: severity, evidence (raw_evidence), and
  scanner_sources. An agent that disagrees with the scanner's severity
  does not get to change it.
- references come from the retrieved knowledge base, never from model
  text, so a fabricated URL cannot reach a report.
- confidence_score/confidence_label and verification_status arrive
  already computed by ai/confidence_scorer.py and ai/verifier.py. This
  module passes them through untouched.
- grounding_verdict carries the critic's judgement and NOTHING else.
  verification_status is never derived from it and it is never derived
  from verification_status. They are independent signals produced by
  different mechanisms (one deterministic and offline, one a model
  review) and are allowed to disagree; collapsing them would destroy the
  information that disagreement carries.
- A finding is never invented: synthesize() requires a ScannerFinding and
  copies its finding_id, so there is no path by which the AI layer can
  emit an enrichment for something no scanner reported.

Agent output that has no dedicated field on the frozen contract is folded
into the nearest text field under an explicit label rather than being
dropped - `limitations` onto the explanation, `required_assumptions` onto
the exploit path, `validation_steps` onto the remediation. Labelling
matters: an assumption appended without a marker would read as an
observation, which is exactly the distinction the schemas were built to
preserve.

build_verifier_input() adapts the agent outputs into the LLMResponse
shape ai/verifier.py already consumes, so the deterministic verifier
keeps working unchanged against the multi-agent pipeline. Changing the
verifier's signature to accept agent types would have been the
alternative, and would have meant editing deterministic logic that is
deliberately frozen.
"""
from typing import Optional

from sentinelai.contracts import (
    AIEnrichedFinding,
    ConfidenceLabel,
    ScannerFinding,
    VerificationStatus,
)

from ..llm_response import LLMResponse
from ..repository_context import RepositoryContext
from ..retrieval import RetrievedChunk
from .schemas import CriticAssessment, EvidenceAssessment, ExploitRemediationAssessment

_LIMITATIONS_LABEL = "Limitations of the available evidence:"
_ASSUMPTIONS_LABEL = "Assumptions required for this exploit path:"
_VALIDATION_LABEL = "Validation steps:"


def _bullets(label: str, items: list[str]) -> str:
    return "\n".join([label, *(f"- {item}" for item in items)])


def _collect_references(retrieved_chunks: list[RetrievedChunk]) -> list[str]:
    """Dedupe references across chunks, preserving first-seen order. KB-sourced only - never model text."""
    seen: set[str] = set()
    references: list[str] = []
    for chunk in retrieved_chunks:
        for reference in chunk.references:
            if reference not in seen:
                seen.add(reference)
                references.append(reference)
    return references


def _build_explanation(evidence: EvidenceAssessment) -> str:
    if not evidence.limitations:
        return evidence.interpretation
    return f"{evidence.interpretation}\n\n{_bullets(_LIMITATIONS_LABEL, evidence.limitations)}"


def _build_exploit_path(assessment: ExploitRemediationAssessment) -> Optional[str]:
    exploit = assessment.exploit
    if exploit.exploit_path is None:
        # No path was established. Assumptions are only meaningful relative to a
        # path, so they are not surfaced on their own - the critic's
        # unsupported_claims is where an unsupported exploit claim would appear.
        return None
    if not exploit.required_assumptions:
        return exploit.exploit_path
    return f"{exploit.exploit_path}\n\n{_bullets(_ASSUMPTIONS_LABEL, exploit.required_assumptions)}"


def _build_remediation(assessment: ExploitRemediationAssessment) -> str:
    plan = assessment.remediation
    if not plan.validation_steps:
        return plan.remediation
    return f"{plan.remediation}\n\n{_bullets(_VALIDATION_LABEL, plan.validation_steps)}"


def build_verifier_input(
    evidence: EvidenceAssessment, assessment: ExploitRemediationAssessment
) -> LLMResponse:
    """Adapt agent output into the LLMResponse shape ai/verifier.py already consumes.

    Lets the deterministic category-consistency check run against the multi-agent
    narrative without changing one line of verifier.py. The text assembled here is
    the same text that reaches the report, so the verifier judges what a reader
    actually sees.
    """
    return LLMResponse(
        title=evidence.title,
        explanation=_build_explanation(evidence),
        exploit_path=_build_exploit_path(assessment),
        impact=assessment.exploit.impact,
        remediation=_build_remediation(assessment),
        patch_suggestion=assessment.remediation.patch_suggestion,
    )


def synthesize(
    finding: ScannerFinding,
    evidence: EvidenceAssessment,
    assessment: ExploitRemediationAssessment,
    critic: CriticAssessment,
    retrieved_chunks: list[RetrievedChunk],
    confidence_score: float,
    confidence_label: ConfidenceLabel,
    verification_status: VerificationStatus,
    repository_context: RepositoryContext,
) -> AIEnrichedFinding:
    """Assemble the final contract. Deterministic: same inputs, same output, no model call."""
    return AIEnrichedFinding(
        # Correlation key and scanner-owned fields - copied, never generated.
        finding_id=finding.finding_id,
        severity=finding.severity,
        scanner_sources=[finding.scanner],
        evidence=finding.raw_evidence,
        # Agent-authored narrative.
        title=evidence.title,
        explanation=_build_explanation(evidence),
        exploit_path=_build_exploit_path(assessment),
        impact=assessment.exploit.impact,
        remediation=_build_remediation(assessment),
        patch_suggestion=assessment.remediation.patch_suggestion,
        # Computed outside the model, passed through untouched.
        confidence_score=confidence_score,
        confidence_label=confidence_label,
        verification_status=verification_status,
        # Critic review signals - independent of verification_status.
        grounding_verdict=critic.verdict,
        supported_claims=list(critic.supported_claims),
        unsupported_claims=list(critic.unsupported_claims),
        # Knowledge-base sourced.
        references=_collect_references(retrieved_chunks),
        repository_context=repository_context.summary,
    )
