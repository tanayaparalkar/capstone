"""
Critic - LLM call 3 of 3.

Re-reads the other agents' claims against the same evidence they were
given and reports which of those claims that evidence actually supports.
It is a review pass, not a proof system: it can catch a claim that
overreaches the evidence in front of it, and it cannot establish that a
supported claim is true. The paper wording for this stage is "grounding
review", never "verification" and never "calibrated".

Its verdict is reported through AIEnrichedFinding.grounding_verdict and
is deliberately kept separate from verification_status, which
ai/verifier.py produces deterministically and offline by a completely
different method. The two are allowed to disagree, and
ai/agents/synthesizer.py is forbidden from deriving either from the
other - a disagreement is information, and collapsing them would destroy
it.

The critic sees the finding, the evidence block, and both prior typed
outputs. It does not get to rewrite them: its only outputs are the two
claim lists and a verdict, so a critical review can never silently edit
the narrative it is reviewing. Correcting the analysis on the strength of
a critique would need a fourth call and a re-review, which this pipeline
does not do.
"""
from sentinelai.contracts import ScannerFinding

from ..prompt_builder import build_evidence_block
from ..repository_context import RepositoryContext
from ..retrieval import RetrievedChunk
from .explainer import LLMFn, run_agent
from .schemas import CriticAssessment, EvidenceAssessment, ExploitRemediationAssessment

_ROLE = (
    "You are a critical reviewer checking another analyst's work on ONE static-analysis finding.\n"
    "You are given the original evidence and the claims that were made from it. Decide which of "
    "those claims the evidence actually supports.\n"
    "You are reviewing grounding, not truth: your question is 'does the supplied evidence support "
    "this claim', not 'is this claim true in general'."
)

_RULES = (
    "Rules:\n"
    "- Put a claim in `supported_claims` only if the evidence above substantiates it.\n"
    "- Put a claim in `unsupported_claims` if it goes beyond the evidence, depends on unstated "
    "assumptions, or asserts something the evidence cannot show. A claim being plausible, or "
    "generally true of this vulnerability class, is NOT the same as it being supported here.\n"
    "- Assumptions that were already declared as assumptions are not unsupported claims; judge "
    "them on whether the analysis correctly treated them as assumptions.\n"
    "- Set `verdict` to `supported` if the analysis stays within the evidence; "
    "`partially_supported` if parts overreach; `insufficient_evidence` if the evidence is too thin "
    "to support the analysis at all.\n"
    "- If you set `verdict` to `partially_supported`, you MUST put at least one item in "
    "`unsupported_claims` saying which claim overreaches and why. A partial verdict with an empty "
    "`unsupported_claims` list is rejected: it says the analysis overreaches without saying how.\n"
    "- If you set `verdict` to `insufficient_evidence`, you MUST put at least one item in "
    "`unsupported_claims` explaining what the evidence cannot establish.\n"
    "- If every claim is supported, set `verdict` to `supported` and leave `unsupported_claims` "
    "empty. Do not use `partially_supported` as a hedge when you have nothing to list.\n"
    "- `supported_claims` may be left empty if nothing is separately worth listing; only "
    "`unsupported_claims` is required by the rules above.\n"
    "- An exploit path is a narrative: it must describe attacker action and how data or control "
    "reaches the vulnerable code. If the exploit path under review is merely a file path, a line "
    "number, a finding id, a category, or a rule id, it describes nothing an attacker does - treat "
    "it as unsupported and say so, because a location is not evidence of exploitability.\n"
    "- Each entry in `supported_claims` and `unsupported_claims` must restate the claim being "
    "judged, in prose. Do not write a file path, line number, or bare identifier - an entry that "
    "does not say what was claimed is unreadable in a report.\n"
    "- Do not rewrite the analysis. Report only what is and is not supported."
)


def _format_claims_under_review(
    evidence: EvidenceAssessment, exploit_remediation: ExploitRemediationAssessment
) -> str:
    lines = ["Claims under review:", f"- Interpretation: {evidence.interpretation}"]
    exploit = exploit_remediation.exploit
    if exploit.exploit_path:
        lines.append(f"- Exploit path: {exploit.exploit_path}")
    if exploit.impact:
        lines.append(f"- Impact: {exploit.impact}")
    if exploit.required_assumptions:
        lines.append("- Declared assumptions:")
        lines.extend(f"  - {item}" for item in exploit.required_assumptions)
    lines.append(f"- Remediation: {exploit_remediation.remediation.remediation}")
    if exploit_remediation.remediation.patch_suggestion:
        lines.append(f"- Patch suggestion: {exploit_remediation.remediation.patch_suggestion}")
    return "\n".join(lines)


def build_critic_prompt(
    finding: ScannerFinding,
    evidence: EvidenceAssessment,
    exploit_remediation: ExploitRemediationAssessment,
    repository_context: RepositoryContext,
    retrieved_chunks: list[RetrievedChunk],
) -> str:
    return "\n\n".join(
        [
            _ROLE,
            build_evidence_block(finding, repository_context, retrieved_chunks),
            _format_claims_under_review(evidence, exploit_remediation),
            _RULES,
        ]
    )


def critique(
    finding: ScannerFinding,
    evidence: EvidenceAssessment,
    exploit_remediation: ExploitRemediationAssessment,
    repository_context: RepositoryContext,
    retrieved_chunks: list[RetrievedChunk],
    generate: LLMFn,
) -> CriticAssessment:
    """One schema-constrained model call reviewing the two prior agents' claims."""
    prompt = build_critic_prompt(
        finding, evidence, exploit_remediation, repository_context, retrieved_chunks
    )
    return run_agent(prompt, generate, CriticAssessment)
