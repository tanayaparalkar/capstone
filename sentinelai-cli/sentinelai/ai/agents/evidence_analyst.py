"""
Evidence Analyst - LLM call 1 of 3.

Reads only what it was given and reports what that evidence does and does
not show. It is deliberately the first stage and the only one that
produces `observed_evidence`: the two downstream agents receive its
validated output rather than re-deriving facts from the raw evidence
themselves, so a claim's provenance stays traceable to this one step.

The prompt's central constraint is negative - do not introduce anything
the evidence does not contain. A scanner finding supplies one rule match
and at most a few lines of code, which is far less than a model's prior
knowledge about the vulnerability class; without that constraint the
model reliably fills the gap with plausible, unverifiable detail, and
every later stage inherits it.

`title` is constrained rather than free-form: the prompt requires it to
restate the scanner's own category and rule, so the report headline never
asserts more than the scanner did. It exists here, on the first call,
because the alternative was a fourth LLM call purely to name the finding.
"""
from sentinelai.contracts import ScannerFinding

from ..prompt_builder import build_evidence_block
from ..repository_context import RepositoryContext
from ..retrieval import RetrievedChunk
from .explainer import LLMFn, run_agent
from .schemas import EvidenceAssessment

_ROLE = (
    "You are a security evidence analyst. You are reviewing ONE static-analysis finding.\n"
    "Your job is to report what the supplied evidence actually shows, and to be explicit about "
    "what it does not show. You are not being asked to speculate about exploitability or fixes - "
    "later stages do that."
)

_RULES = (
    "Rules:\n"
    "- Use ONLY the evidence supplied above. Do not introduce file contents, call sites, "
    "framework behaviour, or configuration that is not shown.\n"
    "- Put in `observed_evidence` only facts readable directly from the supplied evidence. If the "
    "evidence is a single line of code, say what that line does - not what the rest of the program "
    "probably does.\n"
    "- `interpretation` must be one to three sentences explaining WHY the observed evidence is a "
    "security problem: what weakness it creates and what could go wrong because of it. Write "
    "prose. Do NOT put the finding id, rule id, or category here - that belongs in `title`, and "
    "repeating it here leaves the report with no explanation at all.\n"
    "- Put in `limitations` what cannot be determined from this evidence alone, such as whether the "
    "input is attacker-controlled or whether the code is reachable.\n"
    "- `title` must be a short label restating the scanner's own category and rule for this "
    "finding. Do not claim impact or exploitability in the title.\n"
    "- Set `groundedness` to `supported` only if your interpretation follows from the observed "
    "evidence alone; `partially_supported` if it relies on typical patterns not shown here; "
    "`insufficient_evidence` if the evidence is too thin to interpret."
)


def build_evidence_prompt(
    finding: ScannerFinding,
    repository_context: RepositoryContext,
    retrieved_chunks: list[RetrievedChunk],
) -> str:
    return "\n\n".join(
        [_ROLE, build_evidence_block(finding, repository_context, retrieved_chunks), _RULES]
    )


def assess_evidence(
    finding: ScannerFinding,
    repository_context: RepositoryContext,
    retrieved_chunks: list[RetrievedChunk],
    generate: LLMFn,
) -> EvidenceAssessment:
    """One schema-constrained model call. Raises on a response that does not validate."""
    prompt = build_evidence_prompt(finding, repository_context, retrieved_chunks)
    return run_agent(prompt, generate, EvidenceAssessment)
