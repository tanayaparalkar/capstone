"""
Raw structured output the LLM is asked to produce for one finding,
before pipeline.py assembles it into a full AIEnrichedFinding.

Not parsed directly into AIEnrichedFinding because that contract mixes
LLM-authored content with fields the LLM must never determine:
confidence_score/confidence_label are computed by ai/confidence_scorer.py
from evidence-based signals, not self-reported (see below);
verification_status is set by ai/verifier.py; finding_id and
scanner_sources are already known by pipeline.py from the ScannerFinding
being processed, so trusting an LLM-produced value for a correlation
key would be both unnecessary and fragile; references are sourced from
the knowledge base downstream rather than the LLM's own text, to avoid
inventing URLs; related_findings requires visibility across multiple
findings in one scan, which a single-finding explainer call does not
have - that's ai/agents/correlator.py's job, deferred past MVP.

Fields included here are exactly the ones that require the LLM's
reasoning and nothing else: title, explanation, exploit_path, impact,
remediation, patch_suggestion - a 1:1 subset of AIEnrichedFinding's own
fields, mirroring their required/optional shape (exploit_path, impact,
and patch_suggestion stay Optional for the same "not every finding has
a clear exploit narrative" reason AIEnrichedFinding.exploit_path is
nullable).

severity is deliberately excluded even though contracts/ai_finding.py's
docstring allows AIEnrichedFinding.severity to "restate or refine" the
scanner's severity: nothing in the frozen architecture asks for
LLM-driven severity refinement yet, so pipeline.py carries
ScannerFinding.severity through unchanged for MVP. Revisit only if that
capability is explicitly requested later.

model_config forbids extra fields (extra="forbid"), unlike
ScannerFinding/AIEnrichedFinding, which both rely on Pydantic's default
"ignore" for extra keys. That default is right for those two models
because their input is a scanner or a human-controlled fixture - a
known, trustworthy source. This model's input is raw, potentially
hallucinating LLM output; an unexpected key here is far more likely to
mean the LLM misunderstood the task than that the schema is stale, so
this model rejects it immediately rather than silently dropping it.
This addresses hallucinated *structure* only - hallucinated *content* in
an otherwise well-formed response (a plausible but false explanation)
is exactly what ai/verifier.py exists to catch; that is not this
model's job.

Deliberately provider-agnostic: no import of any LLM SDK or
provider-specific response type. This model is the contract between
ai/llm_<provider>.py (whichever provider produces text) and
ai/agents/explainer.py (which parses that text into this shape) - if
this file depended on a specific provider's response format, swapping
providers would require changing this file too, defeating the purpose
of ai/llm_base.py's interface.
"""
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class LLMResponse(BaseModel):
    """Parsed LLM output - immutable: a downstream stage that needs different content constructs a new object."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    title: str = Field(..., description="Short human-readable title for the finding.")
    explanation: str = Field(..., description="Explanation of the vulnerability.")
    exploit_path: Optional[str] = Field(
        None, description="Narrative of how this finding could be exploited, if applicable."
    )
    impact: Optional[str] = Field(None, description="Description of the potential impact if exploited.")
    remediation: str = Field(..., description="Suggested fix.")
    patch_suggestion: Optional[str] = Field(None, description="Concrete patch/diff suggestion, if generated.")
