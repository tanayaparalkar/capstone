# Appendix: SentinelAI Multi-Agent Prompts & Response Schemas

This appendix contains the verbatim prompts and Pydantic response schemas for each of the three stages in the SentinelAI multi-agent enrichment pipeline (`sentinelai/ai/agents/`).

---

## Stage 1: Evidence Analyst

- **Source Code**: [`sentinelai/ai/agents/evidence_analyst.py`](file:///c:/Users/nitaa/Downloads/capstone/sentinelai-cli/sentinelai/ai/agents/evidence_analyst.py)
- **Schema Code**: [`sentinelai/ai/agents/schemas.py:101-140`](file:///c:/Users/nitaa/Downloads/capstone/sentinelai-cli/sentinelai/ai/agents/schemas.py#L101-L140)

### Prompt Text (Verbatim)

```python
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
```

### Prompt Assembly

```python
def build_evidence_prompt(
    finding: ScannerFinding,
    repository_context: RepositoryContext,
    retrieved_chunks: list[RetrievedChunk],
) -> str:
    return "\n\n".join(
        [_ROLE, build_evidence_block(finding, repository_context, retrieved_chunks), _RULES]
    )
```

### Response Schema: `EvidenceAssessment`

```python
class EvidenceAssessment(BaseModel):
    """Output of the Evidence Analyst - LLM call 1 of 3."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    title: NonBlankStr = Field(
        ...,
        description=(
            "Short label for this finding, grounded in the scanner's own category, rule id, and "
            "evidence. Not a free-form headline: the prompt constrains it to restate what the "
            "scanner reported, so the report's title never asserts more than the scanner did."
        ),
    )
    observed_evidence: list[NonBlankStr] = Field(
        default_factory=list,
        description="Facts read directly from the supplied evidence. Nothing inferred, assumed, or recalled.",
    )
    interpretation: NonBlankStr = Field(
        ..., description="Why the observed evidence is security-relevant."
    )
    limitations: list[NonBlankStr] = Field(
        default_factory=list,
        description="What cannot be determined from the supplied evidence alone.",
    )
    groundedness: GroundingVerdict = Field(
        ..., description="How well the interpretation is supported by the observed evidence."
    )
```

---

## Stage 2: Exploit and Remediation Analyst

- **Source Code**: [`sentinelai/ai/agents/exploit_remediation_analyst.py`](file:///c:/Users/nitaa/Downloads/capstone/sentinelai-cli/sentinelai/ai/agents/exploit_remediation_analyst.py)
- **Schema Code**: [`sentinelai/ai/agents/schemas.py:142-217`](file:///c:/Users/nitaa/Downloads/capstone/sentinelai-cli/sentinelai/ai/agents/schemas.py#L142-L217)

### Prompt Text (Verbatim)

```python
_ROLE = (
    "You are two security analysts working on ONE static-analysis finding.\n"
    "As the exploit analyst, you assess whether and how this finding could be exploited.\n"
    "As the remediation analyst, you propose how to fix it and how to confirm the fix.\n"
    "A prior evidence analyst has already established what the evidence shows; work from that."
)

_RULES = (
    "Rules:\n"
    "- Base both sections on the evidence and the prior assessment above. Do not invent code, "
    "call sites, or configuration that was not shown.\n"
    "- `exploit.exploit_path`, when non-null, must be prose describing the steps by which an "
    "attacker would reach and abuse this code: what they control, what they send, and what happens "
    "as a result. Do NOT put a file path, line number, or location here - the location is already "
    "recorded on the finding, and repeating it leaves the report with no exploit narrative at all. "
    "Set it to null unless the evidence supports a concrete path; a guess stated as fact is worse "
    "than null.\n"
    "- `exploit.impact`, when non-null, must state in prose what an attacker gains - the "
    "consequence, not a severity word and not a location. Null if no exploit path holds.\n"
    "- `exploit.required_assumptions` must list every condition your exploit path depends on that "
    "the evidence does NOT establish - for example that the value is attacker-controlled, or that "
    "the code is reachable. Do not restate observed facts here; assumptions and observations are "
    "different things.\n"
    "- `remediation.remediation` is required: give the concrete fix for this specific code, in "
    "prose. Not a location, not a restatement of the rule name.\n"
    "- `remediation.patch_suggestion` may be null. When provided, supply a clean, machine-applicable unified diff (diff --git or unified hunk) or replacement code block grounded strictly in the evidence shown; do not include conversational prose, markdown commentary, or ungrounded surrounding code inside the patch block.\n"
    "- `remediation.validation_steps` should say how an engineer would confirm the fix worked."
)
```

### Prior Evidence Formatter

```python
def _format_prior_assessment(evidence: EvidenceAssessment) -> str:
    lines = ["Prior evidence assessment:", f"- Interpretation: {evidence.interpretation}"]
    if evidence.observed_evidence:
        lines.append("- Observed evidence:")
        lines.extend(f"  - {item}" for item in evidence.observed_evidence)
    if evidence.limitations:
        lines.append("- Limitations of that evidence:")
        lines.extend(f"  - {item}" for item in evidence.limitations)
    lines.append(f"- Groundedness: {evidence.groundedness.value}")
    return "\n".join(lines)
```

### Prompt Assembly

```python
def build_exploit_remediation_prompt(
    finding: ScannerFinding,
    evidence: EvidenceAssessment,
    repository_context: RepositoryContext,
    retrieved_chunks: list[RetrievedChunk],
) -> str:
    return "\n\n".join(
        [
            _ROLE,
            build_evidence_block(finding, repository_context, retrieved_chunks),
            _format_prior_assessment(evidence),
            _RULES,
        ]
    )
```

### Response Schema: `ExploitRemediationAssessment`

```python
class ExploitAssessment(BaseModel):
    """Exploit section of LLM call 2 of 3."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    exploit_path: Optional[NonBlankStr] = Field(
        None,
        description=(
            "How the finding could be exploited, or null when the supplied evidence does not "
            "support a concrete path. Null is the correct answer for a finding whose "
            "reachability cannot be established from one file's evidence."
        ),
    )
    impact: Optional[NonBlankStr] = Field(
        None, description="What an attacker would gain if the exploit path holds, or null."
    )
    required_assumptions: list[NonBlankStr] = Field(
        default_factory=list,
        description=(
            "Conditions that must hold for the exploit path to be real but that the evidence does "
            "not establish - for example reachability from untrusted input. Kept separate from "
            "EvidenceAssessment.observed_evidence so an assumption is never mistaken for a fact."
        ),
    )

class PatchType(str, Enum):
    UNIFIED_DIFF = "unified_diff"
    REPLACEMENT_BLOCK = "replacement_block"
    LINE_REPLACEMENT = "line_replacement"

class StructuredPatch(BaseModel):
    """Machine-applicable code patch specification."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    patch_type: PatchType = Field(
        ..., description="Format of the patch: unified_diff, replacement_block, line_replacement"
    )
    target_file: Optional[str] = Field(None, description="Path to the file to patch")
    line_start: Optional[int] = Field(None, ge=1, description="First affected line (1-indexed)")
    line_end: Optional[int] = Field(None, ge=1, description="Last affected line (1-indexed)")
    patch_content: NonBlankStr = Field(..., description="The actual unified diff or replacement code")

class RemediationPlan(BaseModel):
    """Remediation section of LLM call 2 of 3."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    remediation: NonBlankStr = Field(..., description="How to fix the finding.")
    patch_suggestion: Optional[NonBlankStr] = Field(
        None,
        description=(
            "Concrete code change, or null when one cannot responsibly be proposed from the "
            "supplied evidence. Can be a unified diff or replacement snippet."
        ),
    )
    structured_patch: Optional[StructuredPatch] = Field(
        None,
        description="Optional structured patch specification for automated machine application.",
    )
    validation_steps: list[NonBlankStr] = Field(
        default_factory=list, description="How to confirm the fix worked."
    )

class ExploitRemediationAssessment(BaseModel):
    """Envelope for LLM call 2 of 3, carrying both analysts' sections in one request."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    exploit: ExploitAssessment
    remediation: RemediationPlan
```

---

## Stage 3: Critic

- **Source Code**: [`sentinelai/ai/agents/critic.py`](file:///c:/Users/nitaa/Downloads/capstone/sentinelai-cli/sentinelai/ai/agents/critic.py)
- **Schema Code**: [`sentinelai/ai/agents/schemas.py:219-306`](file:///c:/Users/nitaa/Downloads/capstone/sentinelai-cli/sentinelai/ai/agents/schemas.py#L219-L306)

### Prompt Text (Verbatim)

```python
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
```

### Claims Under Review Formatter

```python
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
```

### Prompt Assembly

```python
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
```

### Response Schema: `CriticAssessment`

```python
class CriticAssessment(BaseModel):
    """Output of the Critic - LLM call 3 of 3."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    verdict: GroundingVerdict = Field(
        ..., description="Overall judgement of how well the narrative is grounded in the evidence."
    )
    unsupported_claims: list[ProseClaim] = Field(
        ...,
        description=(
            "Claims asserted without support in the supplied evidence, or supported only under "
            "assumptions. Distinct from EvidenceAssessment.observed_evidence by construction. "
            "Required to be non-empty for a partially_supported or insufficient_evidence verdict - "
            "see _verdict_matches_claims."
        ),
    )
    supported_claims: list[NonBlankStr] = Field(
        default_factory=list, description="Claims the supplied evidence supports."
    )
```
