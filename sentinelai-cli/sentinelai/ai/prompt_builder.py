"""
Assembles the reasoning prompt from a finding, repository context, and
already-retrieved knowledge-base chunks.

This is the boundary between the retrieval layer's output and the LLM
call: it consumes RetrievedChunk, imported from ai.retrieval's public
surface (`from .retrieval import RetrievedChunk`), not from
ai/retrieval/models.py directly - matching the same package-boundary
fix applied to ai/embeddings_ollama.py's EmbeddingFn import. Note this
does mean Python transitively loads Retriever/InMemoryRetriever/
context_retrieval.py as an import-mechanics side effect of
ai/retrieval/__init__.py eagerly importing its own __all__ - the same
accepted tradeoff already present wherever anything imports from
ai.retrieval. This file's own code never references any of them: no
Retriever, no context_retrieval.py, no retriever implementation is
used in this file's logic. Changing embedding providers or swapping
in-memory/Qdrant retrieval must never require touching this file's
code, and rewording this file must never require touching retrieval's
code - the transitive import graph is not the same thing as a code-level
dependency.

Includes ai/prompt_templates.py's content as a module constant here
rather than a separate file, per the frozen architecture's merge
decision - a plain string, not a Jinja2 template, since there is no
conditional templating problem here Jinja2 would actually be solving.

chunk similarity scores are deliberately excluded from the prompt: a
score is a fact about how well text matched the retrieval query, not
about the vulnerability, and showing it risks the LLM conflating
retrieval relevance with AIEnrichedFinding.confidence_score - a
different, separately-computed value produced later by
confidence_scorer.py, not by the LLM.

repository_context is a required RepositoryContext, not Optional, for
the same reason ai/retrieval/context_retrieval.py's parameter is: the
"no context" default lives with pipeline.py, not duplicated here.

_OUTPUT_FORMAT_INSTRUCTIONS was missing entirely until this file was
observed producing the actual bug it was meant to prevent:
_TASK_INSTRUCTIONS above asked the model to "explain... describe...
note... suggest," with no mention of JSON anywhere in the prompt, while
ai/agents/explainer.py has always called json.loads() on the result and
ai/llm_response.py's LLMResponse has always rejected any extra key.
Nothing in the prompt ever told the model about either constraint.
Verified directly against a real llama3.1:8b response before writing
this: with no format instruction, the model produced a well-formed,
useful, entirely prose explanation with Markdown bold headers - not
malformed output, just the ordinarily reasonable answer to a prompt
that never asked for JSON in the first place. Markdown code fences
(handled defensively by ai/llm_ollama.py's _strip_markdown_fences) are
a *secondary* symptom of the same root cause - a model given no output-
format instruction may reasonably choose to fence code-shaped content
- not a problem this section alone would have covered without the
fence-stripping already in place.

Placed as the *last* section build_prompt() appends, after the finding,
repository context, and retrieved knowledge - not folded into
_TASK_INSTRUCTIONS at the top. Format instructions are more reliably
followed the closer they are to where generation actually begins;
putting the six field names and the "JSON only" requirement immediately
before the model starts producing output, rather than several hundred
tokens earlier, is a real difference for a small local model, not a
stylistic preference.

Field list, names, and required/optional split are read directly from
LLMResponse's own field declarations (ai/llm_response.py) - title,
explanation, and remediation are required; exploit_path, impact, and
patch_suggestion are Optional[str] = None. Instructing the model to
include all six keys with explicit `null` for anything not applicable,
rather than allowing keys to be silently omitted, is the more reliable
instruction for a small model even though Pydantic itself would accept
either: an omitted-vs-present-with-null distinction is exactly the kind
of small inconsistency a smaller model is more likely to get wrong than
a larger one.
"""
from typing import Optional

from sentinelai.contracts import ScannerFinding

from .repository_context import RepositoryContext
from .retrieval import RetrievedChunk

_TASK_INSTRUCTIONS = (
    "You are a security analyst reviewing a single static-analysis finding. Using the "
    "finding's evidence below and the related security knowledge (if any), explain the "
    "vulnerability, describe how it could be exploited if applicable, note its potential "
    "impact, and suggest a remediation."
)

_OUTPUT_FORMAT_INSTRUCTIONS = (
    "Respond with ONLY a single JSON object and no other text - no explanation before or "
    "after it, and no Markdown code fences (no ```). The JSON object must have exactly these "
    'six keys: "title", "explanation", "exploit_path", "impact", "remediation", '
    '"patch_suggestion".\n'
    '"title", "explanation", and "remediation" are required non-empty strings. '
    '"exploit_path", "impact", and "patch_suggestion" are optional: include the key and set '
    "it to JSON null if it does not apply to this finding, rather than omitting the key or "
    'writing a placeholder like "N/A".'
)


def _format_finding_section(finding: ScannerFinding) -> str:
    lines = [
        f"Finding: {finding.finding_id}",
        f"Category: {finding.category}",
        f"Severity: {finding.severity.value}",
        f"Rule: {finding.rule_id}",
    ]
    if finding.file:
        location = finding.file
        if finding.line_start:
            location += f":{finding.line_start}"
            if finding.line_end and finding.line_end != finding.line_start:
                location += f"-{finding.line_end}"
        lines.append(f"Location: {location}")
    if finding.cwe:
        lines.append(f"CWE: {finding.cwe}")
    lines.append(f"Description: {finding.message}")
    if finding.raw_evidence:
        lines.append(f"Evidence:\n{finding.raw_evidence}")
    return "\n".join(lines)


def _format_repository_context_section(repository_context: RepositoryContext) -> Optional[str]:
    if not repository_context.summary:
        return None
    return f"Repository context: {repository_context.summary}"


def _format_retrieved_chunks_section(retrieved_chunks: list[RetrievedChunk]) -> Optional[str]:
    if not retrieved_chunks:
        return None
    lines = [
        "Related security knowledge:",
        "Use the following knowledge only as supporting context. Do not copy it verbatim. "
        "Prefer the finding's evidence if there is any conflict.",
        "",
    ]
    for chunk in retrieved_chunks:
        lines.append(f"- [{chunk.kb_entry_id}] {chunk.text}")
    return "\n".join(lines)


def build_evidence_block(
    finding: ScannerFinding,
    repository_context: RepositoryContext,
    retrieved_chunks: list[RetrievedChunk],
) -> str:
    """The shared evidence every agent reasons over: the finding, repository context, and KB chunks.

    Extracted so ai/agents/'s three prompts present *identical* evidence and
    differ only in their role instruction and output schema. Duplicating this
    formatting per agent would let the agents silently drift apart on what they
    were shown, which would undermine the one property the critic stage depends
    on - that every agent judged the same material.

    Returns only the evidence sections; the role instruction and the output
    contract are each agent's own concern.
    """
    sections = [_format_finding_section(finding)]

    repository_context_section = _format_repository_context_section(repository_context)
    if repository_context_section is not None:
        sections.append(repository_context_section)

    retrieved_chunks_section = _format_retrieved_chunks_section(retrieved_chunks)
    if retrieved_chunks_section is not None:
        sections.append(retrieved_chunks_section)

    return "\n\n".join(sections)


def build_prompt(
    finding: ScannerFinding,
    repository_context: RepositoryContext,
    retrieved_chunks: list[RetrievedChunk],
) -> str:
    sections = [_TASK_INSTRUCTIONS, _format_finding_section(finding)]

    repository_context_section = _format_repository_context_section(repository_context)
    if repository_context_section is not None:
        sections.append(repository_context_section)

    retrieved_chunks_section = _format_retrieved_chunks_section(retrieved_chunks)
    if retrieved_chunks_section is not None:
        sections.append(retrieved_chunks_section)

    sections.append(_OUTPUT_FORMAT_INSTRUCTIONS)

    return "\n\n".join(sections)
