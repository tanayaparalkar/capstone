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

    return "\n\n".join(sections)
