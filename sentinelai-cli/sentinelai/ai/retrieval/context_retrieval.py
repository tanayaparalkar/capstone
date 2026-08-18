"""
Builds a retrieval query from a finding and repository context, and
calls a Retriever with it.

The translation boundary between the domain layer (ScannerFinding,
RepositoryContext) and the generic retrieval layer (Retriever): this
file is allowed to know about ScannerFinding specifically so Retriever
implementations never have to. Query construction lives here rather
than in prompt_builder.py because it is a different job - a short
string optimized for retrieval matching, not the full text sent to the
LLM - and because retrieval happens before prompting, so the reverse
dependency (this file importing prompt_builder.py) would run backwards.

repository_context is a required RepositoryContext, not
Optional[RepositoryContext]: ai/repository_context.py already assigns
"what does no context look like" to the caller (RepositoryContext()),
so this function doesn't need a None-handling branch of its own.
top_k takes no default for the same reason Retriever.retrieve itself
doesn't: ai/config.py's AISettings.retrieval_top_k is the one place
that default lives - reading config here would be exactly the hidden
configuration lookup the frozen constraints rule out.
"""
from sentinelai.contracts import ScannerFinding

from ..repository_context import RepositoryContext
from .base import Retriever
from .models import RetrievedChunk


def _build_query(finding: ScannerFinding, repository_context: RepositoryContext) -> str:
    """Combine the finding's descriptive fields (not its identifiers/location/severity) and any repo context summary."""
    parts = [finding.category.strip(), finding.message.strip()]
    if finding.raw_evidence:
        parts.append(finding.raw_evidence.strip())
    if repository_context.summary:
        parts.append(repository_context.summary.strip())
    return " ".join(parts)


def retrieve_context(
    finding: ScannerFinding,
    repository_context: RepositoryContext,
    retriever: Retriever,
    top_k: int,
) -> list[RetrievedChunk]:
    query = _build_query(finding, repository_context)
    return retriever.retrieve(query, top_k)
