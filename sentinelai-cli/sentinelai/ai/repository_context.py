"""
Repository-context shape for the AI Intelligence Layer.

contracts/scan_result.py already defines RepositoryInfo (name, path,
commit_hash, branch, languages) - that's scan-level metadata about
*which* repository was scanned, and it already exists; nothing here
duplicates it. This module defines the separate, AI-facing shape: a
short free-text summary of the repository relevant to reasoning about a
finding, consumed by ai/retrieval/context_retrieval.py,
ai/prompt_builder.py, and ai/pipeline.py.

Kept to a single optional free-text field by design: any richer
structure (detected languages, file tree, dependency list, ...) would be
guessing at prompt-content decisions that aren't this module's to make.
A summary string is the smallest shape that lets prompt_builder.py
include repository context in a prompt at all, without this module
inventing fields on behalf of a dedicated prompt-design task. No
separate "empty" constant either - RepositoryContext() is already
trivially constructible, matching how statistics/models.py
default-constructs ConfidenceStatistics() directly rather than naming a
shared empty instance.

from_backend_context() adapts a real sentinelai.backend.context_builder.
RepositoryContext into this module's shape. The conversion is
deliberately minimal - repository name and detected languages only, the
two facts backend provides with zero judgment call (name is direct;
languages is already a deterministically sorted list from
detect_languages()). Git metadata, dependencies, and code snippets are
NOT folded in: deciding how to summarize them (which dependencies
matter, what format, how much git history is "relevant to reasoning
about a finding") is a real prompt-content decision with many reasonable
answers, not a mechanical translation - exactly the kind of guessing
this module's single-field design avoids by staying minimal. That
decision belongs to a dedicated prompt-design task, not this adapter.

Importing backend.context_builder here is a one-way dependency
(ai -> backend), not a circular one: backend has no imports from ai,
scanners, or providers anywhere, and the frozen architecture places
Backend upstream of AI Retrieval in its data flow, so this is the
expected direction.

from_backend_context() is called from main.py's scan() command, once
per AI-configured scan, to build the RepositoryContext passed into
enrich_findings().
"""
from typing import Optional

from pydantic import BaseModel, Field

from sentinelai.backend.context_builder import RepositoryContext as BackendRepositoryContext


class RepositoryContext(BaseModel):
    """Repository-context shape for the AI layer: a single free-text summary, by design - see module docstring."""

    summary: Optional[str] = Field(
        default=None,
        description="Free-text description of the repository relevant to reasoning about a finding, if available.",
    )


def from_backend_context(context: BackendRepositoryContext) -> RepositoryContext:
    """Adapt a real backend.context_builder.RepositoryContext into this module's AI-layer shape."""
    summary = f"Repository '{context.repository.name}'"
    languages = ", ".join(language.name for language in context.languages)
    if languages:
        summary += f" ({languages})"
    return RepositoryContext(summary=summary + ".")
