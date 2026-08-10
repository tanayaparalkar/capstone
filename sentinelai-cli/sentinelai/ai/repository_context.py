"""
Interim placeholder for the repository context Nitaanth's repository-
analysis layer will eventually produce.

contracts/scan_result.py already defines RepositoryInfo (name, path,
commit_hash, branch, languages) - that's scan-level metadata about
*which* repository was scanned, and it already exists; nothing here
duplicates it. What's still missing from contracts/ is the actual
analysis content Nitaanth's layer would hand to a reasoning step -
docs/CONTRACTS.md refers to it only as "the full repository-context
object Nithanth builds," without defining its shape yet.

This module is a deliberately minimal stand-in for that missing piece,
the same way providers/base.py's FindingsProvider anticipated a second
real implementation: ai/retrieval/context_retrieval.py, ai/prompt_builder.py,
and ai/pipeline.py can be written and tested against this shape now, and
the swap once Nitaanth's real contract lands is contained to this one
file plus whatever call sites read RepositoryContext.summary.

Kept to a single optional free-text field on purpose: any richer
structure (detected languages, file tree, dependency list, ...) would be
guessing at a design that isn't this module's to make. A summary string
is the smallest shape that lets prompt_builder.py include repository
context in a prompt at all, without inventing fields on Nitaanth's
behalf. No separate "empty" constant either - RepositoryContext() is
already trivially constructible, matching how statistics/models.py
default-constructs ConfidenceStatistics() directly rather than naming a
shared empty instance.
"""
from typing import Optional

from pydantic import BaseModel, Field


class RepositoryContext(BaseModel):
    """Interim repository-context shape - see module docstring. Replace when Nitaanth's real contract lands."""

    summary: Optional[str] = Field(
        default=None,
        description="Free-text description of the repository relevant to reasoning about a finding, if available.",
    )
