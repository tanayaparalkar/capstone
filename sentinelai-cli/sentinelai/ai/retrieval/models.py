"""
Shared return type for the retrieval layer.

RetrievedChunk is what a Retriever implementation (ai/retrieval/base.py
and, later, in_memory_retriever.py / qdrant_retriever.py) returns per
match, and what ai/retrieval/context_retrieval.py, ai/prompt_builder.py,
and ai/pipeline.py consume. Kept in its own file, separate from
base.py, so the Retriever ABC's return type doesn't force base.py and
its implementations into importing each other.

references was added after the fact, correcting a design gap: pipeline.py
originally had no way to populate AIEnrichedFinding.references (which
must be KB-sourced, per ai/llm_response.py's docstring, to avoid the LLM
inventing URLs) without reloading or searching the knowledge base itself
- turning the orchestrator into a second, ad hoc repository. Retrieval
already has the source KnowledgeBaseEntry in hand while building this
object; carrying its references through here means every downstream
consumer gets everything it needs from retrieval's output alone.
Defaults to [] (default_factory, not required) matching
KnowledgeBaseEntry.references' own default - a chunk with no references
is exactly as valid as a KB entry with none.

No import of security_kb.KnowledgeBaseEntry: kb_entry_id stores only
the entry's id (a plain str), not the entry itself, so importing that
model here would add a dependency for no additional type safety.

score is intentionally unconstrained (no ge=0.0/le=1.0, unlike
AIEnrichedFinding.confidence_score in contracts/ai_finding.py): that
field is a project-defined 0-1 contract; this one is whatever
similarity metric the still-pending embedding/retriever choice
produces (e.g. cosine similarity, which can be negative) - bounding it
now would assume a specific metric before that decision is made.
"""
from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    kb_entry_id: str = Field(
        ..., description="id of the security_kb.KnowledgeBaseEntry this chunk was retrieved from."
    )
    text: str = Field(..., description="The retrieved text content, to be included in the reasoning prompt.")
    score: float = Field(
        ..., description="Similarity score from the retriever; scale depends on the retriever implementation."
    )
    references: list[str] = Field(
        default_factory=list,
        description="References carried through from the source KnowledgeBaseEntry.references.",
    )
