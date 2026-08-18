"""
In-memory Retriever implementation.

The only concrete Retriever for the frozen MVP: builds a fixed,
in-process index from a knowledge base once at construction, and
answers retrieve() calls against it. A Qdrant-backed implementation
(qdrant_retriever.py, deferred) would build/query its index completely
differently - upserting to a remote collection rather than holding
vectors in a Python list - which is why indexing lives inside this file
rather than a shared "indexer" module: there is no shared indexing
concept between the two, only a shared query interface
(Retriever.retrieve).

_IndexedEntry and retrieve() now carry each entry's references through
into the returned RetrievedChunk - added after the fact so
ai/pipeline.py can populate AIEnrichedFinding.references from retrieval
output alone, without reloading or searching the knowledge base itself.

Why this must be a class, not a function: retrieve() needs to reuse an
expensively-computed index (one embedding call per KB entry) across
every call, without re-embedding the whole corpus every time a finding
is enriched. A class is where "compute once in __init__, reuse in
retrieve()" naturally lives - unlike security_kb/loader.py, where
re-reading a small JSON file per call would be cheap and a class would
buy nothing.

Embeddings are injected, not imported: `embed` is a plain EmbeddingFn
(Callable[[list[str]], list[list[float]]], defined in .base - moved
there from this file so it's part of the retrieval contract rather
than an implementation detail this file happens to own; see .base's
docstring), supplied by the caller. This file has no import of, and no
knowledge of, any specific embedding provider (Gemini/OpenAI/Ollama/
local/etc.) - matching how ai/embeddings_base.py was deliberately cut
during the architecture review in favor of a plain function signature,
since no ABC was needed to make embeddings swappable. The same `embed`
function is reused for both indexing and querying (never swapped
between calls), which is itself the correctness requirement: comparing
vectors from two different embedding functions would be meaningless.

Index representation: a single list of (kb_entry_id, text, vector)
entries, built once in __init__, searched by a linear scan + cosine
similarity in retrieve(). No numpy, no ANN index, no persistence - the
frozen MVP's knowledge base is a handful of entries; O(n) space and
O(n log n) per query is the correct amount of engineering for that
size, not an approximation of what a production vector database would
do. The trigger to replace this with qdrant_retriever.py is exactly
this: the corpus growing enough for a linear scan to matter.

Cross-package import style matches providers/mock_provider.py: absolute
for the cross-top-level-package reference, via security_kb's public
surface (`from sentinelai.security_kb import KnowledgeBaseEntry`), not
security_kb.models directly - mock_provider.py itself imports
`from sentinelai.contracts import ...` (the package), not a submodule;
an earlier version of this file claimed this match while actually
reaching into security_kb.models directly, which was the same
package-boundary issue caught and fixed for ai/embeddings_ollama.py's
EmbeddingFn import. Relative imports for same-package siblings
(.base, .models) are unaffected.
"""
import math
from typing import NamedTuple

from sentinelai.security_kb import KnowledgeBaseEntry

from .base import EmbeddingFn, Retriever
from .models import RetrievedChunk


class _IndexedEntry(NamedTuple):
    kb_entry_id: str
    text: str
    vector: list[float]
    references: list[str]


def _entry_to_text(entry: KnowledgeBaseEntry) -> str:
    """Combine an entry's semantic content into one string - both what gets embedded and what a match returns as text."""
    parts = [entry.category, entry.vulnerable_pattern]
    if entry.exploit_condition:
        parts.append(entry.exploit_condition)
    parts.append(entry.remediation_guidance)
    return " ".join(parts)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class InMemoryRetriever(Retriever):
    """Retriever backed by an in-process list of embedded knowledge-base entries."""

    def __init__(self, entries: list[KnowledgeBaseEntry], embed: EmbeddingFn) -> None:
        self._embed = embed
        texts = [_entry_to_text(entry) for entry in entries]
        vectors = embed(texts) if texts else []
        self._index = [
            _IndexedEntry(kb_entry_id=entry.id, text=text, vector=vector, references=entry.references)
            for entry, text, vector in zip(entries, texts, vectors)
        ]

    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        if not self._index:
            return []
        query_vector = self._embed([query])[0]
        scored = [(indexed, _cosine_similarity(query_vector, indexed.vector)) for indexed in self._index]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return [
            RetrievedChunk(kb_entry_id=indexed.kb_entry_id, text=indexed.text, score=score, references=indexed.references)
            for indexed, score in scored[:top_k]
        ]
