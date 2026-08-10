"""
Retrieval interface.

Retriever is the abstraction ai/retrieval/context_retrieval.py depends
on, mirroring providers/base.py's FindingsProvider: today
in_memory_retriever.py is the only implementation; a Qdrant-backed
implementation (qdrant_retriever.py) is a drop-in replacement behind
this same interface later, per the project owner's explicit
extensibility requirement ("Qdrant can replace in-memory retrieval").
ABC over typing.Protocol for the same reason FindingsProvider is an ABC
and not a Protocol: matching the one idiom this codebase already uses
for a provider-style swap, rather than introducing a second one.

retrieve() takes a plain str query, not a ScannerFinding or any other
domain object: building the query string from a ScannerFinding (and
optional repository context) is context_retrieval.py's job, per the
frozen architecture. A generic text-in, chunks-out interface is
retrieval-strategy-agnostic and keeps this file (and every
implementation of it) free of a dependency on contracts.ScannerFinding
that a text-only interface has no need for.

top_k has no default here (unlike FindingsProvider.get_scan_result's
mode: ScanMode = ScanMode.STANDARD): AISettings.retrieval_top_k in
ai/config.py is already the single source of truth for that default -
a second default on this signature would risk the two drifting out of
sync. Callers pass it explicitly, sourced from get_settings().

Returns list[RetrievedChunk], not list[KnowledgeBaseEntry]: a chunk is
a per-query result (it carries a similarity score, which is meaningless
without a specific query) while a KnowledgeBaseEntry is static,
query-independent corpus data. Returning entries directly would
conflate the corpus with a search result and leave no field for the
score.

No index/add/delete/persistence/stats/config methods, and no batching
or async variant: nothing in the frozen architecture names any of them.
Indexing in particular is deliberately excluded - it means something
different per implementation (in_memory_retriever.py builds an
in-process array; a Qdrant implementation would upsert to a remote
collection), so forcing a shared index() method into this interface
would leak one implementation's concept into an abstraction the other
doesn't share.

EmbeddingFn lives here, not in in_memory_retriever.py where it was
originally defined: it's part of the retrieval layer's contract (what
"an embedding function" looks like to anything that needs one), not an
implementation detail InMemoryRetriever happens to own. Moved after
ai/embeddings_ollama.py's own review found importing it from an
implementation file created an unnecessary coupling - a provider
depending on another implementation's module instead of on a contract.
Re-exported through ai/retrieval/__init__.py so callers can import it
from the package (`from .retrieval import EmbeddingFn`), the same way
Retriever and RetrievedChunk already are.
"""
from abc import ABC, abstractmethod
from typing import Callable

from .models import RetrievedChunk

EmbeddingFn = Callable[[list[str]], list[list[float]]]


class Retriever(ABC):
    """Source of RetrievedChunk matches for a text query, swappable behind this interface."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """Return up to top_k RetrievedChunk matches for query, ordered most relevant first."""
        raise NotImplementedError
