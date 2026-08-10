"""Retrieval (RAG) layer: knowledge-base retrieval behind a swappable Retriever interface."""
from .base import EmbeddingFn, Retriever
from .context_retrieval import retrieve_context
from .in_memory_retriever import InMemoryRetriever
from .models import RetrievedChunk

__all__ = ["RetrievedChunk", "Retriever", "EmbeddingFn", "InMemoryRetriever", "retrieve_context"]
