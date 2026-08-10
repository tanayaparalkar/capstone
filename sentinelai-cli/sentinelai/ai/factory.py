"""
Construction and wiring for the AI Intelligence Layer's concrete
dependencies - the one file whose entire job is to know about
security_kb, embeddings_ollama, llm_ollama, and
ai/retrieval/in_memory_retriever.py all at once, so nothing else has
to.

Not in ai/pipeline.py: pipeline.py was deliberately built with "no
embedding-provider... knowledge" and no hidden configuration lookups
beyond its own orchestration-level settings (retrieval_top_k,
enable_verification) - its own docstring already names this exact gap
("Constructing a retriever... is a one-time setup concern that happens
before enrich_findings() is ever called, done by whoever assembles the
pipeline's dependencies"). This file is that "whoever." Adding
construction logic to pipeline.py would reintroduce the provider
knowledge that file was explicitly designed not to have.

Not in ai/llm_ollama.py or ai/embeddings_ollama.py: each of those files
knows how to construct itself, but has no business knowing about the
other, or about security_kb or InMemoryRetriever - llm_ollama.py
doesn't import embeddings_ollama.py, and neither imports security_kb.
Putting cross-cutting wiring in either would create a lateral
dependency between two capabilities (generation, embedding) that were
deliberately kept independent.

Not the CLI: this file never touches main.py, scanning, or reporting -
it only wires together already-frozen AI-layer components. Whatever
eventually builds a real entrypoint (paused, per an earlier explicit
decision) can import this file's helpers rather than duplicating
construction logic itself.

Comparison against existing construction patterns: the closest
precedent is main.py's `_get_provider() -> FindingsProvider` - a small
function, in the file that uses the result, typed as the abstract
return type even though it constructs a concrete
MockFindingsProvider(). That precedent is a one-line construction
though; building a working Retriever here is a genuine multi-step
assembly - complex enough to justify its own file rather than a
one-line private helper, but following the exact same "return the
abstraction, not the concrete type" discipline.

create_retriever(entries) takes an already-loaded KB rather than
loading one itself: loading and construction are separate concerns,
the same split already applied throughout this build (parsing from
orchestration, retrieval from prompt building, confidence from
verification). Today there is exactly one KB source
(security_kb.load_knowledge_base()'s bundled fixture), so this split is
harmless overhead - but if a test KB, an updated KB, a remote KB, or a
cached KB shows up later, create_retriever() itself needs no change; a
caller just supplies different entries. create_default_retriever() is
the convenience wrapper over the one KB source that exists today
(create_retriever(load_knowledge_base())) - nothing calls
load_knowledge_base() except that one function.

Helpers return already-frozen abstraction types wherever one exists:
create_retriever()/create_default_retriever() -> Retriever (not
InMemoryRetriever), create_llm_generate_fn() -> LLMFn (a bound
OllamaProvider.generate method, not an OllamaProvider instance -
matching exactly what ai/agents/explainer.py and ai/pipeline.py already
expect to receive, so nothing above this file needs to know
OllamaProvider exists at all). create_embedding_fn() -> EmbeddingFn
naturally returns an abstraction already, since EmbeddingFn is a plain
callable type with no concrete class alternative.

No memoization/caching at this file's level, beyond what each
constructed object already owns itself: InMemoryRetriever's own
_index is "naturally owned by the constructed object" and stays as-is;
this file does not additionally wrap create_retriever()/
create_llm_generate_fn() in a module-level singleton the way
ai/config.py's get_settings() is. Repeated calls build independent
instances - get_settings() itself remains the one process-wide cache
this file relies on, unchanged, not duplicated.

No DI framework, service locator, registry, or plugin system: three
plain functions, each doing exactly the multi-step construction its
name says and nothing else.

Validation is reused, not duplicated: settings.llm_model/
embedding_model are Optional[str]; passing None straight through to
OllamaProvider/make_ollama_embed_fn triggers their own already-built
`if not model: raise ValueError(...)` checks naturally - no new
validation logic needed here.

Reads sentinelai.security_kb.load_knowledge_base(),
.config.get_settings(), .embeddings_ollama.make_ollama_embed_fn(),
.llm_ollama.OllamaProvider, and .retrieval's InMemoryRetriever/
Retriever/EmbeddingFn - all already-approved. Nothing in ai/ imports
this file back (pipeline.py, explainer.py, etc. all receive their
dependencies as parameters and have no reason to import a file that
constructs concrete instances of them). No cycle.

Not part of ai/__init__.py's public surface: that file's frozen
content is `from .pipeline import enrich_findings` only, per the
architecture. This file's helpers are imported directly
(`from sentinelai.ai.factory import create_retriever, ...`).
"""
from sentinelai.security_kb import KnowledgeBaseEntry, load_knowledge_base

from .agents.explainer import LLMFn
from .config import get_settings
from .embeddings_ollama import make_ollama_embed_fn
from .llm_ollama import OllamaProvider
from .retrieval import EmbeddingFn, InMemoryRetriever, Retriever


def create_embedding_fn() -> EmbeddingFn:
    settings = get_settings()
    return make_ollama_embed_fn(host=settings.llm_host, model=settings.embedding_model)


def create_retriever(entries: list[KnowledgeBaseEntry]) -> Retriever:
    """Construct a Retriever from an already-loaded KB. Loading and construction are separate concerns -
    a caller with a test/updated/remote/cached KB source supplies entries directly, no change needed here."""
    embed = create_embedding_fn()
    return InMemoryRetriever(entries, embed)


def create_default_retriever() -> Retriever:
    """create_retriever() over the bundled KB - the convenience path for the one KB source that exists today."""
    return create_retriever(load_knowledge_base())


def create_llm_generate_fn() -> LLMFn:
    settings = get_settings()
    provider = OllamaProvider(host=settings.llm_host, model=settings.llm_model)
    return provider.generate
