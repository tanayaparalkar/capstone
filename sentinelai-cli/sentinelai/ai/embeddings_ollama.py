"""
Ollama embedding function - the one concrete EmbeddingFn implementation
for this project, matching the same provider decision as
ai/llm_ollama.py. No EmbeddingProvider ABC: already decided when
ai/retrieval/in_memory_retriever.py and ai/llm_base.py were built -
EmbeddingFn = Callable[[list[str]], list[list[float]]] has exactly one
concrete implementation and exactly one consumer shape to satisfy; an
ABC would add a class, an abstractmethod, and a subclass here for a
single-method interface a plain closure already satisfies at zero cost.

A factory function returning a closure, not a class: EmbeddingFn's
signature has no parameter for host/model, so whatever produces a
conforming callable must bind those two values before handing it off.
InMemoryRetriever is a class because it holds an expensive, reused
index; ai/llm_ollama.py's OllamaProvider is a class only because it
must subclass LLMProvider. Neither reason applies here - there is no
ABC to subclass and nothing expensive to amortize beyond two
configuration strings - so a closure is the minimal correct shape, not
a class whose __call__ would exist purely to dress up the same two
bound values.

Reuses ai/config.py's llm_host rather than inventing embedding_host:
Ollama serves both /api/generate and /api/embed from the same local
server, so a second host field would almost always hold an identical
value to llm_host in practice - not a genuine second piece of
configuration (llm_host's docstring there now notes this explicitly).
embedding_model is a new field, unavoidable: embedding-capable models
(e.g. "nomic-embed-text") are typically different models from
generation models (e.g. "llama3"), so the two cannot share one field.
Follows llm_model's exact pattern - Optional[str], no safe default,
validated non-empty at the point this factory is actually called, not
at AISettings() construction, so unrelated consumers of AISettings
never need SENTINELAI_AI_EMBEDDING_MODEL set just because it exists.

Uses Ollama's /api/embed endpoint (not the older /api/embeddings),
sending the whole `texts` list as one "input" array in a single
request: EmbeddingFn's own signature already commits to "one Python
call in, one Python call out" for a batch of texts - looping the older
per-text endpoint would satisfy that contract too, but /api/embed does
it in one HTTP round trip instead of len(texts), which is simpler, not
added batching beyond what the contract already requires. Response
order is trusted to match input order, per Ollama's documented API
contract - the same kind of trust already placed in json.loads()
following the JSON standard.

Empty input short-circuits locally to [] without a network call - the
same guard InMemoryRetriever.__init__ already applies at its own call
site (`embed(texts) if texts else []`), duplicated defensively here so
this function is correct even if called directly, not only through
that one guarded call site.

Response fields read: only "embeddings". Ignored: "model" and any
duration/count metadata Ollama includes, for the same reason
ai/llm_ollama.py ignores its own response's metadata fields - no
consumer for any of them here.

No exception handling: urllib.error.URLError/HTTPError,
json.JSONDecodeError, and KeyError (a response missing "embeddings")
all propagate as themselves, matching ai/llm_ollama.py's and every
other file's established propagation policy. No SDK-level retry to
defer to, so no retries here either.

Imports only stdlib json/urllib.request, plus EmbeddingFn from
ai/retrieval's public surface (`from .retrieval import EmbeddingFn`) -
not from in_memory_retriever.py. EmbeddingFn is now defined in
ai/retrieval/base.py, the retrieval layer's contract file, and
re-exported through ai/retrieval/__init__.py, the same way Retriever
and RetrievedChunk already are - so this provider depends on the
retrieval layer's contract, not on one implementation's module. No
contracts, no security_kb, no ai/config.py (this factory takes
host/model as explicit parameters, exactly matching OllamaProvider not
reading get_settings() itself). No cycle: ai/retrieval/base.py never
imports anything from ai/ root.
"""
import json
import urllib.request

from .retrieval import EmbeddingFn

_REQUEST_TIMEOUT_SECONDS = 120


def make_ollama_embed_fn(host: str, model: str) -> EmbeddingFn:
    if not host:
        raise ValueError("host must be a non-empty string")
    if not model:
        raise ValueError("model must be a non-empty string")

    def embed(texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        payload = json.dumps({"model": model, "input": texts}).encode("utf-8")
        request = urllib.request.Request(
            f"{host}/api/embed",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=_REQUEST_TIMEOUT_SECONDS) as response:
            body = json.loads(response.read().decode("utf-8"))

        return body["embeddings"]

    return embed
