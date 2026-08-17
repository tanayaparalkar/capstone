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

Exception handling, unlike ai/llm_ollama.py's deliberate lack of it:
every transport and protocol failure here is translated into
core.errors.AIEnrichmentError. The exception to the propagation policy
is deliberate and specific to this file, because of one failure mode
that is both extremely likely and impossible to diagnose from the raw
error.

Ollama's /api/embed only accepts a dedicated embedding model. Pointing
SENTINELAI_AI_EMBEDDING_MODEL at a text-generation model - the obvious
thing to do when you have already pulled llama3.1:8b for
SENTINELAI_AI_LLM_MODEL and assume one model does both jobs - makes
Ollama answer HTTP 501, and urllib renders that as "HTTP Error 501: Not
Implemented". That message names no model, explains nothing, and
suggests no fix; recovering from it unaided requires knowing Ollama's
model taxonomy, which is not a reasonable thing to expect of someone
running a security scanner. _http_message() below replaces it with the
offending model name, the reason, and the two commands that fix it.

The other branches (unreachable server, non-JSON body, missing
"embeddings" key) are translated for consistency of type, so a caller
depends on one exception rather than on urllib/json internals.

Retries are handled by ai/ollama_http.py's read_with_retry(), which
retries connection-level failures and transient 5xx statuses once, after
a fixed 2-second backoff. HTTP 501 is explicitly excluded from that
retry set: it is deterministic here - the model will not acquire
embedding support on a second attempt - and it is the most common
misconfiguration this file exists to explain, so retrying it would delay
that explanation by the backoff for no benefit. A connection failure is
retried once, since a server that was momentarily unreachable may answer
the second attempt; if it does not, the same actionable message is
raised as before.

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
import urllib.error
import urllib.request

from sentinelai.core.errors import AIEnrichmentError

from .ollama_http import read_with_retry
from .retrieval import EmbeddingFn

_REQUEST_TIMEOUT_SECONDS = 120

_DEDICATED_MODEL_HINT = (
    "Ollama's /api/embed only accepts a dedicated embedding model; a text-generation model "
    "(llama3, llama3.1, mistral, ...) cannot produce embeddings. Fix it with:\n"
    "    ollama pull nomic-embed-text\n"
    "    export SENTINELAI_AI_EMBEDDING_MODEL=nomic-embed-text\n"
    "Leave SENTINELAI_AI_LLM_MODEL pointing at your generation model - the two settings name "
    "two different models and must not be set to the same value."
)


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

        # read_with_retry re-raises the original urllib exception once its retries
        # are exhausted, so the translation below is unchanged - it still sees, and
        # still produces the same actionable message for, exactly the exceptions it
        # handled before.
        try:
            raw_body = read_with_retry(request, _REQUEST_TIMEOUT_SECONDS, host)
        except urllib.error.HTTPError as exc:
            # Must be caught before URLError/OSError: HTTPError subclasses both.
            raise AIEnrichmentError(_http_message(exc, host, model)) from exc
        except (urllib.error.URLError, OSError) as exc:
            raise AIEnrichmentError(
                f"could not reach the Ollama server at {host} for embeddings ({exc}). "
                "Is it running? Start it with `ollama serve`, or set SENTINELAI_AI_LLM_HOST if it "
                "listens somewhere other than the default."
            ) from exc

        # Parsing sits outside the retry deliberately: a malformed body is
        # deterministic, so a second attempt would fail identically.
        try:
            body = json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise AIEnrichmentError(
                f"the Ollama server at {host} returned a non-JSON response to an embedding request ({exc})."
            ) from exc

        try:
            return body["embeddings"]
        except (KeyError, TypeError) as exc:
            raise AIEnrichmentError(
                f"the Ollama embedding response from {host} contained no 'embeddings' field "
                f"(model '{model}'). {_DEDICATED_MODEL_HINT}"
            ) from exc

    return embed


def _http_message(exc: urllib.error.HTTPError, host: str, model: str) -> str:
    """Turn an Ollama HTTP failure into a message naming the model, the cause, and the fix."""
    if exc.code == 501:
        return (
            f"the embedding model '{model}' does not support embeddings "
            f"(HTTP 501 Not Implemented from {host}/api/embed). {_DEDICATED_MODEL_HINT}"
        )
    if exc.code == 404:
        return (
            f"the Ollama server at {host} has no model named '{model}' (HTTP 404). "
            f"Pull it with `ollama pull {model}`, or point SENTINELAI_AI_EMBEDDING_MODEL at a "
            "dedicated embedding model such as nomic-embed-text."
        )
    return (
        f"the embedding request to {host}/api/embed failed with HTTP {exc.code} ({exc.reason}) "
        f"for model '{model}'. {_DEDICATED_MODEL_HINT}"
    )
