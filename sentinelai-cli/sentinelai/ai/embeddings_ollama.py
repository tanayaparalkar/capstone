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
retries connection-level failures and transient 5xx statuses up to the
configured total attempt count, waiting a fixed 2 seconds between
attempts. That count defaults to 2 total attempts and is overridable via
SENTINELAI_AI_MAX_ATTEMPTS (passed in as `max_attempts` by
ai/factory.py); the backoff itself stays fixed at 2 seconds and is not
configurable.

HTTP 501 is explicitly excluded from that retry set, and raising the
attempt count does not add it back: it is deterministic here - the model
will not acquire embedding support on a later attempt - and it is the
most common misconfiguration this file exists to explain, so retrying it
would delay that explanation by the backoff for no benefit. HTTP 404 is
excluded on the same grounds, since a model that was never pulled will
not appear between two attempts. A connection failure *is* retried,
since a server that was momentarily unreachable may answer a later
attempt; once the attempts are exhausted, the same actionable message is
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

from sentinelai.contracts import AIConfig

from .egress import assert_safe
from .llm.ollama import OllamaProvider
from .local_endpoint import validate_local_endpoint
from .ollama_http import MAX_ATTEMPTS, read_with_retry
from .retrieval import EmbeddingFn

_REQUEST_TIMEOUT_SECONDS = 120

# Batch and payload ceilings for one /api/embed call.
#
# Both are chosen from what this pipeline actually sends, measured rather than
# guessed. The largest real batch is the knowledge-base index build: 10 entries
# totalling 5,260 characters. A retrieval query is a single text, 48-465
# characters on the benchmark fixture. The limits below leave roughly an order
# of magnitude of headroom over both, so nothing the project sends today is
# affected, while a pathological input - a minified file, a runaway loop
# appending to a batch - is refused rather than posted.
#
# Per-text length is NOT fixed here: it comes from AIConfig.max_context_chars
# (default 2,000) via the scan session, so the one existing knob keeps governing
# how much source text may be used for context, on the embedding leg as well as
# the prompt leg.
_MAX_BATCH_TEXTS = 128          # vs 10 actually sent
_MAX_TOTAL_CHARS = 32_768       # vs 5,260 actually sent

_DEDICATED_MODEL_HINT = (
    "Ollama's /api/embed only accepts a dedicated embedding model; a text-generation model "
    "(llama3, llama3.1, mistral, ...) cannot produce embeddings. Fix it with:\n"
    "    ollama pull nomic-embed-text\n"
    "    export SENTINELAI_AI_EMBEDDING_MODEL=nomic-embed-text\n"
    "Leave SENTINELAI_AI_LLM_MODEL pointing at your generation model - the two settings name "
    "two different models and must not be set to the same value."
)


def _prepare_texts(texts: list[str], model: str) -> list[str]:
    """Redact, bound and gate every text, returning exactly what may be serialized.

    The order matters and mirrors ai/prompt_builder.py's generation path:
    redact first, then bound, then assert. Redacting first is not cosmetic -
    measured on the benchmark fixture, 2 of 23 retrieval queries carry content
    the egress gate itself classifies as a secret (a GitLeaks generic-api-key
    finding and a Bandit B105 hardcoded password), because the query is built
    from the finding's raw_evidence. Gating before redacting would refuse those
    two enrichments instead of masking them.

    Raises AIEnrichmentError for a bounds violation and EgressViolation from the
    gate. Both are operational faults: ai/pipeline.py records them per finding
    and leaves that finding unenriched. Neither becomes a scanner finding, and
    neither message carries a secret value - the gate reports pattern names
    only, and the bounds errors report counts only.
    """
    from .prompt_builder import current_scan_session  # local: avoids an import cycle at module load

    if len(texts) > _MAX_BATCH_TEXTS:
        raise AIEnrichmentError(
            f"embedding batch has {len(texts)} texts; the limit is {_MAX_BATCH_TEXTS}"
        )

    session = current_scan_session()
    limit = session.max_chars

    # AIConfig.max_context_chars is Field(2000, ge=0), so zero is contract-legal.
    # It must not reach the truncation below: text[:0] turns every input into an
    # empty string, the total check then trivially passes, and an embedding
    # request carrying nothing is sent - input silently discarded and a vector
    # returned for content that was never submitted. Refused here instead, as a
    # configuration fault. Not reachable from the CLI today (nothing reads this
    # field from the environment and resolve_config does not set it), so this
    # guards programmatic construction.
    if limit <= 0:
        raise AIEnrichmentError(
            f"max_context_chars must be positive to embed anything; got {limit}"
        )

    # Redacted against the scan's own SecretIndex when one exists, so the query
    # is masked against the same secrets the prompts are. path/line_start are
    # None because a query is assembled from several finding fields rather than
    # lifted from one file region, so there is no single location to anchor the
    # index's positional entries to; the regex patterns still apply in full.
    redacted = [session.redact(text, None, None) for text in texts]

    # Truncation, not rejection, for an over-long single text: a 2,000-character
    # ceiling is a context-budget decision rather than a safety boundary, and
    # silently dropping the text would change retrieval results more than
    # shortening it does. Deterministic - same input, same prefix, every time.
    bounded = [text[:limit] for text in redacted]

    # Separators are counted because the gate below measures "\n".join(bounded),
    # not the texts in isolation. Without them the two limits disagree by up to
    # _MAX_BATCH_TEXTS - 1 characters: a batch inside that window passed this
    # check and was then refused by the gate as a size violation, which reported
    # an egress fault for what is really a bounds overrun. Measured at the
    # boundary - 128 texts of 256 characters is 32,768 here but 32,895 joined.
    joined_length = sum(len(text) for text in bounded) + max(0, len(bounded) - 1)
    if joined_length > _MAX_TOTAL_CHARS:
        raise AIEnrichmentError(
            f"embedding payload is {joined_length} characters across {len(bounded)} texts; "
            f"the limit is {_MAX_TOTAL_CHARS}"
        )

    # One assertion over exactly the text that is about to be serialized.
    #
    # WHAT THIS CALL DOES, AND WHAT IT DOES NOT. On this path assert_safe
    # contributes its SECRETS re-scan and its SIZE ceiling. Its POLICY check is
    # predetermined and cannot fail here: _check_policy compares
    # provider.name against cfg.provider and consults provider.is_local, and all
    # three values are fixed by this function - OllamaProvider.name is the
    # ClassVar "ollama", cfg.provider is the literal "ollama" below, and
    # OllamaProvider.is_local is the ClassVar True. The gate is never told the
    # destination URL; ai/egress/guard.py does not take one.
    #
    # So this call must NOT be read as validating where the request goes. That
    # guarantee comes from make_ollama_embed_fn's validate_local_endpoint(host)
    # above, which refuses any non-loopback host before this closure exists, and
    # from ai/ollama_http.py, which refuses redirects and disables proxies so the
    # validated address is also the address finally contacted.
    #
    # Making the gate genuinely destination-aware means giving it the host and
    # changing its signature, which belongs with the local/hosted transport split
    # planned for P5 rather than here.
    #
    # The config describes the embedding leg, not the generation leg: embeddings
    # always go to the local Ollama server even when generation uses Groq or
    # Gemini, so reusing the generation AIConfig would make the provider-name
    # check fail every hosted run.
    #
    # safety_factor is derived so the gate's size ceiling equals _MAX_TOTAL_CHARS
    # rather than contradicting it: the gate's default factor is tuned for a
    # single prompt, and a legitimate 10-entry batch exceeds it. limit is
    # guaranteed positive by the check above, so this cannot divide by zero.
    cfg = AIConfig(provider="ollama", model=model, egress="deny", max_context_chars=limit)
    assert_safe(
        "\n".join(bounded), cfg, OllamaProvider, safety_factor=_MAX_TOTAL_CHARS / limit
    )
    return bounded


def make_ollama_embed_fn(
    host: str,
    model: str,
    timeout: float = _REQUEST_TIMEOUT_SECONDS,
    max_attempts: int = MAX_ATTEMPTS,
) -> EmbeddingFn:
    """Transport tunables are injected by ai/factory.py from AISettings.

    Both default to the values this module previously hardcoded, so an existing
    two-argument call is byte-for-byte unchanged. Mirrors the same parameters on
    ai/llm_ollama.py's OllamaProvider - one policy, applied identically to both
    Ollama endpoints.
    """
    # Re-checked here, not only in ai/config.py's validator, so that a caller
    # constructing this directly - a test, or any future code path that does not
    # go through AISettings - cannot reach a remote endpoint with the retrieval
    # query, which carries repository source. RemoteEndpointError subclasses
    # ValueError, so the previous contract for a bad host is unchanged.
    validate_local_endpoint(host, setting="host")
    if not model:
        raise ValueError("model must be a non-empty string")
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")

    def embed(texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        # Redaction, bounds and the egress gate run here, inside embed(), rather
        # than in a wrapper around it. Both callers reach the network through
        # this one function - InMemoryRetriever.__init__ for the knowledge-base
        # index and InMemoryRetriever.retrieve for the query built from a
        # finding's raw_evidence - and so does any direct caller of
        # make_ollama_embed_fn(). Enforcing at the wrapper would leave that
        # third path open; enforcing here closes it by construction.
        safe_texts = _prepare_texts(texts, model)

        payload = json.dumps({"model": model, "input": safe_texts}).encode("utf-8")
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
            raw_body = read_with_retry(request, timeout, host, max_attempts)
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
