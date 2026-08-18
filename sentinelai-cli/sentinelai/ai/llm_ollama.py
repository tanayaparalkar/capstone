"""
Ollama LLMProvider implementation - the one concrete LLM provider for
this project, matching the team's decision. No second provider exists
here; adding one now with nothing to swap to would be speculative work
this build has consistently rejected elsewhere.

A class only because "inherit from LLMProvider" requires it - not
because it holds expensive state the way ai/retrieval/in_memory_retriever.py's
InMemoryRetriever does (host/model are cheap strings, nothing
precomputed). Same justification providers/mock_provider.py's
MockFindingsProvider has: ABC compliance, not amortized computation.

Constructor takes host/model as explicit parameters, not
get_settings() internally - matching InMemoryRetriever's pattern and
the "orchestrator reads config, leaves receive explicit values"
principle already established throughout this pipeline. Whoever
eventually constructs this provider reads
get_settings().llm_host/llm_model and passes them in. Both are
validated non-empty here and raise immediately if not - llm_model has
no safe default at the config level (see ai/config.py), so this is
where "you must actually configure this" surfaces.

Structured output: generate() takes an optional `response_schema` - a
Pydantic model class whose model_json_schema() is sent in Ollama's
`format` field, constraining decoding to that schema. This is what makes
the multi-agent pipeline's nested outputs (ai/agents/schemas.py)
practical on a small local model: asking an 8B model to emit a correct
nested object from prose instructions alone is unreliable, whereas
constrained decoding makes the shape a property of the request rather
than of the model's diligence. The schema instruction is still included
in the prompt as well, because `format` constrains only the *shape* -
the prompt is what conveys what each field is supposed to contain.

When no schema is passed, no `format` key is sent and the request is
byte-identical to the previous unconstrained one, so existing callers
are unaffected.

generate(prompt) -> str: POSTs to {host}/api/generate with
stream: false (Ollama's default streaming mode returns newline-delimited
JSON chunks, not one object - would silently break json.loads(), and
streaming is explicitly out of scope here) and options.temperature: 0.0
(hardcoded as a design decision, not sourced from config - minimizing
sampling randomness for a security-analysis tool, the same category of
fixed choice as the confidence scorer's averaging formula or the
verifier's stopword list, not a deployment-varying value). No seed is
set: nothing in the frozen architecture requires bit-reproducible LLM
output across process runs, and picking an arbitrary seed would itself
be exactly the kind of unjustified invented literal already rejected
for confidence scoring's "scanner_reliability".

Response fields read: only "response" (the generated text). Ignored:
model, created_at, done, total_duration, load_duration,
prompt_eval_count/duration, eval_count/duration - no consumer for any
of them. "context" (Ollama's token-context array for continuing a
conversation) is deliberately never read or threaded into a later
call - using it would mean maintaining conversation history across
calls, out of scope here.

Exception policy: transport and protocol failures are translated into
core.errors.AIEnrichmentError with a message naming the host, the model,
the cause, and the fix - the same treatment ai/embeddings_ollama.py has
always given the /api/embed endpoint, and deliberately the same wording
patterns, so the two endpoints do not fail in two different voices.

This replaced raw propagation, which was measured to be genuinely
unhelpful: with SENTINELAI_AI_LLM_MODEL set to a model that was not
pulled, every finding failed with "HTTP Error 404: Not Found" and the
final error named neither the model nor the fix - while Ollama's own
response body had said, in as many words, "model 'llama3.1' not found".
See _error_detail below for why that sentence was being thrown away.

The translation is confined to the failure path. A successful generation
returns exactly what it returned before, byte for byte.

The HTTP call goes through ai/ollama_http.py's read_with_retry(), which
retries transient transport failures (connection refused, socket
timeout, 500/502/503/504) after a fixed 2-second backoff. That helper
re-raises the original exception once retries are exhausted, and the
translation above then converts it - so retrying is invisible to callers
except in timing. Parsing stays here, outside the retry, because a
malformed response body is deterministic and a second attempt would fail
identically; see that module's docstring.

Per-attempt timeout and attempt count are constructor parameters
defaulting to the values this module used to hardcode. ai/factory.py
supplies them from AISettings; nothing here reads configuration, so the
dependency note below still holds.

No HTTP-transport injection parameter on the constructor: that would
itself be exactly the kind of extra abstraction beyond what
LLMProvider requires. Tests fake the network layer via
unittest.mock.patch on urllib.request.urlopen (stdlib), not a
production-code seam.

_strip_markdown_fences: ai/prompt_builder.py does not currently
instruct the model to return JSON at all (a separate, known gap - see
docs/CONTRACTS.md and the project's own paper-audit notes), so an
instruct-tuned model is free to choose its own formatting, and
llama3.1:8b was observed wrapping its response in a ```json ... ```
(or bare ``` ... ```) code fence - a stylistic default for
code-shaped content, not malformed output. ai/agents/explainer.py's
json.loads() has no tolerance for that, so the fence is stripped here,
at the one place this file already owns "the exact text Ollama
returned." Handles a fence with or without the `json` language tag, and
leaves already-fenceless text (or anything else it doesn't recognize)
untouched, returned as-is for json.loads() to accept or reject on its
own terms - this function does not attempt JSON validation itself.
Confirmed by direct testing to cover a fence with nothing else in the
response; a fence preceded or followed by any other text (e.g. "Here is
the analysis:\n```json...") is not matched and passes through
unchanged, same as today - a different, non-anchored implementation
would be needed for that shape if it's observed in practice.

Imports .llm_base.LLMProvider and .ollama_http (approved leaves),
core.errors.AIEnrichmentError (the project's existing error taxonomy,
already imported the same way by ai/embeddings_ollama.py), plus stdlib
json, re, and urllib only - no security_kb and, deliberately, no
ai/config.py. No cycle possible.
"""
import json
import re
import urllib.error
import urllib.request
from typing import Any, Dict, Optional, Type

from pydantic import BaseModel

from sentinelai.core.errors import AIEnrichmentError

from .llm_base import LLMProvider
from .ollama_http import MAX_ATTEMPTS, read_with_retry

_REQUEST_TIMEOUT_SECONDS = 120
_TEMPERATURE = 0.0

# Cap on how much of an error response body is read and quoted. Ollama's error
# bodies are a single short JSON object; the limit exists so a server returning
# something unexpected (an HTML error page, a truncated stream) cannot paste an
# unbounded amount of text into a user-facing message.
_MAX_ERROR_BODY_BYTES = 2048


def _strip_markdown_fences(text: str) -> str:
    """Remove Markdown code fences (```json ... ``` or ``` ... ```) from LLM output."""
    match = re.match(r'^```(?:json)?\s*\n(.+?)\n```$', text, re.DOTALL)
    if match:
        return match.group(1)
    return text


def _error_detail(exc: urllib.error.HTTPError) -> Optional[str]:
    """Ollama's own `error` string from a failed response body, if it can be read.

    Ollama reports the actual cause in the *body* of a non-2xx response - e.g.
    HTTP 404 carries {"error": "model 'llama3.1' not found"} - and urllib raises
    HTTPError before any caller reads it, so that sentence was previously
    discarded and the user saw only "HTTP Error 404: Not Found".

    HTTPError is itself a readable file object, so the body is recoverable here.
    Everything about reading it is best-effort: a body that is missing, already
    consumed, over-long, not JSON, or not shaped as expected yields None and the
    caller falls back to the status line. Surfacing a message must never be able
    to raise a second exception on top of the first.

    Only the `error` field is ever returned. The response body of a *successful*
    generation carries model output, and nothing in this function is reachable
    for one - this runs solely on the HTTPError path.
    """
    try:
        raw = exc.read(_MAX_ERROR_BODY_BYTES)
    except Exception:
        return None
    if not raw:
        return None

    try:
        body = json.loads(raw.decode("utf-8", errors="replace"))
    except (json.JSONDecodeError, AttributeError):
        return None

    detail = body.get("error") if isinstance(body, dict) else None
    if not isinstance(detail, str):
        return None
    detail = detail.strip()
    return detail or None


def _http_message(exc: urllib.error.HTTPError, host: str, model: str) -> str:
    """Turn an Ollama generation failure into a message naming the model, the cause, and the fix.

    Deliberately mirrors ai/embeddings_ollama.py's `_http_message`: the two
    endpoints fail in the same ways for the same reasons, and a user who has
    seen one message should recognise the other. The 404 text differs only
    where the fix differs - a generation model is pulled by name, whereas the
    embedding message additionally steers users toward a dedicated embedding
    model.
    """
    detail = _error_detail(exc)

    if exc.code == 404:
        message = (
            f"the Ollama server at {host} has no model named '{model}' (HTTP 404). "
            f"Pull it with `ollama pull {model}`, or set SENTINELAI_AI_LLM_MODEL to a model "
            "that is already installed (`ollama list` shows them)."
        )
    else:
        message = (
            f"the generation request to {host}/api/generate failed with HTTP {exc.code} "
            f"({exc.reason}) for model '{model}'."
        )

    # Ollama's own wording is appended rather than substituted: the sentence above
    # carries the fix, which the raw server string never does.
    return f"{message} Ollama reported: {detail}" if detail else message


class OllamaProvider(LLMProvider):
    """LLMProvider backed by a local Ollama server's /api/generate endpoint."""

    def __init__(
        self,
        host: str,
        model: str,
        timeout: float = _REQUEST_TIMEOUT_SECONDS,
        max_attempts: int = MAX_ATTEMPTS,
    ) -> None:
        """Transport tunables are injected, not read from ai/config.py here.

        This module deliberately imports no configuration (see the module
        docstring's dependency note); ai/factory.py is the composition root that
        reads AISettings and passes the values down. Both parameters default to
        the values this module previously hardcoded, so an existing
        two-argument construction is byte-for-byte unchanged.
        """
        if not host:
            raise ValueError("host must be a non-empty string")
        if not model:
            raise ValueError("model must be a non-empty string")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        self._host = host
        self._model = model
        self._timeout = timeout
        self._max_attempts = max_attempts

    def generate(self, prompt: str, response_schema: Optional[Type[BaseModel]] = None) -> str:
        """Return the model's complete response text for `prompt`.

        When `response_schema` is supplied, its JSON Schema is sent in Ollama's
        `format` field, constraining decoding so the response conforms to that
        schema. The schema instruction also stays in the prompt: `format`
        constrains the shape, while the prompt is what tells the model what each
        field is supposed to mean.

        Omitting `response_schema` reproduces the previous request byte for byte -
        no `format` key is sent at all - so unconstrained callers are unaffected.
        """
        request_body: Dict[str, Any] = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": _TEMPERATURE},
        }
        if response_schema is not None:
            request_body["format"] = response_schema.model_json_schema()

        payload = json.dumps(request_body).encode("utf-8")

        request = urllib.request.Request(
            f"{self._host}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        # read_with_retry re-raises the original urllib exception once its retries
        # are exhausted, so the retry policy is untouched by this translation - it
        # still sees exactly the exceptions it saw before, just after the same
        # number of attempts.
        try:
            raw_body = read_with_retry(request, self._timeout, self._host, self._max_attempts)
        except urllib.error.HTTPError as exc:
            # Must be caught before URLError/OSError: HTTPError subclasses both.
            raise AIEnrichmentError(_http_message(exc, self._host, self._model)) from exc
        except (urllib.error.URLError, OSError) as exc:
            raise AIEnrichmentError(
                f"could not reach the Ollama server at {self._host} for generation ({exc}). "
                "Is it running? Start it with `ollama serve`, or set SENTINELAI_AI_LLM_HOST if it "
                "listens somewhere other than the default."
            ) from exc

        # Parsing sits outside the retry deliberately: a malformed body is
        # deterministic, so a second attempt would fail identically.
        try:
            body = json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise AIEnrichmentError(
                f"the Ollama server at {self._host} returned a non-JSON response to a generation "
                f"request ({exc})."
            ) from exc

        try:
            response_text = body["response"]
        except (KeyError, TypeError) as exc:
            raise AIEnrichmentError(
                f"the Ollama generation response from {self._host} contained no 'response' field "
                f"(model '{self._model}')."
            ) from exc

        return _strip_markdown_fences(response_text.strip())
