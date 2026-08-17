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

No exception handling: urllib.error.URLError/HTTPError (network/HTTP
failures), json.JSONDecodeError (malformed response body), and KeyError
(a response missing "response" - a genuine API contract violation) all
propagate as themselves, matching the propagation policy already
established in security_kb/loader.py, ai/agents/explainer.py, and
ai/llm_base.py's own docstring.

The HTTP call goes through ai/ollama_http.py's read_with_retry(), which
retries transient transport failures (connection refused, socket
timeout, 500/502/503/504) exactly once after a fixed 2-second backoff.
That helper re-raises the original exception once retries are exhausted,
so the propagation policy above is unchanged - a caller still sees the
same urllib exception it saw before, just possibly after one extra
attempt. Parsing stays here, outside the retry, because a malformed
response body is deterministic and a second attempt would fail
identically; see that module's docstring.

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

Imports .llm_base.LLMProvider (approved leaf) plus stdlib json, re, and
urllib.request only - no contracts, no security_kb, no ai/config.py.
No cycle possible.
"""
import json
import re
import urllib.request

from .llm_base import LLMProvider
from .ollama_http import read_with_retry

_REQUEST_TIMEOUT_SECONDS = 120
_TEMPERATURE = 0.0


def _strip_markdown_fences(text: str) -> str:
    """Remove Markdown code fences (```json ... ``` or ``` ... ```) from LLM output."""
    match = re.match(r'^```(?:json)?\s*\n(.+?)\n```$', text, re.DOTALL)
    if match:
        return match.group(1)
    return text


class OllamaProvider(LLMProvider):
    """LLMProvider backed by a local Ollama server's /api/generate endpoint."""

    def __init__(self, host: str, model: str) -> None:
        if not host:
            raise ValueError("host must be a non-empty string")
        if not model:
            raise ValueError("model must be a non-empty string")
        self._host = host
        self._model = model

    def generate(self, prompt: str) -> str:
        payload = json.dumps(
            {
                "model": self._model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": _TEMPERATURE},
            }
        ).encode("utf-8")

        request = urllib.request.Request(
            f"{self._host}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        raw_body = read_with_retry(request, _REQUEST_TIMEOUT_SECONDS, self._host)
        body = json.loads(raw_body.decode("utf-8"))

        response_text = body["response"]
        return _strip_markdown_fences(response_text.strip())
