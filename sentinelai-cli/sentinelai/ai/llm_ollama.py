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
ai/llm_base.py's own docstring. Ollama's plain HTTP API has no SDK-level
retry mechanism to defer to, so "no retries unless already part of the
provider SDK" resolves to no retries at all here.

No HTTP-transport injection parameter on the constructor: that would
itself be exactly the kind of extra abstraction beyond what
LLMProvider requires. Tests fake the network layer via
unittest.mock.patch on urllib.request.urlopen (stdlib), not a
production-code seam.

Imports .llm_base.LLMProvider (approved leaf) plus stdlib json and
urllib.request only - no contracts, no security_kb, no ai/config.py.
No cycle possible.
"""
import json
import urllib.request

from .llm_base import LLMProvider

_REQUEST_TIMEOUT_SECONDS = 120
_TEMPERATURE = 0.0


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
        with urllib.request.urlopen(request, timeout=_REQUEST_TIMEOUT_SECONDS) as response:
            body = json.loads(response.read().decode("utf-8"))

        return body["response"]
