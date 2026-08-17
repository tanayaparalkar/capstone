"""
Calls the LLM once for an already-built prompt, parses the response as
JSON, and validates it into LLMResponse.

Kept as a plain function, not a class: no state to hold across calls -
same reasoning as ai/retrieval/context_retrieval.py and
ai/prompt_builder.py. Unlike InMemoryRetriever, there is no expensive
precomputation here to justify a class; every call does exactly one LLM
call and one parse.

`generate: LLMFn` (Callable[[str], str]) is injected the same way
InMemoryRetriever's `embed` is: this file does not import
ai/llm_base.py's LLMProvider ABC or any provider-specific module. That
ABC still exists (or will) for pipeline.py to select and construct a
concrete provider from, per the project owner's explicit "different LLM
providers can be swapped" requirement - but this function only needs to
know the calling convention (prompt in, text out), not the interface
class itself, exactly as in_memory_retriever.py only needs EmbeddingFn's
shape, not an EmbeddingProvider class.

`prompt: str` arrives already built: this function has no knowledge of
ScannerFinding, RepositoryContext, or RetrievedChunk - building the
prompt is ai/prompt_builder.py's job, kept separate so prompt wording
can change without touching LLM-calling logic and vice versa.

Returns LLMResponse, not AIEnrichedFinding: assembling the full contract
needs fields this function has no way to produce correctly
(confidence_score/confidence_label from ai/confidence_scorer.py,
verification_status from ai/verifier.py, finding_id/scanner_sources
already known by pipeline.py, references sourced from the knowledge
base) - see ai/llm_response.py's own docstring for the full reasoning.

JSON/schema failures (json.JSONDecodeError, pydantic.ValidationError -
the latter including a hallucinated extra field, since LLMResponse
forbids extras) are left to propagate as themselves, not wrapped in a
custom exception - same precedent as security_kb/loader.py: no caller
here needs one unified exception type, and the specific failure reason
is more useful during development than a generic "explainer failed"
message. ai/pipeline.py's per-finding exception handling is what turns
whichever exception surfaces into "this finding wasn't enriched," not
this function.

No retry: retrying belongs where a call might succeed on a second
attempt (a transient LLM API failure) - that's whatever wraps `generate`
before it's passed in (ai/utils/retry.py, or the concrete provider
itself), never here. A JSON/schema failure is deterministic given the
same output text; retrying this function's own parsing step would just
fail identically every time, and would conflate two unrelated failure
modes (a flaky network call vs. malformed structured output) that need
different responses.

No confidence scoring, no verification: those are separate,
independently-evolvable pipeline stages (ai/confidence_scorer.py,
ai/verifier.py) - importing either here would collapse three separable
stages into one file, the same merge the frozen architecture has
consistently avoided elsewhere (confidence and verification were kept
as separate files rather than combined into one scoring/ package).
"""
import json
from typing import Optional, Protocol, Type, TypeVar

from pydantic import BaseModel

from ..llm_response import LLMResponse


class LLMFn(Protocol):
    """Calling convention for a text-generating model.

    A Protocol rather than a Callable alias because the contract grew a second,
    optional parameter: `response_schema` lets a caller constrain decoding to a
    Pydantic model's JSON Schema (ai/llm_ollama.py sends it in Ollama's `format`
    field). Callers that do not need it keep calling `generate(prompt)`, and the
    default of None reproduces the original single-argument behaviour exactly.

    generate() must either return a complete response string or raise. It must
    never return partial JSON, streamed fragments, or a provider-specific
    response object (e.g. a GenerateContentResponse or {"text": ...} wrapper) -
    unwrapping that is whichever concrete provider constructs this callable's
    job, not this file's. Violating this silently turns this file
    provider-aware.
    """

    def __call__(self, prompt: str, response_schema: Optional[Type[BaseModel]] = None) -> str:
        ...


_ModelT = TypeVar("_ModelT", bound=BaseModel)


def run_agent(prompt: str, generate: LLMFn, response_schema: Type[_ModelT]) -> _ModelT:
    """Make one model call and validate the response into `response_schema`.

    The single seam every agent in ai/agents/ goes through, so that "call the
    model, parse JSON, validate, fail loudly on anything else" exists once rather
    than once per agent. The schema is passed to `generate` as well as being
    parsed here: constrained decoding makes a conforming response likely, and
    validation makes a non-conforming one impossible to mistake for success.

    Parse and validation failures propagate as themselves (json.JSONDecodeError,
    pydantic.ValidationError), matching the propagation policy already
    established in this package. They are deterministic, so ai/ollama_http.py
    deliberately does not retry them; ai/pipeline.py's per-finding handler is
    what turns whichever exception surfaces into "this finding wasn't enriched."
    """
    raw_text = generate(prompt, response_schema=response_schema)
    data = json.loads(raw_text)
    return response_schema.model_validate(data)


def explain(prompt: str, generate: LLMFn) -> LLMResponse:
    """Single-call enrichment against the flat LLMResponse schema.

    Superseded by the multi-agent pipeline (ai/pipeline.py), which calls
    run_agent() three times per finding instead. Retained because
    ai/verifier.py's deterministic check still consumes an LLMResponse, and
    because this is the narrowest illustration of the unconstrained-generation
    path that ai/ollama_http.py's retry tests exercise.
    """
    raw_text = generate(prompt)
    data = json.loads(raw_text)
    return LLMResponse.model_validate(data)
