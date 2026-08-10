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
from typing import Callable

from ..llm_response import LLMResponse

# generate(prompt) must either return a complete JSON string or raise an exception.
# It must never return partial JSON, streamed fragments, or a provider-specific
# response object (e.g. a GenerateContentResponse or {"text": "..."} wrapper) -
# unwrapping that is whichever concrete provider constructs this callable's job,
# not this file's. Violating this silently turns this file provider-aware.
LLMFn = Callable[[str], str]


def explain(prompt: str, generate: LLMFn) -> LLMResponse:
    raw_text = generate(prompt)
    data = json.loads(raw_text)
    return LLMResponse.model_validate(data)
