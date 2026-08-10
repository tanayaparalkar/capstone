"""
LLM provider interface.

LLMProvider is the abstraction whoever assembles this pipeline's
dependencies constructs a concrete instance of, mirroring
providers/base.py's FindingsProvider and ai/retrieval/base.py's
Retriever: today no concrete implementation exists (ai/llm_<provider>.py
is still pending a provider decision); a real implementation is a
drop-in replacement behind this same interface once chosen, per the
project owner's explicit "different LLM providers can be swapped"
requirement. ABC over typing.Protocol for the same reason
FindingsProvider and Retriever are: the one idiom this codebase already
uses for a provider-style swap.

Exactly one abstract method, generate(prompt) -> str: nothing else has
a named consumer. No token-counting, health-check, or model-metadata
method - none is used anywhere in this pipeline, and adding one now
would be exactly the speculative "might be useful later" surface this
build has consistently rejected.

No embeddings here, and no separate EmbeddingProvider ABC exists
either: already decided when ai/retrieval/in_memory_retriever.py was
built - embed(list[str]) -> list[list[float]] is a plain EmbeddingFn
callable, not a class, because (a) the project owner never named
embedding-backend swapping as a requirement the way LLM providers and
Qdrant were named, and (b) generation and embedding are different
capabilities with different shapes; forcing them into one interface
would prevent choosing a local embedding model alongside a remote LLM,
or vice versa, independently.

Deliberately not passed as this ABC type into its one consumer, unlike
Retriever: ai/agents/explainer.py's explain(prompt, generate) takes a
plain LLMFn = Callable[[str], str], not an LLMProvider instance -
already frozen, unchanged by this file's existence. generate(prompt) is
a single-argument, single-return contract; a bound provider.generate
method satisfies LLMFn exactly, so nothing is lost "unwrapping" the
object to its one relevant method at the injection point, and
explainer.py's import surface stays free of this file entirely.
Retriever is passed as the ABC itself because retrieve(query, top_k)
takes two arguments paired with encapsulated index state, which
benefits from being called as a bound method on a stable, explicitly
typed object reference threaded through context_retrieval.py. Same ABC
pattern, same justification (named swap requirement + testability),
different consumption style because the two interfaces' shapes differ.

Synchronous only, no async variant: nothing in this pipeline is async
anywhere - ai/pipeline.py's enrich_findings() is a plain for loop, and
Viraj's CLI commands are synchronous Typer commands. Adding async now
would be speculation with no current consumer, and would ripple
"contagiously" through every already-frozen caller in this chain.

No exception handling, no custom exception type, no retries: an
implementation's own failures (network errors, API errors, rate
limits, auth failures) propagate as whatever the underlying SDK
raises, matching the propagation policy already established throughout
this pipeline (ai/agents/explainer.py, ai/pipeline.py). Retrying a
transient failure is ai/utils/retry.py's job, wrapping calls from
outside this interface - not something this ABC implements or
requires.

generate(prompt) returns plain text, not LLMResponse: parsing/validating
that text into LLMResponse is ai/agents/explainer.py's job. This file
has no import of, or awareness of, ai/llm_response.py. Note for
implementers, not encoded in the type signature: this pipeline's
prompts (ai/prompt_builder.py) ask for a complete JSON response, so a
correct implementation must return the complete string, never a
partial/streamed fragment or a provider-specific response object - the
same invariant ai/agents/explainer.py's LLMFn already documents.

Imports only the stdlib abc module - a pure leaf, no contracts, no
security_kb, no other ai/ file. Consumed later by ai/llm_<provider>.py
implementations and by whatever constructs/selects a concrete
provider. No cycle possible.
"""
from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Source of a text completion for a prompt, swappable behind this interface."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Return the complete text response for prompt. Never partial/streamed output; raise on failure."""
        raise NotImplementedError
