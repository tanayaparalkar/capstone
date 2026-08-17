"""
Shared HTTP transport for the two Ollama clients, with a retry for
transient failures.

Why this is shared rather than duplicated in ai/llm_ollama.py and
ai/embeddings_ollama.py: those two files are deliberately independent of
each other (neither imports the other), but they perform the *identical*
transport operation against the *same* server, and the retry policy -
what counts as transient, how many attempts, how long to wait - is one
decision that must stay uniform across both endpoints. Duplicating it
would create two places for that single policy to drift, and two places
to keep the "don't retry a deterministic failure" list correct. Both
files importing one leaf module is not a dependency between them.

This module imposes no exception policy on its callers. It returns the
raw response body and re-raises whatever urllib raised once retries are
exhausted, so ai/llm_ollama.py keeps propagating raw exceptions and
ai/embeddings_ollama.py keeps translating them into AIEnrichmentError
with its own actionable messages. Neither behaviour changes.

Transport only, deliberately: the caller decodes and parses the body
itself. In both callers the json.loads() call previously sat *inside*
the `with urlopen(...)` block, so retrying that block as a unit would
also have retried JSON decoding failures. A malformed response body is
deterministic - the same bytes parse the same way on a second attempt -
so retrying it would cost the backoff and fail identically. Splitting
transport from parsing is what keeps the retry scoped to the failures a
retry can actually fix.

Retryable failures are connection-level errors (urllib.error.URLError,
OSError - covering connection refused, DNS failure, socket timeout) plus
the classic transient server/gateway HTTP statuses in
_RETRYABLE_HTTP_STATUS.

HTTP 501 is pointedly NOT in that set despite being a 5xx code: Ollama
returns 501 from /api/embed to mean "this model cannot produce
embeddings," which is a deterministic configuration fact about the
model, not a transient server condition. It is also the single most
common misconfiguration for this project (see ai/embeddings_ollama.py's
_DEDICATED_MODEL_HINT), so retrying it would add the backoff delay to
the one error users hit most, before showing them the same message.
4xx statuses are excluded for the same reason - a missing model (404)
does not appear between two attempts two seconds apart.

Fixed 2 attempts and a fixed 2-second backoff, with no exponential
growth and no jitter: with per-finding failure handling in
ai/pipeline.py, a failing scan can attempt this once per finding, so the
worst-case added delay must stay small and predictable. Jitter and
backoff curves solve thundering-herd contention among many concurrent
clients, which a single local CLI process talking to a local server does
not have.

Note for test authors: any test that simulates a retryable failure must
patch time.sleep, or it will spend the real backoff. See the CONVENTION
paragraph in tests/test_ai_ollama_retry.py.
"""
import logging
import time
import urllib.error
import urllib.request

# Same logger name and silent-by-default convention as the rest of the package.
logger = logging.getLogger("sentinelai")

MAX_ATTEMPTS = 2
RETRY_BACKOFF_SECONDS = 2.0

# Transient server/gateway failures worth a second attempt. 501 is deliberately
# absent - see the module docstring.
_RETRYABLE_HTTP_STATUS = frozenset({500, 502, 503, 504})


def _is_retryable(exc: Exception) -> bool:
    """True for connection-level failures and transient server statuses only."""
    if isinstance(exc, urllib.error.HTTPError):
        # Checked before URLError/OSError: HTTPError subclasses both, and most
        # HTTP statuses are deterministic rather than transient.
        return exc.code in _RETRYABLE_HTTP_STATUS
    return isinstance(exc, (urllib.error.URLError, OSError))


def read_with_retry(request: urllib.request.Request, timeout: int, host: str) -> bytes:
    """POST `request` and return the raw response body, retrying transient failures once.

    `timeout` applies to each attempt independently, preserving each caller's
    existing per-request timeout rather than dividing it across attempts.

    `host` is used only for the retry log line. Nothing about the request or
    response - prompt text, generated content, embedding vectors, evidence
    snippets - is ever logged; only the exception's own message, which carries
    the transport failure reason and no payload.
    """
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except Exception as exc:
            if attempt >= MAX_ATTEMPTS or not _is_retryable(exc):
                raise
            logger.warning(
                "Ollama request to %s failed (attempt %d of %d): %s; retrying in %.0fs",
                host,
                attempt,
                MAX_ATTEMPTS,
                exc,
                RETRY_BACKOFF_SECONDS,
            )
            time.sleep(RETRY_BACKOFF_SECONDS)

    # Unreachable: the loop either returns or raises on its final attempt.
    raise AssertionError("read_with_retry exhausted its loop without returning or raising")
