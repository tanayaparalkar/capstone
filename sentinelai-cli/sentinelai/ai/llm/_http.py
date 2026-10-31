"""
JSON-over-HTTPS POST shared by the three HTTP providers.

stdlib urllib only, like ai/ollama_http.py: no HTTP client is a declared
dependency of this project.

The timeout is a hard deadline on the whole call. urllib's own timeout
applies to each socket operation separately, so a server that drips one
byte at a time could hold a call open indefinitely. The request runs on
a daemon thread and the caller stops waiting at the deadline; the socket
timeout (set to the same value) ends the abandoned thread soon after.

No retries: whether a rate limit or a 5xx is worth a second attempt is
the caller's decision, made with ProviderError.kind and retry_after.

Every failure becomes a ProviderError built from the status code and
fixed text only (see ai/llm/errors.py). Nothing here logs.
"""
import json
import socket
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Optional

from .errors import ProviderError, ProviderErrorKind

# A completion is a few KB; anything this large is not a completion.
_MAX_RESPONSE_BYTES = 8 * 1024 * 1024
_READ_CHUNK = 64 * 1024


def post_json(
    provider: str, url: str, payload: dict, headers: dict[str, str], timeout_s: float
) -> tuple[Any, float]:
    """POST payload as JSON and return (decoded JSON body, latency in ms)."""
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json", **headers},
        method="POST",
    )
    outcome: dict[str, Any] = {}
    finished = threading.Event()

    def run() -> None:
        try:
            outcome["body"] = _send(request, timeout_s)
        except BaseException as exc:  # handed back to the caller's thread below
            outcome["error"] = exc
        finally:
            finished.set()

    started = time.perf_counter()
    threading.Thread(target=run, name=f"{provider}-request", daemon=True).start()
    if not finished.wait(timeout_s):
        raise ProviderError(provider, ProviderErrorKind.TIMEOUT, f"no response within {timeout_s:g}s")
    latency_ms = max((time.perf_counter() - started) * 1000.0, 0.01)

    if "error" in outcome:
        raise _to_provider_error(provider, outcome["error"]) from None
    try:
        return json.loads(outcome["body"]), latency_ms
    except (UnicodeDecodeError, ValueError):
        raise ProviderError(provider, ProviderErrorKind.INVALID_RESPONSE, "response body is not JSON") from None


def _send(request: urllib.request.Request, timeout_s: float) -> bytes:
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        chunks, size = [], 0
        while True:
            chunk = response.read(_READ_CHUNK)
            if not chunk:
                return b"".join(chunks)
            size += len(chunk)
            if size > _MAX_RESPONSE_BYTES:
                raise _ResponseTooLarge()
            chunks.append(chunk)


class _ResponseTooLarge(Exception):
    pass


def _to_provider_error(provider: str, exc: BaseException) -> ProviderError:
    if isinstance(exc, ProviderError):
        return exc
    if isinstance(exc, _ResponseTooLarge):
        return ProviderError(provider, ProviderErrorKind.INVALID_RESPONSE, "response body too large")
    if isinstance(exc, urllib.error.HTTPError):
        # HTTPError before URLError/OSError: it subclasses both.
        status = exc.code
        if status == 404:
            return ProviderError(provider, ProviderErrorKind.HTTP, "model not found", status=404)
        if status in (401, 403):
            return ProviderError(provider, ProviderErrorKind.AUTH, status=status)
        if status == 429:
            return ProviderError(
                provider, ProviderErrorKind.RATE_LIMIT, status=status, retry_after=_retry_after(exc)
            )
        if status in (408, 504):
            return ProviderError(provider, ProviderErrorKind.TIMEOUT, "timeout", status=status)
        return ProviderError(provider, ProviderErrorKind.HTTP, f"non-200 (HTTP {status})", status=status)
    if isinstance(exc, (socket.timeout, TimeoutError)) or (
        isinstance(exc, urllib.error.URLError) and isinstance(exc.reason, (socket.timeout, TimeoutError))
    ):
        return ProviderError(provider, ProviderErrorKind.TIMEOUT, "timeout")
    if isinstance(exc, (urllib.error.URLError, OSError)):
        reason_str = str(getattr(exc, "reason", exc)).lower()
        if "refused" in reason_str or getattr(getattr(exc, "reason", None), "errno", None) in (111, 10061, 61):
            return ProviderError(provider, ProviderErrorKind.TRANSPORT, "connection refused")
        return ProviderError(provider, ProviderErrorKind.TRANSPORT, type(exc).__name__)
    return ProviderError(provider, ProviderErrorKind.TRANSPORT, f"unexpected {type(exc).__name__}")


def _retry_after(exc: urllib.error.HTTPError) -> Optional[float]:
    value = exc.headers.get("Retry-After") if exc.headers is not None else None
    try:
        return max(0.0, float(value)) if value is not None else None
    except ValueError:
        return None  # an HTTP-date; callers fall back to their own backoff


def validate_timeout(timeout_s: float) -> float:
    if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)) or not 0 < timeout_s < float("inf"):
        raise ValueError("timeout_s must be a positive, finite number of seconds")
    return float(timeout_s)


def optional_count(value: Any) -> Optional[int]:
    """A token count from a response, or None when absent or not a non-negative int."""
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None
