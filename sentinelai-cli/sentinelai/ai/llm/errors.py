"""
ProviderError: the one exception an LLMProvider raises for an operational fault.

A provider failure is never a security finding. It is raised to the
caller, which records it in scan metadata and carries on without AI
enrichment for that finding.

What the message may contain is deliberately narrow: the provider name,
the failure kind, the HTTP status, and fixed text written in this
package. Never the request body (it holds source code), never the
response body (providers echo prompts and partial API keys back in
error bodies), never a credential. Providers raise it `from None`, so
the underlying urllib exception - which holds the response body - is
not chained into tracebacks either.
"""
from enum import Enum
from typing import Optional


class ProviderErrorKind(str, Enum):
    CONFIGURATION = "configuration"  # missing credential, invalid host, missing optional dependency
    TRANSPORT = "transport"  # connection refused, DNS failure, TLS error
    TIMEOUT = "timeout"  # the hard request deadline passed
    AUTH = "auth"  # HTTP 401/403, or a credential that could not be exchanged for a token
    RATE_LIMIT = "rate_limit"  # HTTP 429
    HTTP = "http"  # any other non-2xx status
    INVALID_RESPONSE = "invalid_response"  # 2xx but not the documented response shape


class ProviderError(Exception):
    """An LLM provider call failed for an operational reason. Safe to log and to record in scan metadata."""

    def __init__(
        self,
        provider: str,
        kind: ProviderErrorKind,
        detail: Optional[str] = None,
        *,
        status: Optional[int] = None,
        retry_after: Optional[float] = None,
    ) -> None:
        self.provider = provider
        self.kind = kind
        self.status = status
        self.retry_after = retry_after
        message = f"{provider}: {kind.value}"
        if status is not None:
            message += f" (HTTP {status})"
        if detail:
            message += f": {detail}"
        super().__init__(message)
