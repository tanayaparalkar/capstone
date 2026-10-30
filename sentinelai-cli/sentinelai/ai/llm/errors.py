"""
ProviderError: the one exception an LLMProvider raises for an operational fault.
"""
from enum import Enum

from .base import ProviderError


class ProviderErrorKind(str, Enum):
    CONFIGURATION = "configuration"  # missing credential, invalid host, missing optional dependency
    TRANSPORT = "transport"  # connection refused, DNS failure, TLS error
    TIMEOUT = "timeout"  # the hard request deadline passed
    AUTH = "auth"  # HTTP 401/403, or a credential that could not be exchanged for a token
    RATE_LIMIT = "rate_limit"  # HTTP 429
    HTTP = "http"  # any other non-2xx status
    INVALID_RESPONSE = "invalid_response"  # 2xx but not the documented response shape


__all__ = ["ProviderError", "ProviderErrorKind"]
