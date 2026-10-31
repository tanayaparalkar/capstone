"""
Loopback-only validation for the Ollama endpoint URL.

Why this exists
---------------
ai/retrieval/context_retrieval.py builds its retrieval query from the
finding's own fields - category, message and `raw_evidence`, which is a
verbatim excerpt of the scanned repository's source - and
ai/embeddings_ollama.py posts that text to `{host}/api/embed`. The host
comes from AISettings.llm_host (SENTINELAI_AI_LLM_HOST), which was
previously checked only for being a non-empty string. A host naming
anything other than this machine therefore sent repository source code
off it, silently, as an ordinary side effect of retrieval.

SentinelAI is a local-first tool: ai/config.py's own docstring describes
Ollama as "a locally-hosted provider with no credential to configure".
This module turns that description into an enforced property rather than
a convention.

Scope, deliberately narrow
--------------------------
One function answering one question: does this URL name a loopback
endpoint on this machine? There is no allow-list, no policy object, and
no escape hatch. An override environment variable would defeat the
purpose, because the exposure being closed is itself an environment
variable - so a remote host is refused outright rather than warned about.

Accepted: scheme http or https, and a host that is either exactly
`localhost` or any IP address Python's `ipaddress` module classifies as
loopback (127.0.0.0/8, or ::1 in the bracketed IPv6 form). Everything
else is rejected: another hostname, a non-loopback address of any kind,
a missing scheme, a non-HTTP scheme, a malformed URL, or an unparsable
port.

Messages name the scheme and host only, never the whole URL, so a
credential someone pasted into the setting is not echoed back into an
exception or a log line.

What this does NOT guarantee
----------------------------
It validates the URL that was configured, not the socket the request
finally reaches. Two gaps are known and are not closed here:

- `localhost` is resolved by the operating system at request time, so a
  hosts-file or resolver change that maps it elsewhere is not detected.
  The IP-literal forms are not exposed to that.
- urllib follows HTTP redirects by default, so a server genuinely
  listening on loopback could redirect a request onward. Preventing that
  needs a non-redirecting opener in ai/ollama_http.py, which is a change
  to the shared transport both Ollama clients use and is out of scope
  for this fix.

Both are recorded rather than papered over: this is a configuration
check, not a complete egress-control system.
"""
import ipaddress
from urllib.parse import urlsplit

_ALLOWED_SCHEMES = frozenset({"http", "https"})
_ALLOWED_HOSTNAMES = frozenset({"localhost"})

_REMEDY = (
    "SentinelAI sends repository source excerpts to this endpoint for embedding, so it must "
    "run on this machine. Point it at your local Ollama server, e.g. http://localhost:11434."
)


class RemoteEndpointError(ValueError):
    """The configured Ollama endpoint is not on this machine.

    A ValueError subclass so that pydantic reports it as an ordinary field
    validation failure, and so a direct caller that already catches
    ValueError for a bad host keeps working unchanged.
    """


def validate_local_endpoint(url: str, *, setting: str = "SENTINELAI_AI_LLM_HOST") -> str:
    """Return `url` unchanged if it names a loopback HTTP endpoint; raise RemoteEndpointError otherwise.

    `setting` only names the source in the error message, so the same check can be
    reported against an environment variable or a function argument.
    """
    if not isinstance(url, str) or not url.strip():
        raise RemoteEndpointError(f"{setting} must be a non-empty URL. {_REMEDY}")

    candidate = url.strip()
    try:
        parts = urlsplit(candidate)
    except ValueError as exc:
        raise RemoteEndpointError(f"{setting} is not a valid URL ({exc}). {_REMEDY}") from exc

    if parts.scheme.lower() not in _ALLOWED_SCHEMES:
        named = parts.scheme.lower() or "no scheme"
        raise RemoteEndpointError(f"{setting} must use http or https; got {named}. {_REMEDY}")

    # .hostname and .port each raise ValueError on a malformed authority
    # (an unterminated IPv6 literal, a non-numeric port), so both are
    # resolved here rather than at the point of use.
    try:
        hostname = parts.hostname
        parts.port
    except ValueError as exc:
        raise RemoteEndpointError(f"{setting} has a malformed host or port ({exc}). {_REMEDY}") from exc

    if not hostname:
        raise RemoteEndpointError(f"{setting} names no host. {_REMEDY}")

    lowered = hostname.lower()
    if lowered in _ALLOWED_HOSTNAMES:
        return candidate

    try:
        address = ipaddress.ip_address(lowered)
    except ValueError:
        # A hostname that is not `localhost`. Not resolved here on purpose:
        # resolving it would both perform a lookup during configuration and
        # make the verdict depend on whatever DNS answers at that moment.
        raise RemoteEndpointError(
            f"{setting} names the remote host {lowered!r}, which is not on this machine. {_REMEDY}"
        ) from None

    if address.is_loopback:
        return candidate

    raise RemoteEndpointError(
        f"{setting} names the non-loopback address {lowered!r}, which is not on this machine. {_REMEDY}"
    )
