"""
Egress guard: the last check before a payload is handed to a provider.

An assertion, not a transform. assert_safe() never edits the payload; it
either returns None or raises EgressViolation, and anything it cannot
positively verify is a violation. Three checks, in this order:

1. POLICY - egress "deny" with a provider that is not local is refused
   before the payload is looked at: the call is simply not permitted.
   "Local" means provider.is_local is exactly True, and "allow" means
   cfg.egress is exactly "allow"; any other value counts as remote/deny.
   A provider whose name disagrees with cfg.provider is also refused, so
   a wiring mistake cannot route a "local" config to a remote provider.
2. SECRETS - the payload is re-scanned against every pattern in
   ai/redaction/patterns.py. Any match is a violation, reported by
   pattern name only. Also a violation: a «REDACTED:...» marker whose name
   is not one the redactor emits, because a forged marker is exactly what
   unredacted text dressed up as redacted text looks like.
3. SIZE - longer than cfg.max_context_chars * safety_factor is refused.

The duplication with ai/redaction/redactor.py is deliberate and must
not be optimised away. Phase 1 is a best-effort transform; this is a
verification gate. The two share the pattern *definitions* and nothing
else: this module never calls redact_text() or any redactor helper, so a
bug in the redactor's span merging, filtering or replacement cannot also
blind the check that is meant to catch it. That independence is what
lets "no unredacted payload reaches a remote endpoint" hold by
construction rather than by diligence.

The one thing both sides must agree on is what a redacted span looks
like, so a pattern match that consists only of valid redaction markers
and whitespace is not a secret - otherwise every correctly redacted PEM
block would be refused. That rule is re-implemented here, stricter than
the redactor's (marker names must be known names).

Known limit: secrets that only SecretIndex knows about (GitLeaks
positions with no pattern match) cannot be re-verified here, because the
guard sees the payload, not the repository. Those rely on Phase 1 alone.

Never logs, and never puts any part of the payload in an exception.
"""
import re
from typing import Optional

from sentinelai.ai.llm.base import LLMProvider
from sentinelai.ai.redaction.patterns import MARKER_PREFIX, PATTERN_NAMES, PATTERNS
from sentinelai.contracts import AIConfig

# The largest prompt the current pipeline builds is the exploit/remediation
# agent's: ~6.4k characters of fixed instructions and diff format, plus one code
# slice (max_context_chars), five KB chunks (~3.7k) and the finding and prior
# assessment - about 14k at the default max_context_chars of 2000. A factor of
# 8 (16k) admits that and still refuses the case this check exists for, an
# untruncated slice of minified code tens of KB long.
SAFETY_FACTOR = 8.0

_MARKER = re.compile(re.escape(MARKER_PREFIX) + r"(?P<name>[^»\n]{0,80})»")
_MARKER_START = re.compile(re.escape(MARKER_PREFIX))
_GITLEAKS_MARKER_NAME = re.compile(r"gitleaks:[a-z0-9._-]+")


class EgressViolation(Exception):
    """A payload may not be sent. `check` is "policy", "secrets" or "size"; `pattern_names` lists names only."""

    def __init__(self, check: str, message: str, pattern_names: Optional[list[str]] = None) -> None:
        self.check = check
        self.pattern_names = list(pattern_names or [])
        super().__init__(f"egress {check} violation: {message}")


def assert_safe(payload: str, cfg: AIConfig, provider: LLMProvider, *, safety_factor: float = SAFETY_FACTOR) -> None:
    """Raises EgressViolation. Returns None on success."""
    _check_policy(cfg, provider)
    if not isinstance(payload, str):
        raise EgressViolation("secrets", f"payload is {type(payload).__name__}, not str; it cannot be verified")
    _check_secrets(payload)
    _check_size(payload, cfg, safety_factor)


def _check_policy(cfg: AIConfig, provider: LLMProvider) -> None:
    name = getattr(provider, "name", None)
    if name != cfg.provider:
        raise EgressViolation("policy", f"provider {name!r} does not match configured provider {cfg.provider!r}")
    if cfg.egress != "allow" and getattr(provider, "is_local", None) is not True:
        raise EgressViolation("policy", f"egress is denied and provider {name!r} is not local")


def _check_secrets(payload: str) -> None:
    hits = set()
    for pattern in PATTERNS:
        if pattern.name in hits:
            continue
        group = "s" if "s" in pattern.regex.groupindex else 0
        for match in pattern.regex.finditer(payload):
            value = match.group(group)
            if value is None:
                continue
            # Judge only what is left once genuine markers are removed: the shared filters
            # skip any value that merely *starts* with a marker, which would wave through
            # `"«REDACTED:secret-assignment»hunter2"`.
            remainder = _strip_valid_markers(value)
            if not remainder.strip():
                continue
            if pattern.accept is not None and not pattern.accept(remainder):
                continue
            hits.add(pattern.name)
            break
    for match in _MARKER_START.finditer(payload):
        marker = _MARKER.match(payload, match.start())
        if marker is None or not _valid_marker_name(marker.group("name")):
            hits.add("forged-redaction-marker")
            break
    if hits:
        names = sorted(hits)
        raise EgressViolation("secrets", f"payload matches secret pattern(s): {', '.join(names)}", names)


def _check_size(payload: str, cfg: AIConfig, safety_factor: float) -> None:
    limit = cfg.max_context_chars * safety_factor
    if not limit >= 0:  # also catches NaN
        raise EgressViolation("size", "size limit is not a valid number")
    if len(payload) > limit:
        raise EgressViolation("size", f"payload is {len(payload)} characters; the limit is {int(limit)}")


def _strip_valid_markers(value: str) -> str:
    """value with every genuine redaction marker removed; forged markers stay, and are checked as text."""
    return _MARKER.sub(lambda m: "" if _valid_marker_name(m.group("name")) else m.group(0), value)


def _valid_marker_name(name: str) -> bool:
    return name in PATTERN_NAMES or _GITLEAKS_MARKER_NAME.fullmatch(name) is not None
