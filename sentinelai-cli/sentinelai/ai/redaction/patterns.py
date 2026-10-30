"""
Secret patterns for prompt redaction.

Each SecretPattern names one kind of credential and a regex that finds
it. When the regex has a named group "s", only that group is the secret
and everything around it (the key name, the quotes, the URL scheme and
username, the PEM header) is kept, so the model can still see that a
credential sits there and what kind it is. Without an "s" group the
whole match is the secret.

PATTERNS is ordered most-specific first. ai/redaction/redactor.py
merges overlapping matches and labels the merged span with the
earliest pattern in this order, so a JWT assigned to `token = "..."`
is reported as "jwt", not as a generic "secret-assignment".

Bias is toward masking: a false positive costs the model a little
context, a false negative sends a credential off the machine. The few
filters below (placeholder templates, dotted attribute access, short
integers) exist only to keep ordinary code byte-identical, and each one
rejects something that cannot be a literal secret.

Known gaps, by design rather than oversight:
- A bare high-entropy string with no key name, prefix, or scheme
  around it is not detected here. That is what SecretIndex's GitLeaks
  positions cover.
- A raw "/" in the password of a non-database URL is not matched
  (RFC 3986 requires it percent-encoded); database schemes allow it.
"""
import re
from dataclasses import dataclass
from typing import Callable, Optional

MARKER_PREFIX = "«REDACTED:"


def redaction_marker(name: str) -> str:
    """The shape-preserving stand-in for a masked secret, e.g. «REDACTED:aws-access-key»."""
    return f"{MARKER_PREFIX}{name}»"


@dataclass(frozen=True)
class SecretPattern:
    name: str
    regex: "re.Pattern[str]"
    accept: Optional[Callable[[str], bool]] = None


# Template placeholders and env-var references, never literal secrets:
# ${DB_PASSWORD}, {password}, %(password)s, <your-token>, $API_KEY
_PLACEHOLDER = re.compile(r"\$\{[^}]*\}|\{[^{}]*\}|%\([^)]*\)s|<[^<>]*>|\$[A-Za-z_]\w*")
_DOTTED_NAME = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+")
_SHORT_INT = re.compile(r"\d{1,5}")


def _not_placeholder(value: str) -> bool:
    return not value.startswith(MARKER_PREFIX) and _PLACEHOLDER.fullmatch(value) is None


def _credential_like(value: str) -> bool:
    """Header/bearer values: a token has a digit or is long; 'Bearer tokens' in prose is neither."""
    return _not_placeholder(value) and (len(value) >= 20 or any(c.isdigit() for c in value))


def _unquoted_secret_like(value: str) -> bool:
    """Unquoted `key = value` / `key: value`: reject identifiers, attribute access, type names and small ints."""
    if not _not_placeholder(value) or _DOTTED_NAME.fullmatch(value) or _SHORT_INT.fullmatch(value):
        return False
    return len(value) >= 16 or any(not (c.isalpha() or c == "_") for c in value)


_KEYWORDS = r"(?:password|passwd|secret|api[_-]?key|apikey|token|access[_-]?key|private[_-]?key)"
_KEY_NAME = rf"[\w.-]*{_KEYWORDS}[\w.-]*"
_DB_SCHEMES = r"(?:postgres(?:ql)?|mysql|mariadb|mssql|sqlserver|oracle|mongodb(?:\+srv)?|redis|rediss|amqps?|cockroachdb)"

PATTERNS: tuple[SecretPattern, ...] = (
    SecretPattern(
        "pem-private-key",
        re.compile(
            # The header's own line break stays outside the secret so the marker sits on its own line.
            r"(?s)-----BEGIN(?:[ ][A-Z0-9]+)*[ ]PRIVATE[ ]KEY(?:[ ]BLOCK)?-----(?:\r?\n)?"
            r"(?P<s>.*?)"
            # A slice can cut a key in half: an unterminated block is masked to the end.
            r"(?:-----END(?:[ ][A-Z0-9]+)*[ ]PRIVATE[ ]KEY(?:[ ]BLOCK)?-----|\Z)"
        ),
    ),
    SecretPattern(
        "aws-access-key",
        re.compile(
            r"(?<![A-Za-z0-9])(?:AKIA|ASIA|ABIA|ACCA|AGPA|AIDA|AIPA|ANPA|ANVA|AROA|APKA|ASCA|A3T[A-Z0-9])"
            r"[A-Z0-9]{16}(?![A-Za-z0-9])"
        ),
    ),
    SecretPattern(
        "aws-secret-key",
        re.compile(
            r"(?i)(?:aws[\w.-]{0,30}?(?:secret|sk)|secret_?access_?key)[\w.-]{0,30}?[\"'`]?\]?"
            r"[ \t]*(?::=|=>|=|:)[ \t]*[rbuf]{0,2}[\"'`]?"
            r"(?P<s>[A-Za-z0-9/+]{40})(?![A-Za-z0-9/+=])"
        ),
    ),
    SecretPattern(
        "github-token",
        re.compile(r"(?<![A-Za-z0-9_])(?:gh[pousr]_[A-Za-z0-9]{36,255}|github_pat_[A-Za-z0-9_]{22,255})(?![A-Za-z0-9_])"),
    ),
    SecretPattern(
        "slack-token",
        re.compile(r"(?<![A-Za-z0-9])(?:xox[abposre]-[A-Za-z0-9-]{10,}|xapp-\d-[A-Za-z0-9-]{10,})"),
    ),
    SecretPattern(
        "slack-webhook",
        re.compile(r"https://hooks\.slack\.com/(?:services|workflows|triggers)/(?P<s>[A-Za-z0-9_/+-]{20,})"),
    ),
    SecretPattern(
        "stripe-key",
        re.compile(r"(?<![A-Za-z0-9])(?:(?:sk|rk|pk)_(?:live|test)_[A-Za-z0-9]{10,}|whsec_[A-Za-z0-9]{20,})"),
    ),
    SecretPattern(
        "google-api-key",
        re.compile(r"(?<![A-Za-z0-9_-])AIza[0-9A-Za-z_-]{35}(?![0-9A-Za-z_-])"),
    ),
    SecretPattern(
        "google-oauth-client-secret",
        re.compile(r"(?<![A-Za-z0-9_-])GOCSPX-[A-Za-z0-9_-]{28}(?![A-Za-z0-9_-])"),
    ),
    SecretPattern(
        "jwt",
        re.compile(r"(?<![A-Za-z0-9_-])eyJ[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]*"),
    ),
    SecretPattern(
        "connection-string-password",
        re.compile(
            # Database URLs never carry a path containing "@", so a raw "/" in the password is allowed.
            rf"(?i)(?<![\w+.-]){_DB_SCHEMES}://[^\s:/@\"'`]*:(?P<s>[^\s\"'`]+)@"
        ),
        _not_placeholder,
    ),
    SecretPattern(
        "connection-string-password",
        re.compile(r"(?<![\w+.-])[A-Za-z][A-Za-z0-9+.-]*://[^\s:/@\"'`]*:(?P<s>[^\s\"'`/]+)@"),
        _not_placeholder,
    ),
    SecretPattern(
        "url-secret-param",
        re.compile(
            r"(?i)[?&;](?:password|passwd|pwd|secret|token|access_token|api[_-]?key|apikey)="
            r"(?P<s>[^&\s\"'`;#]+)"
        ),
        _not_placeholder,
    ),
    SecretPattern(
        "bearer-token",
        re.compile(r"(?i)(?<![A-Za-z0-9])bearer[ \t]+(?P<s>[A-Za-z0-9\-._~+/]+=*)"),
        _credential_like,
    ),
    SecretPattern(
        "authorization-header",
        re.compile(
            r"(?i)(?<![\w-])(?:proxy-)?(?:authorization|x-api-key|x-auth-token|api-key)[\"'`]?\]?"
            r"[ \t]*[:=][ \t]*(?:[rbuf]{0,2}[\"'`])?[ \t]*"
            r"(?:(?:basic|bearer|token|digest|bot|apikey)[ \t]+)?"
            r"(?P<s>[A-Za-z0-9\-._~+/]+=*)"
        ),
        _credential_like,
    ),
    SecretPattern(
        "secret-assignment",
        re.compile(
            rf"""(?ix)
            (?<![\w.-])["'`]?{_KEY_NAME}["'`]?\]?               # identifier, dict key or JSON key
            (?:[ \t]*:[ \t]*[A-Za-z_][\w\[\]., |]*?)?             # optional type annotation
            [ \t]*(?:===?|!==?|:=|=>|\?\?=|\|\|=|=|:)[ \t]*      # assignment, comparison or mapping
            [rbuf]{{0,2}}(?P<q>["'`])(?P<s>(?:\\.|(?!(?P=q))[^\\\n])+)(?P=q)
            """
        ),
        _not_placeholder,
    ),
    SecretPattern(
        "secret-assignment",
        re.compile(
            # Unquoted .env / YAML / INI style: the whole rest of the line is the value.
            rf"""(?imx)
            ^[ \t]*(?:export[ \t]+)?["']?{_KEY_NAME}["']?[ \t]*(?:=|:)[ \t]*
            (?P<s>[^\s"'`\#()\[\]{{}},;]+)[ \t]*(?:\#.*)?\r?$
            """
        ),
        _unquoted_secret_like,
    ),
    SecretPattern(
        "pem-private-key",
        re.compile(
            # Body lines of a key whose BEGIN/END fell outside the slice: a whole line of
            # mixed-case base64 with a digit, 40+ characters long.
            r"(?m)^[ \t]*(?P<s>(?=[A-Za-z0-9+/]*\d)(?=[A-Za-z0-9+/]*[a-z])(?=[A-Za-z0-9+/]*[A-Z])"
            r"[A-Za-z0-9+/]{40,}={0,2})[ \t]*\r?$"
        ),
    ),
)

PATTERN_NAMES: frozenset[str] = frozenset(pattern.name for pattern in PATTERNS)
