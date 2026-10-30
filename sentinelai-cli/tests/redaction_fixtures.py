"""
Planted-secret fixture for the redaction and egress tests, built at import time.

Not a checked-in file on purpose. Every value here is a fake credential in a
real format (AWS, Stripe, Slack, GitHub, Google, PEM, ...), and committing them
literally would trip GitHub push protection and this repository's own GitLeaks
CI gate. So no source line holds a complete secret: each is assembled from a
split prefix plus seeded random characters when the tests run. The seed is
fixed, so the fixture is byte-identical on every run and every machine.

Not named test_*.py so pytest does not collect it, like agent_fakes.py.

PLANTED_TEXT is the fixture document; PLANTED lists {"pattern", "secret"} for
each of the 29 secrets in it, where "secret" is exactly the text that must not
survive redaction (for a PEM block, its body; for a URL, the password).
"""
import random
import string

_rng = random.Random(20261029)
_UP = string.ascii_uppercase
_AL = string.ascii_letters + string.digits
_NL = chr(92) + "n"  # a literal backslash-n, as inside a JSON string

# Prefixes are split so no source line or compiled constant is itself secret-shaped.
_AKIA, _ASIA = "AK" + "IA", "AS" + "IA"
_GHP, _GH_PAT = "gh" + "p_", "github" + "_pat_"
_XOXB, _SK_LIVE, _WHSEC = "xo" + "xb-", "sk" + "_live_", "wh" + "sec_"
_AIZA, _GOCSPX, _JWT_HEAD = "AI" + "za", "GOC" + "SPX-", "ey" + "J"
_SLACK_HOOK = "https://hooks." + "slack.com/services/"
_PRIVATE_KEY = "PRIVATE" + " KEY"


def _r(chars: str, n: int) -> str:
    return "".join(_rng.choice(chars) for _ in range(n))


def _b64(n: int) -> str:
    return _r(_AL + "+/", n)


def _pem_body(lines: int) -> str:
    return "\n".join(_b64(64) for _ in range(lines - 1)) + "\n" + _b64(28) + "=="


def _jwt() -> str:
    header = _JWT_HEAD + "hbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
    return header + "." + _JWT_HEAD + _r(_AL + "-_", 60) + "." + _r(_AL + "-_", 43)


def _pem(label: str, body: str) -> str:
    kind = f"{label} {_PRIVATE_KEY}" if label else _PRIVATE_KEY
    return f"-----BEGIN {kind}-----\n{body}\n-----END {kind}-----"


_secrets: dict = {}


def _put(key: str, pattern: str, value: str) -> str:
    _secrets[key] = (pattern, value)
    return value


aws1 = _put("aws1", "aws-access-key", _AKIA + _r(_UP + string.digits, 16))
aws2 = _put("aws2", "aws-access-key", _ASIA + _r(_UP + string.digits, 16))
awss1 = _put("awss1", "aws-secret-key", _r(_AL + "/+", 40))
awss2 = _put("awss2", "aws-secret-key", _r(_AL + "/+", 40))
rsa = _put("rsa", "pem-private-key", _pem_body(8))
ossh = _put("ossh", "pem-private-key", _pem_body(5))
sa_body = _put("sa_pem", "pem-private-key", _NL + _NL.join(_b64(64) for _ in range(3)) + _NL)
sa_kid = _put("sa_kid", "secret-assignment", _r("0123456789abcdef", 40))
jwt1 = _put("jwt1", "jwt", _jwt())
jwt2 = _put("jwt2", "jwt", _jwt())
bearer = _put("bearer", "bearer-token", _r(_AL, 40))
basic = _put("basic", "authorization-header", _r(_AL, 30) + "==")
pg = _put("pg", "connection-string-password", "Pg" + _r(_AL, 14) + "!x9")
mongo = _put("mongo", "connection-string-password", "M0" + _r(_AL, 18))
redis = _put("redis", "connection-string-password", "rd" + _r(_AL, 20) + "7")
jdbc = _put("jdbc", "url-secret-param", "Jd" + _r(_AL, 16) + "4")
pw1 = _put("pw1", "secret-assignment", "Adm1n-" + _r(_AL, 14) + "#")
api1 = _put("api1", "secret-assignment", _r(_AL, 32))
cs1 = _put("cs1", "secret-assignment", "cS" + _r(_AL, 26) + "%")
envpw = _put("envpw", "secret-assignment", "Env" + _r(_AL, 15) + "9!")
yamltok = _put("yamltok", "secret-assignment", _r(_AL, 28) + "5")
gh1 = _put("gh1", "github-token", _GHP + _r(_AL, 36))
gh2 = _put("gh2", "github-token", _GH_PAT + _r(_AL, 22) + "_" + _r(_AL, 59))
slack = _put("slack", "slack-token", _XOXB + _r(string.digits, 12) + "-" + _r(string.digits, 13) + "-" + _r(_AL, 24))
hook = _put("hook", "slack-webhook", "T" + _r(_UP + string.digits, 10) + "/B" + _r(_UP + string.digits, 10) + "/" + _r(_AL, 24))
stripe = _put("stripe", "stripe-key", _SK_LIVE + _r(_AL, 24))
whsec = _put("whsec", "stripe-key", _WHSEC + _r(_AL, 32))
gapi = _put("gapi", "google-api-key", _AIZA + _r(_AL + "-_", 35))
goc = _put("goc", "google-oauth-client-secret", _GOCSPX + _r(_AL + "-_", 28))

PLANTED_TEXT = f'''# ---- app/db/queries.py: the leak path this phase closes ----
# A Semgrep SQL-injection slice is built around find_user(); the credential
# two lines above it rides along unless every slice is redacted.
import psycopg2

ADMIN_PASSWORD = "{pw1}"
CONN = psycopg2.connect("postgresql://app_user:{pg}@db.internal:5432/app")


def find_user(cursor, username):
    cursor.execute(f"SELECT * FROM users WHERE name = '{{username}}'")
    return cursor.fetchone()


# ---- settings.py ----
import boto3

s3 = boto3.client(
    "s3",
    aws_access_key_id="{aws1}",
    aws_secret_access_key="{awss1}",
)
stripe.api_key = "{stripe}"
GOOGLE_MAPS_KEY = "{gapi}"
SLACK_WEBHOOK = "{_SLACK_HOOK}{hook}"
MONGO_URI = "mongodb+srv://svc-reporting:{mongo}@cluster0.example.net/reports"
gh = Github("{gh2}")
session_token = "{jwt1}"

# ---- ~/.aws/credentials ----
[default]
aws_access_key_id = {aws2}
aws_secret_access_key = {awss2}

# ---- .env ----
DB_PASSWORD={envpw}
GITHUB_TOKEN={gh1}
STRIPE_WEBHOOK_SECRET={whsec}

# ---- config.yaml ----
cache:
  url: redis://:{redis}@cache.internal:6379/0
slack:
  bot: {slack}
  auth_token: {yamltok}
reporting:
  jdbc: jdbc:mysql://reports.internal:3306/rpt?user=rpt&password={jdbc}&ssl=true

# ---- frontend/config.js ----
const clientSecret = `{cs1}`;
const settings = {{ "api_key": "{api1}" }};

# ---- service-account.json ----
{{
  "type": "service_account",
  "private_key_id": "{sa_kid}",
  "private_key": "-----BEGIN {_PRIVATE_KEY}-----{sa_body}-----END {_PRIVATE_KEY}-----{_NL}",
  "client_id": "104857600000000000000",
  "client_secret": "{goc}"
}}

# ---- deploy/id_rsa ----
{_pem("RSA", rsa)}

# ---- deploy/id_ed25519 ----
{_pem("OPENSSH", ossh)}

# ---- scripts/smoke.sh ----
curl -H "Authorization: Bearer {bearer}" https://api.example.com/v1/health
curl -H "Authorization: Basic {basic}" https://legacy.example.com/

# ---- captured request ----
GET /v1/me HTTP/1.1
Host: api.example.com
Authorization: Bearer {jwt2}
'''

PLANTED = [{"pattern": pattern, "secret": value} for pattern, value in _secrets.values()]

# ---- other fake credentials the tests need, built the same way ----
# A separate generator, so adding one here never changes the planted fixture above.
_extra = random.Random("sentinelai-test-credentials")


def _x(chars: str, n: int) -> str:
    return "".join(_extra.choice(chars) for _ in range(n))


# Matched by no redaction pattern: only a GitLeaks position in the SecretIndex can find these.
OPAQUE = _x(_AL, 24)
OPAQUE_2 = _x(_AL, 24)
FAKE_AWS_KEY_ID = _AKIA + _x(_UP + string.digits, 16)
FAKE_GROQ_KEY = "gs" + "k_SENTINEL_" + _x(_AL, 32)
FAKE_GEMINI_TOKEN = "ya" + "29.SENTINEL_" + _x(_AL, 32)
