"""Secret redaction: nothing secret survives, nothing ordinary changes, and only pattern names come back.

The core property, checked for every planted secret: no substring of the
secret of length >= 8 appears anywhere in the redacted output. Checking
every 8-character window is sufficient - any longer surviving substring
contains one.
"""
import ast
import json
import random
import string
from collections import Counter
from pathlib import Path

import pytest
from redaction_fixtures import FAKE_AWS_KEY_ID, OPAQUE, OPAQUE_2, PLANTED, PLANTED_TEXT

from sentinelai.ai.redaction import PATTERN_NAMES, SecretIndex, redact_text, redaction_marker
from sentinelai.contracts import ScannerFinding, Severity

DATA = Path(__file__).parent / "data" / "redaction"
CLEAN_BYTES = (DATA / "clean_code.txt").read_bytes()
WINDOW = 8
_PK = " ".join(("PRIVATE", "KEY"))


def _surviving_windows(secret: str, output: str) -> list[str]:
    return [secret[i : i + WINDOW] for i in range(len(secret) - WINDOW + 1) if secret[i : i + WINDOW] in output]


def _assert_names_only(hits: list[str], secrets: list[str]) -> None:
    """Fails if any returned string is, or contains any 8-character piece of, a secret."""
    for hit in hits:
        assert hit in PATTERN_NAMES or hit.startswith("gitleaks:"), hit
        for secret in secrets:
            assert hit != secret
            assert not _surviving_windows(secret, hit)


@pytest.fixture(scope="module")
def planted_result():
    return redact_text(PLANTED_TEXT)


# ---- planted-secrets fixture ----


def test_fixture_plants_every_supported_pattern():
    assert len(PLANTED) >= 20
    assert {entry["pattern"] for entry in PLANTED} == PATTERN_NAMES
    for entry in PLANTED:
        assert len(entry["secret"]) >= WINDOW
        assert entry["secret"] in PLANTED_TEXT


@pytest.mark.parametrize("entry", PLANTED, ids=lambda e: e["pattern"])
def test_no_piece_of_any_planted_secret_survives(planted_result, entry):
    output, _ = planted_result
    assert _surviving_windows(entry["secret"], output) == []


def test_each_planted_secret_is_masked_exactly_once_under_its_own_name(planted_result):
    _, hits = planted_result
    assert Counter(hits) == Counter(entry["pattern"] for entry in PLANTED)


def test_output_carries_a_shaped_marker_for_each_pattern(planted_result):
    output, _ = planted_result
    for name in PATTERN_NAMES:
        assert redaction_marker(name) in output


def test_line_count_is_preserved(planted_result):
    output, _ = planted_result
    assert output.count("\n") == PLANTED_TEXT.count("\n")


def test_hits_are_pattern_names_only(planted_result):
    _, hits = planted_result
    _assert_names_only(hits, [entry["secret"] for entry in PLANTED])


def test_names_only_check_would_catch_a_leaked_value():
    secret = PLANTED[0]["secret"]
    with pytest.raises(AssertionError):
        _assert_names_only([secret], [secret])
    with pytest.raises(AssertionError):
        _assert_names_only(["secret-assignment", secret[:WINDOW]], [secret])


def test_context_around_a_secret_is_kept(planted_result):
    output, _ = planted_result
    assert 'ADMIN_PASSWORD = "«REDACTED:secret-assignment»"' in output
    assert "postgresql://app_user:«REDACTED:connection-string-password»@db.internal:5432/app" in output
    assert f"-----BEGIN RSA {_PK}-----\n«REDACTED:pem-private-key»\n" in output
    assert "cursor.execute(f\"SELECT * FROM users WHERE name = '{username}'\")" in output


def test_redaction_is_idempotent(planted_result):
    output, _ = planted_result
    assert redact_text(output) == (output, [])


# ---- generated property test ----


def _rand(rng, chars, n):
    return "".join(rng.choice(chars) for _ in range(n))


_AL = string.ascii_letters + string.digits
_UPD = string.ascii_uppercase + string.digits
_PW = _AL + "!@#$%^&*-_=+.,:;?~|"

# (expected name, secret generator, line template where {} is the secret)
_GENERATORS = [
    ("aws-access-key", lambda r: "AKIA" + _rand(r, _UPD, 16), 'key_id = "{}"'),
    ("aws-access-key", lambda r: "ASIA" + _rand(r, _UPD, 16), "{}"),
    ("aws-secret-key", lambda r: _rand(r, _AL + "/+", 40), "AWS_SECRET_ACCESS_KEY={}"),
    ("aws-secret-key", lambda r: _rand(r, _AL + "/+", 40), '"SecretAccessKey": "{}",'),
    ("pem-private-key", lambda r: "\n".join(_rand(r, _AL + "+/", 64) for _ in range(4)),
     f"-----BEGIN EC {_PK}-----\n{{}}\n-----END EC {_PK}-----"),
    ("pem-private-key", lambda r: "\n".join(_rand(r, _AL + "+/", 64) for _ in range(3)),
     f"-----BEGIN {_PK}-----\n{{}}"),
    ("jwt", lambda r: "eyJ" + _rand(r, _AL + "-_", 30) + "." + _rand(r, _AL + "-_", 50) + "." + _rand(r, _AL + "-_", 43),
     "client.auth({})"),
    ("bearer-token", lambda r: _rand(r, _AL, 8) + "7" + _rand(r, _AL + "-._~+/", 30), "Authorization: Bearer {}"),
    ("authorization-header", lambda r: _rand(r, _AL, 24) + "=", "Proxy-Authorization: Basic {}"),
    ("authorization-header", lambda r: _rand(r, _AL, 20) + "9", "x-api-key: {}"),
    ("connection-string-password", lambda r: _rand(r, _PW + "/", 18) + "1", "mysql://root:{}@10.0.0.5:3306/app"),
    ("connection-string-password", lambda r: _rand(r, _PW, 18) + "1", "amqp://guest:{}@mq/vhost"),
    ("url-secret-param", lambda r: _rand(r, _AL, 20), "https://api.example.com/v2/items?page=2&token={}"),
    ("secret-assignment", lambda r: _rand(r, _PW, 20), 'DB_PASSWORD = "{}"'),
    ("secret-assignment", lambda r: _rand(r, _PW, 20), "    api_key: str = '{}'"),
    ("secret-assignment", lambda r: _rand(r, _PW, 20), 'if password == "{}":'),
    ("secret-assignment", lambda r: _rand(r, _PW, 20), '{{"apiKey": "{}"}}'),
    ("secret-assignment", lambda r: _rand(r, _AL, 20) + "3", "export SERVICE_TOKEN={}"),
    ("secret-assignment", lambda r: _rand(r, _AL, 20) + "3", "  client_secret: {}"),
    ("github-token", lambda r: "gho_" + _rand(r, _AL, 36), "{}"),
    ("github-token", lambda r: "github_pat_" + _rand(r, _AL + "_", 70), "token {}"),
    ("slack-token", lambda r: "xoxp-" + _rand(r, string.digits, 12) + "-" + _rand(r, _AL, 30), "{}"),
    ("slack-webhook", lambda r: "T" + _rand(r, _UPD, 9) + "/B" + _rand(r, _UPD, 9) + "/" + _rand(r, _AL, 24),
     "https://hooks.slack.com/services/{}"),
    ("stripe-key", lambda r: "rk_test_" + _rand(r, _AL, 30), "{}"),
    ("google-api-key", lambda r: "AIza" + _rand(r, _AL + "-_", 35), "maps.init({})"),
    ("google-oauth-client-secret", lambda r: "GOCSPX-" + _rand(r, _AL + "-_", 28), "{}"),
]
_CASES_PER_GENERATOR = 40


@pytest.mark.parametrize("name,make_secret,template", _GENERATORS, ids=[g[0] for g in _GENERATORS])
def test_generated_secrets_never_survive(name, make_secret, template):
    rng = random.Random(f"{name}:{template}")
    for _ in range(_CASES_PER_GENERATOR):
        secret = make_secret(rng)
        line = template.format(secret)
        text = f"def handler(request):\n    {line}\n    return request\n"
        output, hits = redact_text(text)
        assert _surviving_windows(secret, output) == [], line
        assert name in hits, line
        _assert_names_only(hits, [secret])
        assert output.count("\n") == text.count("\n")


def test_escaped_quote_inside_a_literal_does_not_end_the_secret():
    secret = r'abc\"defghijk\"lmnopqrs'
    output, hits = redact_text(f'password = "{secret}"')
    assert hits == ["secret-assignment"]
    assert "defghijk" not in output and "lmnopqrs" not in output


# ---- false positives ----


def test_clean_code_is_byte_identical_with_no_hits():
    output, hits = redact_text(CLEAN_BYTES.decode("utf-8"))
    assert output.encode("utf-8") == CLEAN_BYTES
    assert hits == []


@pytest.mark.parametrize(
    "line",
    [
        'DATABASE_URL = f"postgresql://{user}:{password}@{host}/{db}"',
        "REDIS_URL=redis://${REDIS_USER}:${REDIS_PASSWORD}@cache:6379",
        'password = "{password}"',
        "api_key = <your-api-key>",
        "DB_PASSWORD=${DB_PASSWORD}",
        "password = settings.password",
        "token: Optional[str] = None",
        "token: SecretStr",
        "TOKEN_TTL = 3600",
        'headers = {"Authorization": f"Bearer {token}"}',
        '"Authorization": auth_header,',
        "# Bearer tokens are short-lived",
        "https://example.com:8443/path",
    ],
)
def test_ordinary_lines_are_untouched(line):
    assert redact_text(line) == (line, [])


def test_crlf_line_endings_are_preserved():
    text = 'a = 1\r\npassword = "hunter2-plus"\r\nb = 2\r\n'
    output, hits = redact_text(text)
    assert output == 'a = 1\r\npassword = "«REDACTED:secret-assignment»"\r\nb = 2\r\n'
    assert hits == ["secret-assignment"]


def test_pem_body_cut_off_by_the_slice_is_masked():
    body = "\n".join(_rand(random.Random(i), _AL + "+/", 64) for i in range(3))
    output, hits = redact_text(f"{body}\n-----END RSA {_PK}-----\n")
    assert hits == ["pem-private-key"] * 3
    assert all(not _surviving_windows(line, output) for line in body.splitlines())


# ---- SecretIndex ----

# Neither of these is matched by any pattern: only the GitLeaks position can find them.
_OPAQUE, _OPAQUE_2 = OPAQUE, OPAQUE_2


def _gitleaks(file, line_start, line_end=None, evidence='GENERIC = "REDACTED"', rule="generic-api-key", scanner="gitleaks"):
    return ScannerFinding(
        finding_id=f"{file}:{rule}:{line_start}",
        scanner=scanner,
        category=rule,
        severity=Severity.HIGH,
        file=file,
        line_start=line_start,
        line_end=line_end,
        rule_id=rule,
        message="Detected a Generic API Key.",
        raw_evidence=evidence,
    )


def _index(*findings):
    index = SecretIndex()
    index.add_from_findings(list(findings))
    return index


def test_opaque_secret_is_not_caught_by_patterns_alone():
    text = f'GENERIC = "{_OPAQUE}"'
    assert redact_text(text) == (text, [])


def test_flagged_line_is_masked_using_the_redacted_evidence_as_a_template():
    index = _index(_gitleaks("/repo/app/settings.py", 41))
    text = f'import os\nGENERIC = "{_OPAQUE}"\nDEBUG = False\n'
    output, hits = index.redact_slice(text, "/repo/app/settings.py", 40)
    assert output == 'import os\nGENERIC = "«REDACTED:gitleaks:generic-api-key»"\nDEBUG = False\n'
    assert hits == ["gitleaks:generic-api-key"]


def test_slice_built_for_a_different_finding_still_masks_the_flagged_line():
    index = _index(_gitleaks("/repo/app/db.py", 12))
    text = (
        "def find_user(cursor, name):\n"
        f'    GENERIC = "{_OPAQUE}"\n'
        "    cursor.execute(f\"SELECT * FROM users WHERE name = '{name}'\")\n"
    )
    output, _ = index.redact_slice(text, "/repo/app/db.py", 11)
    assert _OPAQUE[:WINDOW] not in output
    assert "SELECT * FROM users" in output


def test_template_mismatch_masks_the_whole_line_keeping_indentation():
    index = _index(_gitleaks("/repo/app.py", 2, evidence='OLD_NAME = "REDACTED"'))
    text = f'def f():\n    renamed_since_scan = "{_OPAQUE}"\n'
    output, hits = index.redact_slice(text, "/repo/app.py", 1)
    assert output == "def f():\n    «REDACTED:gitleaks:generic-api-key»\n"
    assert hits == ["gitleaks:generic-api-key"]


def test_missing_evidence_masks_the_whole_line():
    index = _index(_gitleaks("/repo/app.py", 1, evidence=None))
    output, _ = index.redact_slice(f"x = '{_OPAQUE}'  # note\n", "/repo/app.py", 1)
    assert output == "«REDACTED:gitleaks:generic-api-key»\n"


def test_multi_line_finding_masks_every_line_and_keeps_line_count():
    index = _index(_gitleaks("/repo/k.txt", 3, 5, evidence="REDACTED"))
    text = f"one\ntwo\n{_OPAQUE}\n  {_OPAQUE_2}\n{_OPAQUE}\nsix\n"
    output, hits = index.redact_slice(text, "/repo/k.txt", 1)
    assert output == "one\ntwo\n«REDACTED:gitleaks:generic-api-key»\n\n\nsix\n"
    assert hits == ["gitleaks:generic-api-key"]


def test_finding_partly_outside_the_slice_masks_the_overlap():
    index = _index(_gitleaks("/repo/k.txt", 8, 12, evidence="REDACTED"))
    text = f"{_OPAQUE}\n{_OPAQUE_2}\nkept\n"
    output, _ = index.redact_slice(text, "/repo/k.txt", 11)
    assert output == "«REDACTED:gitleaks:generic-api-key»\n\nkept\n"


def test_flagged_line_outside_the_slice_changes_nothing():
    index = _index(_gitleaks("/repo/app.py", 50))
    text = f'GENERIC = "{_OPAQUE}"\n'
    assert index.redact_slice(text, "/repo/app.py", 1) == (text, [])


def test_flag_in_another_file_changes_nothing():
    index = _index(_gitleaks("/repo/a/settings.py", 1))
    text = f'GENERIC = "{_OPAQUE}"\n'
    assert index.redact_slice(text, "/repo/b/settings.py", 1) == (text, [])


@pytest.mark.parametrize(
    "finding_path,slice_path",
    [
        ("/repo/app/settings.py", "app/settings.py"),
        ("C:\\work\\repo\\app\\settings.py", "app/settings.py"),
        ("C:\\work\\repo\\app\\settings.py", "c:/work/repo/app/settings.py"),
        ("./app/settings.py", "/repo/app/settings.py"),
    ],
)
def test_absolute_relative_and_windows_paths_match(finding_path, slice_path):
    index = _index(_gitleaks(finding_path, 1))
    output, _ = index.redact_slice(f'GENERIC = "{_OPAQUE}"\n', slice_path, 1)
    assert _OPAQUE not in output


def test_unredacted_literal_is_masked_in_any_file_and_any_line():
    index = _index(_gitleaks("/repo/.env", 3, evidence=_OPAQUE))
    text = f"# copied here by mistake\nfallback = load() or '{_OPAQUE}'\n"
    output, hits = index.redact_slice(text, "/repo/other/module.py", 200)
    assert _OPAQUE not in output
    assert hits == ["gitleaks:generic-api-key"]


def test_findings_from_other_scanners_are_ignored():
    index = _index(_gitleaks("/repo/app.py", 1, scanner="semgrep", evidence=_OPAQUE))
    text = f'GENERIC = "{_OPAQUE}"\n'
    assert index.redact_slice(text, "/repo/app.py", 1) == (text, [])


def test_findings_without_a_location_still_contribute_their_literal():
    finding = _gitleaks("/x", 1, evidence=_OPAQUE).model_copy(update={"file": None, "line_start": None})
    output, _ = _index(finding).redact_slice(f"v = '{_OPAQUE}'", "/repo/app.py", 1)
    assert _OPAQUE not in output


def test_redact_slice_also_applies_the_patterns():
    index = _index(_gitleaks("/repo/app.py", 1))
    text = f'GENERIC = "{_OPAQUE}"\nDB_PASSWORD = "hunter2-plus"\n'
    output, hits = index.redact_slice(text, "/repo/app.py", 1)
    assert hits == ["gitleaks:generic-api-key", "secret-assignment"]
    assert _OPAQUE not in output and "hunter2-plus" not in output


def test_index_wins_the_name_when_it_overlaps_a_pattern():
    index = _index(_gitleaks("/repo/app.py", 1, evidence='key = "REDACTED"', rule="aws-access-token"))
    _, hits = index.redact_slice(f'key = "{FAKE_AWS_KEY_ID}"', "/repo/app.py", 1)
    assert hits == ["gitleaks:aws-access-token"]


def test_index_hits_are_names_only():
    index = _index(_gitleaks("/repo/app.py", 1), _gitleaks("/repo/.env", 9, evidence=_OPAQUE_2))
    _, hits = index.redact_slice(f'GENERIC = "{_OPAQUE}"\nalso = "{_OPAQUE_2}"\n', "/repo/app.py", 1)
    assert len(hits) == 2
    _assert_names_only(hits, [_OPAQUE, _OPAQUE_2])


def test_hostile_rule_id_cannot_smuggle_text_into_names():
    index = _index(_gitleaks("/repo/app.py", 1, rule=f"rule {_OPAQUE}\n«x»"))
    _, hits = index.redact_slice("anything\n", "/repo/app.py", 1)
    assert len(hits) == 1 and hits[0].startswith("gitleaks:")
    assert all(c in string.ascii_lowercase + string.digits + "._-:" for c in hits[0])


def test_repr_does_not_expose_indexed_literals():
    index = _index(_gitleaks("/repo/.env", 3, evidence=_OPAQUE))
    assert _OPAQUE not in repr(index) and _OPAQUE not in str(index)


def test_invalid_line_start_error_does_not_echo_the_slice():
    with pytest.raises(ValueError) as excinfo:
        SecretIndex().redact_slice(f'GENERIC = "{_OPAQUE}"', "/repo/app.py", 0)
    assert _OPAQUE not in str(excinfo.value)


# ---- module boundaries ----

_REDACTION_SOURCES = sorted((Path(__file__).parent.parent / "sentinelai" / "ai" / "redaction").glob("*.py"))
_FORBIDDEN = ("sentinelai.ai.llm", "sentinelai.ai.egress", "sentinelai.ai.context", "logging")


@pytest.mark.parametrize("source", _REDACTION_SOURCES, ids=lambda p: p.name)
def test_redaction_imports_no_llm_egress_context_or_logging(source):
    imported = []
    for node in ast.walk(ast.parse(source.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            imported += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            # Relative imports may only reach siblings inside ai/redaction/.
            assert node.level <= 1, f"{source.name} imports from outside its package"
            imported.append(node.module or "")
        elif isinstance(node, ast.Call) and getattr(node.func, "id", None) == "print":
            pytest.fail(f"{source.name} calls print()")
    for name in imported:
        assert not any(name == f or name.startswith(f + ".") for f in _FORBIDDEN), name
