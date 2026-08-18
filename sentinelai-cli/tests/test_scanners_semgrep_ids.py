"""
Tests for Semgrep finding_id assignment (sentinelai/scanners/semgrep.py).

Semgrep only computes real content-based fingerprints for registry-
authenticated runs. This wrapper invokes it anonymously, so every result
comes back with the literal string "requires login" in extra.fingerprint.
That value is truthy, so it was previously accepted as the finding_id and
every Semgrep finding in a scan shared one id.

finding_id is the correlation key joining ScannerFinding to
AIEnrichedFinding (statistics/calculator.py and all four report renderers
each build a `{finding_id: ...}` dict from it), so duplicates silently
collapse entries rather than failing loudly. The uniqueness test below is
therefore the substantive one; the placeholder tests cover the specific
values that caused it.

_convert_result is imported directly because it is a module-level
function, not a method on SemgrepScanner - the scanner class only calls
it from scan(). No subprocess or real Semgrep binary is involved here:
these tests feed it the parsed-JSON dicts Semgrep would have produced,
matching how every other scanner test in this suite works.
"""
from sentinelai.scanners.semgrep import _convert_result


def _raw(**overrides) -> dict:
    base = {
        "check_id": "python.lang.security.audit.eval-detected.eval-detected",
        "path": "app.py",
        "start": {"line": 10},
        "end": {"line": 10},
        "extra": {"message": "Something", "severity": "WARNING"},
    }
    base.update(overrides)
    return base


def test_semgrep_finding_id_is_never_placeholder():
    raw = {
        "check_id": "",
        "path": "app.py",
        "start": {"line": 10},
        "extra": {"message": "Something"},
    }

    finding = _convert_result(raw, 0)

    assert finding.finding_id.startswith("semgrep-")
    assert "requires login" not in finding.finding_id


def test_requires_login_fingerprint_is_rejected():
    # The exact value a real anonymous Semgrep run emits.
    raw = _raw(extra={"message": "Something", "fingerprint": "requires login"})

    finding = _convert_result(raw, 3)

    assert finding.finding_id == "semgrep-3"


def test_placeholder_match_is_case_and_whitespace_insensitive():
    raw = _raw(extra={"message": "Something", "fingerprint": "  Requires Login  "})

    finding = _convert_result(raw, 1)

    assert finding.finding_id == "semgrep-1"


def test_real_fingerprint_is_preserved():
    # A logged-in run produces a genuine content-based fingerprint, which is a
    # better id than the positional fallback and must still be used.
    raw = _raw(extra={"message": "Something", "fingerprint": "a1b2c3d4e5f6"})

    finding = _convert_result(raw, 7)

    assert finding.finding_id == "a1b2c3d4e5f6"


def test_missing_fingerprint_falls_back_to_position():
    finding = _convert_result(_raw(), 2)

    assert finding.finding_id == "semgrep-2"


def test_ids_are_unique_across_findings_sharing_one_rule():
    # Two eval() calls in two files yield two results with an identical check_id
    # and an identical placeholder fingerprint (verified against real Semgrep).
    # Both the placeholder and check_id are therefore unusable as a per-finding key.
    results = [
        _raw(path="a.py", extra={"message": "m", "fingerprint": "requires login"}),
        _raw(path="b.py", extra={"message": "m", "fingerprint": "requires login"}),
    ]

    findings = [_convert_result(result, index) for index, result in enumerate(results)]
    ids = [finding.finding_id for finding in findings]

    assert len(set(ids)) == len(ids), f"duplicate finding_ids: {ids}"
