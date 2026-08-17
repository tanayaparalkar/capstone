"""
Tests for category normalization in sentinelai/ai/verifier.py.

verify_finding() checks whether the generated explanation is on-topic by
matching the finding's category against the text. That match only works
if the category is first reduced to words an explanation would plausibly
contain. Replacing only hyphens left Bandit's underscore-joined test
names (`subprocess_popen_with_shell_equals_true`) as one unsplittable
token, so on-topic Bandit explanations were REJECTED wholesale.

These tests cover the normalization function itself plus the two
end-to-end verdicts that depend on it, so a future change to the stop
list cannot silently alter what verify_finding() concludes.
"""
from sentinelai.ai.llm_response import LLMResponse
from sentinelai.ai.verifier import _normalize_category, verify_finding
from sentinelai.contracts import ScannerFinding, Severity, VerificationStatus


def _finding(category: str) -> ScannerFinding:
    return ScannerFinding(
        finding_id="SENT-001",
        scanner="bandit",
        category=category,
        severity=Severity.HIGH,
        rule_id="B602",
        message="test finding",
        raw_evidence="subprocess.call(cmd, shell=True)",
    )


def _response(explanation: str) -> LLMResponse:
    return LLMResponse(
        title="A finding",
        explanation=explanation,
        remediation="Use an argument list instead.",
    )


def test_normalizes_underscored_bandit_categories():
    cat = _normalize_category("subprocess_popen_with_shell_equals_true")

    assert "subprocess popen with shell equals true" in cat


def test_normalizes_generic_semgrep_security():
    cat = _normalize_category("security")

    assert cat  # must not collapse to an empty string


def test_hyphenated_categories_still_normalize():
    # Pre-existing behaviour for Semgrep/GitLeaks-style hyphenated categories.
    assert _normalize_category("sql-injection") == "sql"


def test_generic_terms_are_dropped_but_distinguishing_ones_kept():
    assert _normalize_category("command-injection") == "command"


def test_empty_category_stays_empty():
    # verify_finding() relies on this to report INSUFFICIENT_EVIDENCE.
    assert _normalize_category("   ") == ""


def test_underscored_category_now_verifies_an_on_topic_explanation():
    # The end-to-end effect of the fix: this previously returned REJECTED because
    # the 40-character token could never appear in generated text.
    status = verify_finding(
        _finding("subprocess_popen_with_shell_equals_true"),
        _response("Calling subprocess with shell=True lets an attacker inject commands."),
    )

    assert status == VerificationStatus.VERIFIED


def test_off_topic_explanation_is_still_rejected():
    # The fix must not turn the check into one that accepts anything.
    status = verify_finding(
        _finding("yaml_load"),
        _response("This code has an inconsistent indentation style."),
    )

    assert status == VerificationStatus.REJECTED
