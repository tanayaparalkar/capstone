"""
Deterministic pre-flight patch validation: sentinelai/patching/.

Three properties are asserted separately.

*Correctness* - a well-formed diff passes and each malformed shape is rejected
with the specific code that names what was wrong, not a generic failure.

*Restraint* - the validator reports and never repairs. A diff with wrong hunk
counts comes back rejected with the original text untouched; nothing hands back
a corrected diff. This is tested explicitly because silently repairing would
mean applying something the model never proposed.

*Purity* - no model call, no subprocess, no filesystem, no network. The tests
below need none of those to run, which is itself the evidence.

The fixtures at the end are the exact diffs three live generations produced.
They are kept verbatim so the validator is measured against real generator
output rather than against malformed strings invented to be easy to catch.
"""
import ast

import pytest

from sentinelai.contracts import StructuredPatch
from sentinelai.patching import (
    PatchIssueCode,
    validate_python_replacement,
    validate_structured_patch,
    validate_unified_diff,
)

VALID_DIFF = (
    "--- a/app/parser.py\n"
    "+++ b/app/parser.py\n"
    "@@ -10,6 +10,6 @@\n"
    " import yaml\n"
    " \n"
    " def load_config(raw):\n"
    "-    return yaml.load(raw)\n"
    "+    return yaml.safe_load(raw)\n"
    " \n"
    " def main():\n"
)


def _codes(diff):
    return validate_unified_diff(diff).codes


# --- the happy path -----------------------------------------------------------------------------------------------


def test_a_well_formed_diff_is_valid():
    result = validate_unified_diff(VALID_DIFF)
    assert result.is_valid
    assert result.issues == ()


def test_result_is_truthy_when_valid():
    assert bool(validate_unified_diff(VALID_DIFF))
    assert not bool(validate_unified_diff("nonsense"))


def test_omitted_counts_default_to_one():
    """'@@ -5 +5 @@' is legal unified diff and means one line each side."""
    diff = "--- a/f.py\n+++ b/f.py\n@@ -5 +5 @@\n-old\n+new\n"
    assert validate_unified_diff(diff).is_valid


def test_multiple_hunks_are_each_validated():
    diff = (
        "--- a/f.py\n+++ b/f.py\n"
        "@@ -1,2 +1,2 @@\n-a\n+b\n c\n"
        "@@ -10,2 +10,2 @@\n-d\n+e\n f\n"
    )
    assert validate_unified_diff(diff).is_valid


def test_no_newline_marker_counts_toward_neither_side():
    diff = "--- a/f.py\n+++ b/f.py\n@@ -1,1 +1,1 @@\n-old\n\\ No newline at end of file\n+new\n"
    assert validate_unified_diff(diff).is_valid


def test_section_heading_after_the_closing_at_signs_is_permitted():
    diff = "--- a/f.py\n+++ b/f.py\n@@ -1,1 +1,1 @@ def main():\n-old\n+new\n"
    assert validate_unified_diff(diff).is_valid


# --- malformed headers --------------------------------------------------------------------------------------------


def test_empty_diff_is_rejected():
    assert _codes("") == (PatchIssueCode.EMPTY_DIFF,)
    assert _codes("   \n  ") == (PatchIssueCode.EMPTY_DIFF,)


def test_missing_old_file_header_is_rejected():
    assert PatchIssueCode.MISSING_OLD_FILE_HEADER in _codes("+++ b/f.py\n@@ -1,1 +1,1 @@\n-a\n+b\n")


def test_missing_new_file_header_is_rejected():
    assert PatchIssueCode.MISSING_NEW_FILE_HEADER in _codes("--- a/f.py\n@@ -1,1 +1,1 @@\n-a\n+b\n")


def test_both_header_failures_are_reported_together():
    assert set(_codes("@@ -1,1 +1,1 @@\n-a\n+b\n")) == {
        PatchIssueCode.MISSING_OLD_FILE_HEADER,
        PatchIssueCode.MISSING_NEW_FILE_HEADER,
    }


# --- malformed hunks ----------------------------------------------------------------------------------------------


def test_no_hunk_is_rejected():
    assert _codes("--- a/f.py\n+++ b/f.py\n") == (PatchIssueCode.NO_HUNKS,)


def test_malformed_hunk_header_is_rejected():
    assert _codes("--- a/f.py\n+++ b/f.py\n@@ bad @@\n-a\n+b\n") == (
        PatchIssueCode.MALFORMED_HUNK_HEADER,
    )


def test_content_where_a_hunk_header_belongs_is_rejected():
    assert PatchIssueCode.MALFORMED_HUNK_HEADER in _codes("--- a/f.py\n+++ b/f.py\njunk\n")


def test_hunk_count_mismatch_is_rejected():
    diff = "--- a/f.py\n+++ b/f.py\n@@ -5,6 +5,6 @@\n-old\n+new\n"
    result = validate_unified_diff(diff)
    assert result.codes == (PatchIssueCode.HUNK_COUNT_MISMATCH,)
    assert "declares 6 old and 6 new" in result.issues[0].message
    assert "body contains 1 old and 1 new" in result.issues[0].message


def test_mismatch_on_only_one_side_is_still_rejected():
    assert _codes("--- a/f.py\n+++ b/f.py\n@@ -1,2 +1,1 @@\n c\n-old\n+new\n") == (
        PatchIssueCode.HUNK_COUNT_MISMATCH,
    )


# --- malformed prefixes -------------------------------------------------------------------------------------------


def test_missing_line_prefix_is_rejected():
    diff = "--- a/f.py\n+++ b/f.py\n@@ -1,2 +1,2 @@\ndef f():\n-old\n+new\n"
    result = validate_unified_diff(diff)
    assert PatchIssueCode.MALFORMED_LINE_PREFIX in result.codes


def test_empty_body_line_is_rejected_with_an_explanatory_message():
    """A blank context line must be a single space; a truly empty line ends the diff early."""
    diff = "--- a/f.py\n+++ b/f.py\n@@ -1,3 +1,3 @@\n a\n\n-old\n+new\n"
    result = validate_unified_diff(diff)
    assert PatchIssueCode.MALFORMED_LINE_PREFIX in result.codes
    assert "single space" in result.issues[0].message


def test_prefix_failure_suppresses_the_derived_count_failure():
    """One malformed body is one problem, not two; counts are untrustworthy once a prefix is bad."""
    diff = "--- a/f.py\n+++ b/f.py\n@@ -1,9 +1,9 @@\nno-prefix\n-old\n+new\n"
    assert validate_unified_diff(diff).codes == (PatchIssueCode.MALFORMED_LINE_PREFIX,)


def test_issue_line_numbers_index_the_diff_text():
    diff = "--- a/f.py\n+++ b/f.py\n@@ -1,2 +1,2 @@\n a\nbad\n"
    issue = next(i for i in validate_unified_diff(diff).issues if i.code is PatchIssueCode.MALFORMED_LINE_PREFIX)
    assert issue.line == 5


# --- no-op diffs --------------------------------------------------------------------------------------------------


def test_a_context_only_diff_is_rejected_as_changing_nothing():
    diff = "--- a/f.py\n+++ b/f.py\n@@ -1,2 +1,2 @@\n a\n b\n"
    result = validate_unified_diff(diff)
    assert result.codes == (PatchIssueCode.NO_CHANGES,)


# --- Python replacement blocks ------------------------------------------------------------------------------------


def test_valid_python_replacement_passes():
    assert validate_python_replacement("x = 1\n").is_valid


def test_invalid_python_replacement_is_rejected():
    result = validate_python_replacement("def f(:\n")
    assert result.codes == (PatchIssueCode.REPLACEMENT_SYNTAX_ERROR,)


def test_an_indented_fragment_is_accepted():
    """Replacements are usually lifted from inside a function, so they arrive indented."""
    assert validate_python_replacement("    return hashlib.sha256(x).hexdigest()\n").is_valid


def test_an_indented_fragment_that_is_still_invalid_is_rejected():
    assert not validate_python_replacement("    return (\n").is_valid


def test_replacement_is_only_parsed_for_python_targets():
    """Parsing a Go snippet as Python would be a false rejection."""
    patch = StructuredPatch(diff=VALID_DIFF, file="main.go", replacement="func main() {}")
    assert validate_structured_patch(patch).is_valid


def test_python_replacement_is_parsed_for_py_targets():
    patch = StructuredPatch(diff=VALID_DIFF, file="app/f.py", replacement="def f(:")
    assert PatchIssueCode.REPLACEMENT_SYNTAX_ERROR in validate_structured_patch(patch).codes


def test_absent_replacement_is_not_an_error():
    assert validate_structured_patch(StructuredPatch(diff=VALID_DIFF, file="app/f.py")).is_valid


def test_diff_and_replacement_issues_are_both_reported():
    patch = StructuredPatch(
        diff="--- a/f.py\n+++ b/f.py\n@@ -5,6 +5,6 @@\n-old\n+new\n",
        file="app/f.py",
        replacement="def f(:",
    )
    assert set(validate_structured_patch(patch).codes) == {
        PatchIssueCode.HUNK_COUNT_MISMATCH,
        PatchIssueCode.REPLACEMENT_SYNTAX_ERROR,
    }


# --- restraint and purity -----------------------------------------------------------------------------------------


def test_validation_never_returns_a_repaired_diff():
    """Nothing on the result carries patch text; there is no channel through which a fix could be handed back."""
    result = validate_unified_diff("--- a/f.py\n+++ b/f.py\n@@ -5,6 +5,6 @@\n-old\n+new\n")
    assert not hasattr(result, "repaired")
    assert not hasattr(result, "diff")
    assert not result.is_valid


def test_validation_does_not_mutate_its_input():
    diff = "--- a/f.py\n+++ b/f.py\n@@ -5,6 +5,6 @@\n-old\n+new\n"
    before = diff
    validate_unified_diff(diff)
    assert diff == before


def test_validation_is_deterministic():
    diff = "--- a/f.py\n+++ b/f.py\n@@ -9,9 +9,9 @@\nbad\n"
    assert validate_unified_diff(diff).issues == validate_unified_diff(diff).issues


def test_module_makes_no_llm_subprocess_or_filesystem_call():
    """The deterministic/LLM separation, asserted at the import level."""
    import sentinelai.patching.validator as validator

    source = ast.parse(open(validator.__file__).read())
    imported = {
        node.module.split(".")[0]
        for node in ast.walk(source)
        if isinstance(node, ast.ImportFrom) and node.module
    } | {
        alias.name.split(".")[0]
        for node in ast.walk(source)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    for forbidden in ("subprocess", "os", "pathlib", "urllib", "socket", "shutil", "requests"):
        assert forbidden not in imported, f"validator must not import {forbidden}"


# --- real generator output ----------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "label, diff, expected",
    [
        (
            "run 1: context-only, display line numbers retained",
            "--- a/crypto_utils.py\n+++ b/crypto_utils.py\n@@ -5,6 +5,6 @@\n"
            " 5 def hash_password(password: str) -> str:\n"
            " 6     return hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000)",
            PatchIssueCode.HUNK_COUNT_MISMATCH,
        ),
        (
            "run 2: context lines lost their prefix",
            "--- a/crypto_utils.py\n+++ b/crypto_utils.py\n@@ -5,6 +5,6 @@\n\n"
            "def hash_password(password: str) -> str:\n"
            "-    return hashlib.md5(password.encode()).hexdigest()\n"
            "+    return hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000).hex()",
            PatchIssueCode.MALFORMED_LINE_PREFIX,
        ),
        (
            "run 3: prefixes correct, counts wrong",
            "--- a/crypto_utils.py\n+++ b/crypto_utils.py\n@@ -5,6 +5,7 @@\n"
            "  def hash_password(password: str) -> str:\n"
            "-    return hashlib.md5(password.encode()).hexdigest()\n"
            "+    import bcrypt\n"
            "+    return bcrypt.hashpw(password.encode(), bcrypt.gensalt())",
            PatchIssueCode.HUNK_COUNT_MISMATCH,
        ),
    ],
)
def test_real_generator_output_is_rejected_with_the_right_code(label, diff, expected):
    """Every diff three live runs produced. Each is caught, and named correctly."""
    result = validate_unified_diff(diff)
    assert not result.is_valid, label
    assert expected in result.codes, f"{label}: got {result.codes}"
