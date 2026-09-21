"""
Atomic patch application: sentinelai/patching/applicator.py.

This is the first component in the project that writes into a scanned
repository, so the tests are weighted toward what must NOT happen. Roughly: one
group proves a good patch lands exactly, and three groups prove that every
failure leaves the target byte-identical to how it started.

Atomicity is tested by observation rather than by assertion about intent - a
failed write is forced with a real filesystem error and the file is then read
back and compared against its original bytes, and the directory is checked for
leftover temporary files. Asserting "os.replace was called" would test the
implementation; reading the file back tests the guarantee.
"""
import ast
import os
import stat
from pathlib import Path
from unittest.mock import patch as mock_patch

import pytest

from sentinelai.contracts import StructuredPatch
from sentinelai.core.errors import SentinelAIError
from sentinelai.patching import (
    PatchApplicationError,
    PatchApplicator,
    PatchError,
    PatchStrategy,
    PatchTargetNotFoundError,
    PatchValidationError,
    PatchWriteError,
    ReplacementMismatchError,
    validate_structured_patch,
)

ORIGINAL = "import yaml\n\ndef load_config(raw):\n    return yaml.load(raw)\n\ndef main():\n"
PATCHED = "import yaml\n\ndef load_config(raw):\n    return yaml.safe_load(raw)\n\ndef main():\n"

DIFF = (
    "--- a/parser.py\n"
    "+++ b/parser.py\n"
    "@@ -1,6 +1,6 @@\n"
    " import yaml\n"
    " \n"
    " def load_config(raw):\n"
    "-    return yaml.load(raw)\n"
    "+    return yaml.safe_load(raw)\n"
    " \n"
    " def main():\n"
)


def _target(tmp_path: Path, text: str = ORIGINAL, name: str = "parser.py") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def _patch(path: Path, **overrides) -> StructuredPatch:
    defaults = dict(diff=DIFF, file=str(path))
    defaults.update(overrides)
    return StructuredPatch(**defaults)


# --- successful application -------------------------------------------------------------------------------------


def test_applies_a_unified_diff_and_returns_the_new_contents(tmp_path):
    path = _target(tmp_path)
    result = PatchApplicator().apply(_patch(path))

    assert result == PATCHED
    assert path.read_text(encoding="utf-8") == PATCHED


def test_render_computes_the_result_without_touching_disk(tmp_path):
    path = _target(tmp_path)
    assert PatchApplicator().render(_patch(path), ORIGINAL) == PATCHED
    assert path.read_text(encoding="utf-8") == ORIGINAL, "render() must not write"


def test_lines_outside_the_hunk_are_preserved_verbatim(tmp_path):
    original = "header\n" + ORIGINAL + "trailer\n"
    diff = DIFF.replace("@@ -1,6 +1,6 @@", "@@ -2,6 +2,6 @@")
    path = _target(tmp_path, original)
    result = PatchApplicator().apply(_patch(path, diff=diff))

    assert result.startswith("header\n")
    assert result.endswith("trailer\n")


def test_multiple_hunks_apply_in_order(tmp_path):
    original = "a\nb\nc\nd\ne\nf\n"
    diff = (
        "--- a/f.py\n+++ b/f.py\n"
        "@@ -1,2 +1,2 @@\n-a\n+A\n b\n"
        "@@ -5,2 +5,2 @@\n-e\n+E\n f\n"
    )
    path = _target(tmp_path, original)
    assert PatchApplicator().apply(_patch(path, diff=diff)) == "A\nb\nc\nd\nE\nf\n"


def test_crlf_line_endings_are_preserved(tmp_path):
    original = ORIGINAL.replace("\n", "\r\n")
    path = tmp_path / "parser.py"
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write(original)

    result = PatchApplicator().apply(_patch(path))

    assert "\r\n" in result
    assert "\n" not in result.replace("\r\n", ""), "a CRLF file must not be rewritten with LF"


def test_file_permissions_are_preserved(tmp_path):
    path = _target(tmp_path)
    os.chmod(path, 0o640)
    PatchApplicator().apply(_patch(path))
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o640


def test_no_temporary_files_are_left_behind(tmp_path):
    path = _target(tmp_path)
    PatchApplicator().apply(_patch(path))
    assert [p.name for p in tmp_path.iterdir()] == ["parser.py"]


# --- idempotence ------------------------------------------------------------------------------------------------


def test_reapplying_the_same_patch_is_refused_rather_than_duplicated(tmp_path):
    """After the fix is in place the '-' line no longer matches, so the second apply must fail.

    This is the desirable behaviour, not a limitation: a diff is anchored to the
    text it removes, so re-applying would mean matching something that is gone.
    Refusing keeps the file correct; applying "approximately" would corrupt it.
    """
    path = _target(tmp_path)
    PatchApplicator().apply(_patch(path))
    assert path.read_text(encoding="utf-8") == PATCHED

    with pytest.raises(PatchApplicationError):
        PatchApplicator().apply(_patch(path))

    assert path.read_text(encoding="utf-8") == PATCHED, "the failed re-apply must change nothing"


def test_applying_twice_via_render_is_deterministic(tmp_path):
    applicator = PatchApplicator()
    assert applicator.render(_patch(tmp_path / "f.py"), ORIGINAL) == applicator.render(
        _patch(tmp_path / "f.py"), ORIGINAL
    )


# --- failed validation ------------------------------------------------------------------------------------------


def test_an_invalid_patch_is_refused_before_the_file_is_touched(tmp_path):
    path = _target(tmp_path)
    bad = _patch(path, diff="--- a/f.py\n+++ b/f.py\n@@ -1,9 +1,9 @@\n-a\n+b\n")

    with pytest.raises(PatchValidationError) as excinfo:
        PatchApplicator().apply(bad)

    assert "hunk_count_mismatch" in str(excinfo.value)
    assert path.read_text(encoding="utf-8") == ORIGINAL


def test_a_supplied_validation_verdict_is_honoured(tmp_path):
    """The caller may pass a verdict it already computed rather than paying twice."""
    path = _target(tmp_path)
    patch = _patch(path)
    assert PatchApplicator().apply(patch, validation=validate_structured_patch(patch)) == PATCHED


def test_validation_is_computed_when_not_supplied(tmp_path):
    path = _target(tmp_path)
    with pytest.raises(PatchValidationError):
        PatchApplicator().apply(_patch(path, diff="garbage"))


def test_malformed_patch_never_reaches_the_filesystem(tmp_path):
    path = _target(tmp_path)
    with mock_patch("sentinelai.patching.applicator.PatchApplicator._read") as read:
        with pytest.raises(PatchValidationError):
            PatchApplicator().apply(_patch(path, diff="garbage"))
    read.assert_not_called()


# --- missing or unusable target ---------------------------------------------------------------------------------


def test_nonexistent_target_raises(tmp_path):
    with pytest.raises(PatchTargetNotFoundError):
        PatchApplicator().apply(_patch(tmp_path / "absent.py"))


def test_directory_target_raises(tmp_path):
    with pytest.raises(PatchTargetNotFoundError):
        PatchApplicator().apply(_patch(tmp_path))


def test_a_target_outside_the_configured_root_is_refused(tmp_path):
    outside = _target(tmp_path, name="outside.py")
    contained = tmp_path / "repo"
    contained.mkdir()

    with pytest.raises(PatchTargetNotFoundError) as excinfo:
        PatchApplicator(root=contained).apply(_patch(outside))

    assert "outside the configured root" in str(excinfo.value)
    assert outside.read_text(encoding="utf-8") == ORIGINAL


def test_a_target_inside_the_configured_root_is_allowed(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    path = _target(repo)
    assert PatchApplicator(root=repo).apply(_patch(path)) == PATCHED


# --- patch does not fit -----------------------------------------------------------------------------------------


def test_context_mismatch_is_refused_and_changes_nothing(tmp_path):
    path = _target(tmp_path, "completely\ndifferent\ncontent\nhere\nand\nmore\n")

    with pytest.raises(PatchApplicationError) as excinfo:
        PatchApplicator().apply(_patch(path))

    assert "does not match the file at line" in str(excinfo.value)
    assert path.read_text(encoding="utf-8") == "completely\ndifferent\ncontent\nhere\nand\nmore\n"


def test_hunk_running_past_end_of_file_is_refused(tmp_path):
    path = _target(tmp_path, "one\n")
    diff = "--- a/f.py\n+++ b/f.py\n@@ -1,3 +1,3 @@\n one\n-two\n+TWO\n three\n"

    with pytest.raises(PatchApplicationError) as excinfo:
        PatchApplicator().apply(_patch(path, diff=diff))

    assert "the file ends at line" in str(excinfo.value)
    assert path.read_text(encoding="utf-8") == "one\n"


def test_trailing_whitespace_is_significant_when_matching(tmp_path):
    """rstrip() while matching would let a diff apply to a line it does not describe."""
    path = _target(tmp_path, "value = 1   \nnext\n")
    diff = "--- a/f.py\n+++ b/f.py\n@@ -1,2 +1,2 @@\n-value = 1\n+value = 2\n next\n"

    with pytest.raises(PatchApplicationError):
        PatchApplicator().apply(_patch(path, diff=diff))


def test_overlapping_hunks_are_refused(tmp_path):
    path = _target(tmp_path, "a\nb\nc\nd\n")
    diff = (
        "--- a/f.py\n+++ b/f.py\n"
        "@@ -1,3 +1,3 @@\n-a\n+A\n b\n c\n"
        "@@ -2,2 +2,2 @@\n-b\n+B\n c\n"
    )
    with pytest.raises(PatchApplicationError) as excinfo:
        PatchApplicator().apply(_patch(path, diff=diff))
    assert "overlapping hunks" in str(excinfo.value)
    assert path.read_text(encoding="utf-8") == "a\nb\nc\nd\n"


# --- replacement strategy ---------------------------------------------------------------------------------------


def test_replacement_block_is_applied_over_the_given_span(tmp_path):
    path = _target(tmp_path)
    patch = _patch(path, start_line=4, end_line=4, replacement="return yaml.safe_load(raw)")

    result = PatchApplicator().apply(patch, strategy=PatchStrategy.REPLACEMENT)

    assert result == ORIGINAL.replace(
        "    return yaml.load(raw)", "return yaml.safe_load(raw)"
    )


def test_replacement_indentation_is_stripped_by_the_contract(tmp_path):
    """KNOWN LIMITATION, recorded rather than worked around.

    StructuredPatch.replacement is a NonBlankStr, and that alias sets
    strip_whitespace=True, so Pydantic removes the block's leading indentation
    before the applicator ever sees it. A fragment lifted from inside a function
    therefore arrives dedented, and re-indenting it here would mean guessing at
    code the model did not send - exactly the repair this package refuses to do.

    Worse for multi-line blocks: only the FIRST line loses its indent while
    inner lines keep theirs, so the result is inconsistently indented rather
    than uniformly dedented.

    Fixing this needs a contract change (a non-stripping string type for this
    one field), which is out of scope for this phase. Until then the unified
    diff is the dependable representation, which is why it is the default
    strategy. This test exists so the limitation cannot be forgotten, and will
    fail loudly if the contract is ever changed.
    """
    patch = _patch(tmp_path / "f.py", start_line=1, end_line=1, replacement="    indented = 1")

    assert patch.replacement == "indented = 1", "contract strips leading whitespace"

    multi = _patch(tmp_path / "f.py", start_line=1, end_line=1, replacement="    if x:\n        y()")
    assert multi.replacement == "if x:\n        y()", "only the first line is dedented"


def test_replacement_without_a_span_is_a_mismatch(tmp_path):
    path = _target(tmp_path)
    patch = _patch(path, replacement="    pass")

    with pytest.raises(ReplacementMismatchError) as excinfo:
        PatchApplicator().apply(patch, strategy=PatchStrategy.REPLACEMENT)

    assert "cannot be located" in str(excinfo.value)
    assert path.read_text(encoding="utf-8") == ORIGINAL


def test_replacement_span_outside_the_file_is_a_mismatch(tmp_path):
    path = _target(tmp_path)
    patch = _patch(path, start_line=99, end_line=120, replacement="    pass")

    with pytest.raises(ReplacementMismatchError) as excinfo:
        PatchApplicator().apply(patch, strategy=PatchStrategy.REPLACEMENT)

    assert "fall outside the file" in str(excinfo.value)
    assert path.read_text(encoding="utf-8") == ORIGINAL


def test_replacement_with_no_replacement_block_is_a_mismatch(tmp_path):
    path = _target(tmp_path)
    with pytest.raises(ReplacementMismatchError):
        PatchApplicator().apply(_patch(path, start_line=1, end_line=1), strategy=PatchStrategy.REPLACEMENT)


def test_inverted_span_is_a_mismatch(tmp_path):
    path = _target(tmp_path)
    patch = _patch(path, start_line=4, end_line=2, replacement="x = 1")
    with pytest.raises(ReplacementMismatchError):
        PatchApplicator().apply(patch, strategy=PatchStrategy.REPLACEMENT)


def test_diff_is_the_default_strategy(tmp_path):
    """A patch carrying both forms must not apply one or the other by accident."""
    path = _target(tmp_path)
    patch = _patch(path, start_line=1, end_line=6, replacement="wiped")
    assert PatchApplicator().apply(patch) == PATCHED


# --- atomic write behaviour -------------------------------------------------------------------------------------


def test_a_write_failure_leaves_the_original_intact(tmp_path):
    path = _target(tmp_path)

    with mock_patch("sentinelai.patching._filesystem.os.replace", side_effect=OSError("disk full")):
        with pytest.raises(PatchWriteError) as excinfo:
            PatchApplicator().apply(_patch(path))

    assert "disk full" in str(excinfo.value)
    assert path.read_text(encoding="utf-8") == ORIGINAL, "the target must be byte-identical"


def test_a_write_failure_leaves_no_temporary_file(tmp_path):
    path = _target(tmp_path)

    with mock_patch("sentinelai.patching._filesystem.os.replace", side_effect=OSError("disk full")):
        with pytest.raises(PatchWriteError):
            PatchApplicator().apply(_patch(path))

    assert [p.name for p in tmp_path.iterdir()] == ["parser.py"]


def test_the_temporary_file_is_created_beside_the_target(tmp_path):
    """A rename is only atomic within one filesystem, so /tmp would not do."""
    path = _target(tmp_path)
    seen = {}
    real_mkstemp = __import__("tempfile").mkstemp

    def recording(*args, **kwargs):
        seen["dir"] = kwargs.get("dir")
        return real_mkstemp(*args, **kwargs)

    with mock_patch("sentinelai.patching._filesystem.tempfile.mkstemp", side_effect=recording):
        PatchApplicator().apply(_patch(path))

    assert seen["dir"] == str(tmp_path)


def test_contents_are_flushed_and_fsynced_before_the_rename(tmp_path):
    """Durability: a rename over unflushed data can survive the data it points at."""
    path = _target(tmp_path)
    order = []

    with mock_patch("sentinelai.patching._filesystem.os.fsync", side_effect=lambda fd: order.append("fsync")):
        with mock_patch(
            "sentinelai.patching._filesystem.os.replace",
            side_effect=lambda src, dst: order.append("replace") or os.rename(src, dst),
        ):
            PatchApplicator().apply(_patch(path))

    assert order == ["fsync", "replace"]


# --- error hierarchy and purity ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "error",
    [
        PatchValidationError,
        PatchTargetNotFoundError,
        PatchApplicationError,
        ReplacementMismatchError,
        PatchWriteError,
    ],
)
def test_every_error_is_a_sentinelai_error(error):
    """No parallel hierarchy: a caller catching SentinelAIError keeps working."""
    assert issubclass(error, PatchError)
    assert issubclass(error, SentinelAIError)


def test_applicator_makes_no_llm_or_subprocess_call():
    """The deterministic/LLM separation, asserted at the import level."""
    import sentinelai.patching.applicator as applicator

    tree = ast.parse(open(applicator.__file__).read())
    imported = {
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    } | {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    for forbidden in ("subprocess", "urllib", "socket", "requests", "httpx"):
        assert forbidden not in imported, f"applicator must not import {forbidden}"
    assert "ai" not in imported
