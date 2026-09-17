"""
Finding-anchored source windows: backend/code_context.py.

Two properties matter here and are tested separately. The window must be
*correct* - the right lines, numbered from the right place, clamped at both
file boundaries - because a window that is off by one puts the marker on the
wrong line and tells the model to fix code that is fine. And unavailability
must be *silent* - every unreadable case returns None rather than raising,
because this feeds a prompt and a missing file must degrade the prompt, not
fail the scan.
"""
from pathlib import Path

from sentinelai.backend.code_context import (
    DEFAULT_CONTEXT_LINES,
    CodeContext,
    extract_code_context,
)

# 10 numbered lines, so an assertion can name the exact content it expects.
_NUMBERED = "\n".join(f"line{n}" for n in range(1, 11)) + "\n"


def _write(tmp_path: Path, text: str = _NUMBERED, name: str = "sample.py") -> str:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


# --- window correctness ---------------------------------------------------------------------------------------------


def test_window_is_centred_on_the_finding(tmp_path):
    context = extract_code_context(_write(tmp_path), line_start=5, context_lines=2)

    assert (context.start_line, context.end_line) == (3, 7)
    assert context.lines == ("line3", "line4", "line5", "line6", "line7")


def test_focus_span_is_recorded_separately_from_the_window(tmp_path):
    """The marker depends on focus_*, the numbering on start_line - they are different things."""
    context = extract_code_context(_write(tmp_path), line_start=4, line_end=6, context_lines=1)

    assert (context.start_line, context.end_line) == (3, 7)
    assert (context.focus_start, context.focus_end) == (4, 6)


def test_window_clamps_at_the_start_of_file(tmp_path):
    """A finding on line 1 must not produce a negative or zero start line."""
    context = extract_code_context(_write(tmp_path), line_start=1, context_lines=5)

    assert context.start_line == 1
    assert context.lines[0] == "line1"


def test_window_clamps_at_end_of_file(tmp_path):
    context = extract_code_context(_write(tmp_path), line_start=9, context_lines=5)

    assert context.end_line == 10
    assert context.lines[-1] == "line10"


def test_context_lines_zero_yields_the_finding_lines_only(tmp_path):
    """0 is a meaningful width, not an empty one."""
    context = extract_code_context(_write(tmp_path), line_start=4, line_end=5, context_lines=0)

    assert context.lines == ("line4", "line5")


def test_line_end_below_line_start_is_clamped_not_trusted(tmp_path):
    """Scanners disagree about line_end; a nonsense value must not invert the span."""
    context = extract_code_context(_write(tmp_path), line_start=6, line_end=2, context_lines=1)

    assert (context.focus_start, context.focus_end) == (6, 6)
    assert (context.start_line, context.end_line) == (5, 7)


def test_line_end_past_eof_is_clamped(tmp_path):
    context = extract_code_context(_write(tmp_path), line_start=9, line_end=999, context_lines=0)

    assert context.focus_end == 10


def test_default_width_matches_the_documented_default(tmp_path):
    """The prompt's advertised +/-20 comes from here; a silent change would widen every prompt."""
    assert DEFAULT_CONTEXT_LINES == 20
    context = extract_code_context(_write(tmp_path), line_start=1)
    assert context.end_line == 10  # whole 10-line file falls inside the default window


def test_result_is_frozen(tmp_path):
    context = extract_code_context(_write(tmp_path), line_start=1)
    assert isinstance(context, CodeContext)
    try:
        context.start_line = 99  # type: ignore[misc]
    except Exception:
        return
    raise AssertionError("CodeContext must be immutable")


# --- unavailability is silent ---------------------------------------------------------------------------------------


def test_missing_file_returns_none(tmp_path):
    assert extract_code_context(str(tmp_path / "nope.py"), line_start=1) is None


def test_directory_path_returns_none(tmp_path):
    assert extract_code_context(str(tmp_path), line_start=1) is None


def test_no_file_returns_none():
    """Dependency findings (Trivy/OSV) carry no file - not an error, just no context."""
    assert extract_code_context(None, line_start=1) is None


def test_no_line_start_returns_none(tmp_path):
    """Dependency findings are manifest-scoped and line-less by design."""
    assert extract_code_context(_write(tmp_path), line_start=None) is None


def test_line_past_eof_returns_none_rather_than_the_wrong_lines(tmp_path):
    """A stale line number means the file changed since the scan; wrong context is worse than none."""
    assert extract_code_context(_write(tmp_path), line_start=99) is None


def test_empty_file_returns_none(tmp_path):
    assert extract_code_context(_write(tmp_path, text=""), line_start=1) is None


def test_undecodable_file_returns_none(tmp_path):
    path = tmp_path / "blob.py"
    path.write_bytes(b"\xff\xfe\x00\x01binary")
    assert extract_code_context(str(path), line_start=1) is None


def test_oversized_file_returns_none(tmp_path):
    """A minified bundle costs more to read than the window is worth."""
    assert extract_code_context(_write(tmp_path, text="x" * 2_000_001), line_start=1) is None


def test_negative_context_lines_returns_none(tmp_path):
    assert extract_code_context(_write(tmp_path), line_start=5, context_lines=-1) is None
