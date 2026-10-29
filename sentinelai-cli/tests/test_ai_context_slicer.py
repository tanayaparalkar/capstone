"""slice_around(): a bounded window of source around one line, with both a line and a character ceiling."""
import ast
from pathlib import Path

import pytest

from sentinelai.ai.context import CodeSlice, CodeSliceError, slice_around


def _write(tmp_path, content, name="src.py"):
    path = tmp_path / name
    path.write_bytes(content if isinstance(content, bytes) else content.encode("utf-8"))
    return path


def _numbered(count):
    return "".join(f"line {n}\n" for n in range(1, count + 1))


# ---- window placement ----


def test_window_is_centred_on_the_line(tmp_path):
    result = slice_around(_write(tmp_path, _numbered(30)), 15)
    assert (result.line_start, result.line_end) == (10, 20)
    assert result.text.splitlines() == [f"line {n}" for n in range(10, 21)]
    assert result.truncated is False


def test_finding_at_line_1_clamps_the_start(tmp_path):
    result = slice_around(_write(tmp_path, _numbered(30)), 1)
    assert (result.line_start, result.line_end) == (1, 6)
    assert result.text.splitlines()[0] == "line 1"
    assert result.truncated is False


def test_finding_at_eof_clamps_the_end(tmp_path):
    result = slice_around(_write(tmp_path, _numbered(30)), 30)
    assert (result.line_start, result.line_end) == (25, 30)
    assert result.text.splitlines()[-1] == "line 30"
    assert result.truncated is False


def test_last_line_without_trailing_newline(tmp_path):
    result = slice_around(_write(tmp_path, "a\nb\nlast"), 3)
    assert (result.line_start, result.line_end, result.text) == (1, 3, "a\nb\nlast")


def test_file_shorter_than_the_window(tmp_path):
    result = slice_around(_write(tmp_path, _numbered(3)), 2, before=10, after=10)
    assert (result.line_start, result.line_end) == (1, 3)
    assert result.text == "line 1\nline 2\nline 3"
    assert result.truncated is False


def test_single_line_file(tmp_path):
    result = slice_around(_write(tmp_path, "only\n"), 1)
    assert (result.line_start, result.line_end, result.text, result.truncated) == (1, 1, "only", False)


def test_zero_context_returns_just_the_line(tmp_path):
    result = slice_around(_write(tmp_path, _numbered(10)), 4, before=0, after=0)
    assert (result.line_start, result.line_end, result.text) == (4, 4, "line 4")


def test_asymmetric_line_window(tmp_path):
    result = slice_around(_write(tmp_path, _numbered(30)), 15, before=2, after=7)
    assert (result.line_start, result.line_end) == (13, 22)


def test_line_numbers_follow_newlines_only(tmp_path):
    # str.splitlines() would also split on form feed and U+2028, shifting line 3 away from the scanner's line 3.
    result = slice_around(_write(tmp_path, "a\fb\nc d\ntarget\n"), 3, before=0, after=0)
    assert result.text == "target"


def test_crlf_terminators_are_not_kept(tmp_path):
    result = slice_around(_write(tmp_path, b"a\r\nb\r\nc\r\n"), 2)
    assert result.text == "a\nb\nc"


def test_blank_lines_count(tmp_path):
    result = slice_around(_write(tmp_path, "\n\ntarget\n\n"), 3, before=1, after=1)
    assert (result.line_start, result.line_end, result.text) == (2, 4, "\ntarget\n")


def test_path_is_reported_as_given(tmp_path):
    path = _write(tmp_path, _numbered(3))
    assert slice_around(path, 1).path == str(path)
    assert slice_around(str(path), 1).path == str(path)


# ---- character ceiling ----


def test_single_50kb_line_is_cut_to_max_chars(tmp_path):
    line = "var a=" + "x" * 50_000 + ";"
    result = slice_around(_write(tmp_path, line + "\n", "bundle.min.js"), 1, max_chars=2000)
    assert result.truncated is True
    assert result.text == line[:2000]
    assert (result.line_start, result.line_end) == (1, 1)


def test_50kb_target_line_drops_its_neighbours(tmp_path):
    content = "header\n" + "y" * 50_000 + "\nfooter\n"
    result = slice_around(_write(tmp_path, content), 2, max_chars=2000)
    assert result.truncated is True
    assert (result.line_start, result.line_end) == (2, 2)
    assert len(result.text) == 2000


def test_50kb_neighbour_is_dropped_but_the_target_kept_whole(tmp_path):
    content = "z" * 50_000 + "\ntarget = 1\nafter = 2\n"
    result = slice_around(_write(tmp_path, content), 2, max_chars=2000)
    assert result.truncated is True
    assert result.text == "target = 1\nafter = 2"
    assert (result.line_start, result.line_end) == (2, 3)


def test_char_ceiling_trims_symmetrically_around_the_target(tmp_path):
    lines = [f"{n:02d}" + "." * 97 for n in range(1, 22)]  # 21 lines of 99 chars
    result = slice_around(_write(tmp_path, "\n".join(lines) + "\n"), 11, before=10, after=10, max_chars=500)
    # 500 chars fits 5 lines (5*99 + 4 newlines = 499): the target plus two either side.
    assert (result.line_start, result.line_end) == (9, 13)
    assert result.truncated is True
    assert len(result.text) <= 500


def test_char_ceiling_gives_spare_room_to_the_side_that_has_lines(tmp_path):
    lines = [f"{n:02d}" + "." * 97 for n in range(1, 22)]
    result = slice_around(_write(tmp_path, "\n".join(lines) + "\n"), 2, before=10, after=10, max_chars=500)
    assert (result.line_start, result.line_end) == (1, 5)
    assert result.truncated is True


def test_text_never_exceeds_max_chars(tmp_path):
    content = "".join(f"{'w' * (n * 37 % 300)}\n" for n in range(1, 200))
    path = _write(tmp_path, content)
    for line in (1, 50, 199):
        for max_chars in (1, 10, 299, 300, 1000, 5000):
            result = slice_around(path, line, before=20, after=20, max_chars=max_chars)
            assert len(result.text) <= max_chars
            assert result.line_start <= line <= result.line_end


def test_window_that_fits_exactly_is_not_truncated(tmp_path):
    path = _write(tmp_path, "aaaa\nbbbb\ncccc\n")
    result = slice_around(path, 2, max_chars=14)  # 3 * 4 + 2 newlines
    assert (result.text, result.truncated) == ("aaaa\nbbbb\ncccc", False)
    assert slice_around(path, 2, max_chars=13).truncated is True


def test_multibyte_characters_are_counted_as_characters(tmp_path):
    result = slice_around(_write(tmp_path, "é" * 3000 + "\n"), 1, max_chars=2000)
    assert result.text == "é" * 2000
    assert result.truncated is True


def test_huge_line_is_not_read_whole_into_memory(tmp_path, monkeypatch):
    path = _write(tmp_path, "q" * 2_000_000 + "\nnext\n")
    reads = []
    real_open = Path.open

    def recording_open(self, *args, **kwargs):
        handle = real_open(self, *args, **kwargs)
        real_readline = handle.readline

        class Wrapped:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                handle.close()

            def readline(self, limit=-1):
                data = real_readline(limit)
                reads.append(len(data))
                return data

        return Wrapped()

    monkeypatch.setattr(Path, "open", recording_open)
    result = slice_around(path, 2, max_chars=100)
    assert result.text == "next"
    assert max(reads) <= 100 * 4 + 4


# ---- failure is loud ----


def test_missing_file_raises_a_clear_error(tmp_path):
    missing = tmp_path / "gone.py"
    with pytest.raises(CodeSliceError, match="gone.py") as excinfo:
        slice_around(missing, 1)
    assert isinstance(excinfo.value.__cause__, FileNotFoundError)


def test_directory_raises_a_clear_error(tmp_path):
    with pytest.raises(CodeSliceError, match="cannot read source"):
        slice_around(tmp_path, 1)


def test_unreadable_file_raises_a_clear_error(tmp_path, monkeypatch):
    path = _write(tmp_path, _numbered(3))

    def denied(self, *args, **kwargs):
        raise PermissionError(13, "Permission denied", str(self))

    monkeypatch.setattr(Path, "open", denied)
    with pytest.raises(CodeSliceError, match="Permission denied"):
        slice_around(path, 1)


def test_line_past_eof_raises_instead_of_returning_an_empty_slice(tmp_path):
    with pytest.raises(CodeSliceError, match=r"line 31 is past the end of the file \(30 lines\)"):
        slice_around(_write(tmp_path, _numbered(30)), 31)


def test_line_far_past_eof_reports_the_real_length(tmp_path):
    with pytest.raises(CodeSliceError, match=r"\(3 lines\)"):
        slice_around(_write(tmp_path, _numbered(3)), 100, before=5)


def test_empty_file_raises(tmp_path):
    with pytest.raises(CodeSliceError, match=r"\(0 lines\)"):
        slice_around(_write(tmp_path, b""), 1)


@pytest.mark.parametrize(
    "kwargs",
    [dict(line=0), dict(line=-3), dict(line=1, before=-1), dict(line=1, after=-1), dict(line=1, max_chars=0)],
)
def test_invalid_arguments_raise_value_error(tmp_path, kwargs):
    with pytest.raises(ValueError):
        slice_around(_write(tmp_path, _numbered(3)), **kwargs)


# ---- encoding ----


def test_non_utf8_bytes_are_replaced_and_numbering_is_kept(tmp_path):
    content = b"ok = 1\nname = 'caf\xe9'\nblob = b'\xff\xfe\x80'\nend = 2\n"
    result = slice_around(_write(tmp_path, content), 3, before=1, after=1)
    assert (result.line_start, result.line_end) == (2, 4)
    assert result.text == "name = 'caf�'\nblob = b'���'\nend = 2"
    assert result.truncated is False


def test_non_utf8_bytes_count_toward_max_chars(tmp_path):
    result = slice_around(_write(tmp_path, b"\xff" * 5000 + b"\n"), 1, max_chars=100)
    assert result.text == "�" * 100
    assert result.truncated is True


def test_utf8_bom_is_left_for_the_caller(tmp_path):
    result = slice_around(_write(tmp_path, b"\xef\xbb\xbfimport os\n"), 1)
    assert result.text == "﻿import os"


# ---- contract and boundaries ----


def test_code_slice_rejects_an_inverted_range():
    with pytest.raises(ValueError):
        CodeSlice(text="", path="x", line_start=5, line_end=4, truncated=False)


def test_slicer_imports_nothing_from_other_ai_subpackages():
    source = Path(__file__).parent.parent / "sentinelai" / "ai" / "context" / "slicer.py"
    for node in ast.walk(ast.parse(source.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom):
            assert node.level == 0, "slicer.py must not use relative imports"
            assert not (node.module or "").startswith("sentinelai"), node.module
        elif isinstance(node, ast.Import):
            assert not any(alias.name.startswith("sentinelai") for alias in node.names)
