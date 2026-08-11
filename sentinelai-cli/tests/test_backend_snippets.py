"""Tests for sentinelai.backend.snippets - fixed-size code snippet extraction."""
from pathlib import Path

import pytest

from sentinelai.backend.loader import load_repository
from sentinelai.backend.snippets import CodeSnippet, extract_snippets


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _lines(n: int, prefix: str = "line") -> str:
    return "\n".join(f"{prefix}_{i}" for i in range(1, n + 1)) + "\n"


def test_empty_repository(tmp_path):
    result = extract_snippets(load_repository(str(tmp_path)))

    assert result == []


def test_normal_case_single_short_file(tmp_path):
    _write(tmp_path / "main.py", "print('hello')\nprint('world')\n")

    result = extract_snippets(load_repository(str(tmp_path)))

    assert result == [
        CodeSnippet(file_path="main.py", start_line=1, end_line=2, content="print('hello')\nprint('world')")
    ]


def test_empty_file_produces_no_snippets(tmp_path):
    _write(tmp_path / "empty.py", "")

    result = extract_snippets(load_repository(str(tmp_path)))

    assert result == []


def test_unsupported_extension_is_skipped(tmp_path):
    _write(tmp_path / "notes.txt", "just some notes\n")
    _write(tmp_path / "main.py", "print(1)\n")

    result = extract_snippets(load_repository(str(tmp_path)))

    assert [s.file_path for s in result] == ["main.py"]


def test_ignored_directories_are_not_walked(tmp_path):
    _write(tmp_path / "main.py", "print(1)\n")
    _write(tmp_path / ".git" / "objects" / "fake.py", "print('should not appear')\n")
    _write(tmp_path / "node_modules" / "pkg" / "index.js", "console.log('nope')\n")
    _write(tmp_path / "__pycache__" / "cached.py", "print('nope')\n")
    _write(tmp_path / ".venv" / "lib" / "site.py", "print('nope')\n")
    _write(tmp_path / "venv" / "lib" / "site.py", "print('nope')\n")
    _write(tmp_path / "dist" / "bundle.js", "console.log('nope')\n")
    _write(tmp_path / "build" / "out.py", "print('nope')\n")
    _write(tmp_path / ".pytest_cache" / "cached.py", "print('nope')\n")
    _write(tmp_path / ".mypy_cache" / "cached.py", "print('nope')\n")
    _write(tmp_path / ".ruff_cache" / "cached.py", "print('nope')\n")
    _write(tmp_path / ".cache" / "cached.py", "print('nope')\n")

    result = extract_snippets(load_repository(str(tmp_path)))

    assert [s.file_path for s in result] == ["main.py"]


def test_unicode_filenames_are_supported(tmp_path):
    _write(tmp_path / "café☕.py", "print('unicode filename')\n")

    result = extract_snippets(load_repository(str(tmp_path)))

    assert [s.file_path for s in result] == ["café☕.py"]
    assert result[0].content == "print('unicode filename')"


def test_unicode_file_content_is_preserved(tmp_path):
    _write(tmp_path / "main.py", "# café ☕ — comment\nprint('héllo wörld')\n")

    result = extract_snippets(load_repository(str(tmp_path)))

    assert result[0].content == "# café ☕ — comment\nprint('héllo wörld')"


def test_binary_files_are_skipped_safely(tmp_path):
    (tmp_path / "corrupted.py").write_bytes(bytes([0, 1, 2, 255, 254, 128, 200]) * 10)
    _write(tmp_path / "main.py", "print(1)\n")

    result = extract_snippets(load_repository(str(tmp_path)))

    assert [s.file_path for s in result] == ["main.py"]


def test_large_file_is_split_into_multiple_chunks(tmp_path):
    _write(tmp_path / "big.py", _lines(45))

    result = extract_snippets(load_repository(str(tmp_path)))

    assert len(result) == 3
    assert (result[0].start_line, result[0].end_line) == (1, 20)
    assert (result[1].start_line, result[1].end_line) == (21, 40)
    assert (result[2].start_line, result[2].end_line) == (41, 45)
    assert result[0].content.splitlines()[0] == "line_1"
    assert result[2].content.splitlines()[-1] == "line_45"


def test_file_exactly_at_chunk_boundary_produces_one_chunk(tmp_path):
    _write(tmp_path / "exact.py", _lines(20))

    result = extract_snippets(load_repository(str(tmp_path)))

    assert len(result) == 1
    assert (result[0].start_line, result[0].end_line) == (1, 20)


def test_deterministic_ordering_regardless_of_creation_order(tmp_path):
    _write(tmp_path / "zeta.py", "print('z')\n")
    _write(tmp_path / "alpha.py", "print('a')\n")
    _write(tmp_path / "mu.py", "print('m')\n")

    result = extract_snippets(load_repository(str(tmp_path)))

    assert [s.file_path for s in result] == ["alpha.py", "mu.py", "zeta.py"]


def test_snippets_within_a_file_are_ordered_by_start_line(tmp_path):
    _write(tmp_path / "big.py", _lines(50))

    result = extract_snippets(load_repository(str(tmp_path)))

    assert [s.start_line for s in result] == [1, 21, 41]


def test_snippet_is_immutable(tmp_path):
    _write(tmp_path / "main.py", "print(1)\n")
    result = extract_snippets(load_repository(str(tmp_path)))

    with pytest.raises(Exception):
        result[0].content = "changed"
