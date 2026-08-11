"""Tests for sentinelai.backend.languages - language detection by file extension."""
from pathlib import Path

import pytest

from sentinelai.backend.languages import LanguageInfo, detect_languages
from sentinelai.backend.loader import load_repository


def _touch(path: Path, content: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def test_empty_repository(tmp_path):
    result = detect_languages(load_repository(str(tmp_path)))

    assert result == []


def test_single_python_repository(tmp_path):
    _touch(tmp_path / "main.py")
    _touch(tmp_path / "utils.py")

    result = detect_languages(load_repository(str(tmp_path)))

    assert result == [LanguageInfo(name="Python", file_count=2, extensions=(".py",))]


def test_mixed_language_repository(tmp_path):
    _touch(tmp_path / "main.py")
    _touch(tmp_path / "app.js")
    _touch(tmp_path / "server.go")

    result = detect_languages(load_repository(str(tmp_path)))

    assert {info.name for info in result} == {"Python", "JavaScript", "Go"}
    assert all(info.file_count == 1 for info in result)


def test_ignored_directories_are_not_walked(tmp_path):
    _touch(tmp_path / "main.py")
    _touch(tmp_path / ".git" / "objects" / "fake.py")
    _touch(tmp_path / "node_modules" / "pkg" / "index.js")
    _touch(tmp_path / "__pycache__" / "cached.py")
    _touch(tmp_path / ".venv" / "lib" / "site.py")
    _touch(tmp_path / "venv" / "lib" / "site.py")
    _touch(tmp_path / "dist" / "bundle.js")
    _touch(tmp_path / "build" / "out.py")
    _touch(tmp_path / ".cache" / "cached.py")
    _touch(tmp_path / ".mypy_cache" / "cached.py")
    _touch(tmp_path / ".pytest_cache" / "cached.py")
    _touch(tmp_path / ".ruff_cache" / "cached.py")

    result = detect_languages(load_repository(str(tmp_path)))

    assert result == [LanguageInfo(name="Python", file_count=1, extensions=(".py",))]


def test_hidden_non_cache_directory_is_not_ignored(tmp_path):
    # Hidden directories are only skipped when they're known cache/build
    # artifacts, not just because they start with ".": repositories can
    # legitimately keep source under a dot-prefixed directory (e.g. ".github"
    # scripts, ".config" tooling), and those files must still be counted.
    _touch(tmp_path / ".config" / "setup.py")

    result = detect_languages(load_repository(str(tmp_path)))

    assert result == [LanguageInfo(name="Python", file_count=1, extensions=(".py",))]


def test_uppercase_extension_is_matched_case_insensitively(tmp_path):
    _touch(tmp_path / "Main.PY")

    result = detect_languages(load_repository(str(tmp_path)))

    assert result == [LanguageInfo(name="Python", file_count=1, extensions=(".py",))]


def test_files_without_extension_are_ignored(tmp_path):
    _touch(tmp_path / "README")
    _touch(tmp_path / "Makefile")
    _touch(tmp_path / "main.py")

    result = detect_languages(load_repository(str(tmp_path)))

    assert result == [LanguageInfo(name="Python", file_count=1, extensions=(".py",))]


def test_unknown_extensions_are_ignored(tmp_path):
    _touch(tmp_path / "notes.txt")
    _touch(tmp_path / "data.json")
    _touch(tmp_path / "main.py")

    result = detect_languages(load_repository(str(tmp_path)))

    assert result == [LanguageInfo(name="Python", file_count=1, extensions=(".py",))]


def test_multiple_extensions_for_c_cpp_are_aggregated(tmp_path):
    _touch(tmp_path / "main.c")
    _touch(tmp_path / "util.h")
    _touch(tmp_path / "app.cpp")

    result = detect_languages(load_repository(str(tmp_path)))

    assert result == [LanguageInfo(name="C/C++", file_count=3, extensions=(".c", ".cpp", ".h"))]


def test_deterministic_ordering_by_count_then_name(tmp_path):
    _touch(tmp_path / "a.py")
    _touch(tmp_path / "b.py")
    _touch(tmp_path / "c.js")
    _touch(tmp_path / "d.go")
    _touch(tmp_path / "e.go")

    result = detect_languages(load_repository(str(tmp_path)))

    assert [info.name for info in result] == ["Go", "Python", "JavaScript"]


def test_language_info_is_immutable(tmp_path):
    _touch(tmp_path / "main.py")
    result = detect_languages(load_repository(str(tmp_path)))

    with pytest.raises(Exception):
        result[0].file_count = 99
