"""
Tests for sentinelai.backend.loader - repository loading and validation.
"""
from pathlib import Path

import pytest

from sentinelai.backend.loader import (
    LoadedRepository,
    RepositoryNotADirectoryError,
    RepositoryNotFoundError,
    load_repository,
    validate_repository,
)


# --- validate_repository ---


def test_validate_repository_valid_path(tmp_path):
    result = validate_repository(str(tmp_path))
    assert result == tmp_path.resolve()
    assert result.is_absolute()


def test_validate_repository_relative_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "subdir").mkdir()
    result = validate_repository("subdir")
    assert result == (tmp_path / "subdir").resolve()


def test_validate_repository_absolute_path(tmp_path):
    result = validate_repository(str(tmp_path.resolve()))
    assert result.is_absolute()
    assert result == tmp_path.resolve()


def test_validate_repository_nonexistent_path(tmp_path):
    missing = tmp_path / "does-not-exist"
    with pytest.raises(RepositoryNotFoundError):
        validate_repository(str(missing))


def test_validate_repository_file_instead_of_directory(tmp_path):
    file_path = tmp_path / "a_file.txt"
    file_path.write_text("content")
    with pytest.raises(RepositoryNotADirectoryError):
        validate_repository(str(file_path))


def test_validate_repository_current_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = validate_repository(".")
    assert result == tmp_path.resolve()


def test_validate_repository_empty_directory(tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    result = validate_repository(str(empty_dir))
    assert result == empty_dir.resolve()


# --- symlinks ---


def _create_symlink(link, target, target_is_directory=False):
    try:
        link.symlink_to(target, target_is_directory=target_is_directory)
    except OSError as e:
        if getattr(e, "winerror", None) == 1314 or "privilege" in str(e).lower():
            pytest.skip("Symlink creation requires elevated privileges on Windows")
        raise


def test_validate_repository_symlink_to_directory(tmp_path):
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    link = tmp_path / "link"
    _create_symlink(link, real_dir, target_is_directory=True)
    result = validate_repository(str(link))
    assert result == real_dir.resolve()


def test_validate_repository_broken_symlink(tmp_path):
    broken_link = tmp_path / "broken"
    _create_symlink(broken_link, tmp_path / "does-not-exist-target")
    with pytest.raises(RepositoryNotFoundError):
        validate_repository(str(broken_link))


def test_validate_repository_symlink_loop(tmp_path):
    loop_a = tmp_path / "loop_a"
    loop_b = tmp_path / "loop_b"
    _create_symlink(loop_a, loop_b)
    _create_symlink(loop_b, loop_a)
    with pytest.raises(RepositoryNotFoundError):
        validate_repository(str(loop_a))


# --- unicode ---


def test_validate_repository_unicode_directory_name(tmp_path):
    unicode_dir = tmp_path / "répository-日本語-🔒"
    unicode_dir.mkdir()
    result = validate_repository(str(unicode_dir))
    assert result == unicode_dir.resolve()


# --- load_repository ---


def test_load_repository_returns_loaded_repository(tmp_path):
    result = load_repository(str(tmp_path))
    assert isinstance(result, LoadedRepository)
    assert result.absolute_path == tmp_path.resolve()
    assert result.name == tmp_path.resolve().name


def test_load_repository_relative_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path.parent)
    result = load_repository(tmp_path.name)
    assert result.original_path == Path(tmp_path.name)
    assert result.absolute_path == tmp_path.resolve()


def test_load_repository_absolute_path(tmp_path):
    result = load_repository(str(tmp_path.resolve()))
    assert result.absolute_path.is_absolute()
    assert result.absolute_path == tmp_path.resolve()


def test_load_repository_current_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = load_repository(".")
    assert result.absolute_path == tmp_path.resolve()
    assert result.name == tmp_path.resolve().name
    assert result.name != ""


def test_load_repository_nonexistent_path(tmp_path):
    missing = tmp_path / "does-not-exist"
    with pytest.raises(RepositoryNotFoundError):
        load_repository(str(missing))


def test_load_repository_file_instead_of_directory(tmp_path):
    file_path = tmp_path / "a_file.txt"
    file_path.write_text("content")
    with pytest.raises(RepositoryNotADirectoryError):
        load_repository(str(file_path))


def test_load_repository_empty_directory(tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    result = load_repository(str(empty_dir))
    assert result.name == "empty"
    assert result.absolute_path == empty_dir.resolve()


def test_load_repository_unicode_directory_name(tmp_path):
    unicode_dir = tmp_path / "répository-日本語-🔒"
    unicode_dir.mkdir()
    result = load_repository(str(unicode_dir))
    assert result.name == "répository-日本語-🔒"


# --- design rules: purity / exception isolation ---


def test_loaded_repository_is_immutable(tmp_path):
    result = load_repository(str(tmp_path))
    with pytest.raises(Exception):
        result.name = "changed"


def test_loaded_repository_carries_original_path_unresolved(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path.parent)
    result = load_repository(tmp_path.name)
    # original_path preserves what was given (relative, ".", etc.) - it is not the same
    # object as absolute_path, and callers get both from this one object.
    assert result.original_path == Path(tmp_path.name)
    assert result.original_path != result.absolute_path
    assert result.absolute_path == tmp_path.resolve()


def test_loaded_repository_original_path_matches_input_when_already_absolute(tmp_path):
    result = load_repository(str(tmp_path.resolve()))
    assert result.original_path == tmp_path.resolve()
    assert result.original_path == result.absolute_path


def test_backend_exceptions_do_not_reuse_core_errors():
    from sentinelai.core.errors import SentinelAIError

    assert not issubclass(RepositoryNotFoundError, SentinelAIError)
    assert not issubclass(RepositoryNotADirectoryError, SentinelAIError)
