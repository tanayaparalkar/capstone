"""Tests for sentinelai.backend.context_builder - repository context orchestration."""
from pathlib import Path
from unittest.mock import Mock

import git as gitpython
import pytest

from sentinelai.backend import context_builder as context_builder_module
from sentinelai.backend.context_builder import RepositoryContext, build_repository_context
from sentinelai.backend.dependencies import Dependency
from sentinelai.backend.git import GitMetadata
from sentinelai.backend.languages import LanguageInfo
from sentinelai.backend.loader import load_repository
from sentinelai.backend.snippets import CodeSnippet


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_empty_repository(tmp_path):
    context = build_repository_context(load_repository(str(tmp_path)))

    assert context.git_metadata == GitMetadata(is_git_repository=False, branch=None, commit_hash=None)
    assert context.languages == ()
    assert context.dependencies == ()
    assert context.snippets == ()


def test_non_git_repository_has_no_error(tmp_path):
    _write(tmp_path / "main.py", "print(1)\n")

    context = build_repository_context(load_repository(str(tmp_path)))

    assert context.git_metadata.is_git_repository is False
    assert context.git_metadata.branch is None
    assert context.git_metadata.commit_hash is None


def test_normal_repository(tmp_path):
    _write(tmp_path / "main.py", "print(1)\n")
    _write(tmp_path / "requirements.txt", "requests==2.31.0\n")

    context = build_repository_context(load_repository(str(tmp_path)))

    assert context.repository.absolute_path == tmp_path.resolve()
    assert context.languages == (LanguageInfo(name="Python", file_count=1, extensions=(".py",)),)
    assert context.dependencies == (
        Dependency(name="requests", version="==2.31.0", ecosystem="pypi", source_file="requirements.txt"),
    )
    assert context.snippets == (CodeSnippet(file_path="main.py", start_line=1, end_line=1, content="print(1)"),)


def test_repository_with_everything_simultaneously(tmp_path):
    repo = gitpython.Repo.init(tmp_path)
    with repo.config_writer() as cw:
        cw.set_value("user", "email", "test@example.com")
        cw.set_value("user", "name", "Test")
    _write(tmp_path / "main.py", "print(1)\n")
    _write(tmp_path / "app.js", "console.log(1);\n")
    _write(tmp_path / "requirements.txt", "flask>=2.0\n")
    repo.index.add(["main.py", "app.js", "requirements.txt"])
    commit = repo.index.commit("initial commit")

    context = build_repository_context(load_repository(str(tmp_path)))

    assert context.git_metadata == GitMetadata(
        is_git_repository=True, branch=repo.active_branch.name, commit_hash=commit.hexsha
    )
    assert {language.name for language in context.languages} == {"Python", "JavaScript"}
    assert context.dependencies == (
        Dependency(name="flask", version=">=2.0", ecosystem="pypi", source_file="requirements.txt"),
    )
    assert {snippet.file_path for snippet in context.snippets} == {"main.py", "app.js"}


def test_repository_context_is_immutable(tmp_path):
    context = build_repository_context(load_repository(str(tmp_path)))

    with pytest.raises(Exception):
        context.git_metadata = GitMetadata(is_git_repository=True, branch="x", commit_hash="y")

    assert isinstance(context.languages, tuple)
    assert isinstance(context.dependencies, tuple)
    assert isinstance(context.snippets, tuple)


def test_deterministic_output(tmp_path):
    _write(tmp_path / "main.py", "print(1)\n")
    _write(tmp_path / "requirements.txt", "requests==2.31.0\n")

    repository = load_repository(str(tmp_path))
    first = build_repository_context(repository)
    second = build_repository_context(repository)

    assert first == second


def test_each_extractor_is_called_exactly_once(tmp_path, monkeypatch):
    repository = load_repository(str(tmp_path))

    mock_git_metadata = GitMetadata(is_git_repository=False, branch=None, commit_hash=None)
    mock_languages = [LanguageInfo(name="Python", file_count=1, extensions=(".py",))]
    mock_dependencies = [Dependency(name="requests", version=None, ecosystem="pypi", source_file="requirements.txt")]
    mock_snippets = [CodeSnippet(file_path="main.py", start_line=1, end_line=1, content="print(1)")]

    mock_extract_git_metadata = Mock(return_value=mock_git_metadata)
    mock_detect_languages = Mock(return_value=mock_languages)
    mock_extract_dependencies = Mock(return_value=mock_dependencies)
    mock_extract_snippets = Mock(return_value=mock_snippets)

    monkeypatch.setattr(context_builder_module, "extract_git_metadata", mock_extract_git_metadata)
    monkeypatch.setattr(context_builder_module, "detect_languages", mock_detect_languages)
    monkeypatch.setattr(context_builder_module, "extract_dependencies", mock_extract_dependencies)
    monkeypatch.setattr(context_builder_module, "extract_snippets", mock_extract_snippets)

    context = build_repository_context(repository)

    mock_extract_git_metadata.assert_called_once_with(repository)
    mock_detect_languages.assert_called_once_with(repository)
    mock_extract_dependencies.assert_called_once_with(repository)
    mock_extract_snippets.assert_called_once_with(repository)

    assert context.git_metadata == mock_git_metadata
    assert context.languages == tuple(mock_languages)
    assert context.dependencies == tuple(mock_dependencies)
    assert context.snippets == tuple(mock_snippets)


def test_unexpected_exception_from_an_extractor_propagates_unchanged(tmp_path, monkeypatch):
    repository = load_repository(str(tmp_path))

    class SentinelError(Exception):
        pass

    def _raise(_repository):
        raise SentinelError("boom")

    monkeypatch.setattr(context_builder_module, "detect_languages", _raise)

    with pytest.raises(SentinelError, match="boom"):
        build_repository_context(repository)
