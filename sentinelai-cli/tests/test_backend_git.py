"""
Tests for sentinelai.backend.git - Git metadata extraction.

Every fixture repository is created fresh inside a pytest tmp_path via
GitPython itself, not this project's own repository.
"""
from pathlib import Path

import git
import pytest

from sentinelai.backend.git import GitMetadata, extract_git_metadata
from sentinelai.backend.loader import load_repository

_COMMITTER_CONFIG = {"user.email": "test@example.com", "user.name": "Test"}


def _init_repo(path: Path) -> git.Repo:
    repo = git.Repo.init(path)
    with repo.config_writer() as cw:
        for key, value in _COMMITTER_CONFIG.items():
            section, option = key.split(".")
            cw.set_value(section, option, value)
    return repo


def _commit(repo: git.Repo, filename: str = "file.txt", message: str = "commit") -> str:
    (Path(repo.working_tree_dir) / filename).write_text("content")
    repo.index.add([filename])
    return repo.index.commit(message).hexsha


def test_normal_git_repository(tmp_path):
    repo = _init_repo(tmp_path)
    sha = _commit(repo)

    result = extract_git_metadata(load_repository(str(tmp_path)))

    assert result == GitMetadata(is_git_repository=True, branch=repo.active_branch.name, commit_hash=sha)


def test_detached_head(tmp_path):
    repo = _init_repo(tmp_path)
    sha = _commit(repo)
    repo.git.checkout(sha)

    result = extract_git_metadata(load_repository(str(tmp_path)))

    assert result.is_git_repository is True
    assert result.branch is None
    assert result.commit_hash == sha


def test_repository_with_no_commits(tmp_path):
    git.Repo.init(tmp_path)

    result = extract_git_metadata(load_repository(str(tmp_path)))

    assert result.is_git_repository is True
    assert result.branch is not None  # HEAD still symbolically points to a branch name
    assert result.commit_hash is None


def test_non_git_directory(tmp_path):
    result = extract_git_metadata(load_repository(str(tmp_path)))

    assert result == GitMetadata(is_git_repository=False, branch=None, commit_hash=None)


def test_git_metadata_is_immutable(tmp_path):
    git.Repo.init(tmp_path)
    result = extract_git_metadata(load_repository(str(tmp_path)))
    with pytest.raises(Exception):
        result.branch = "changed"


def test_branch_name_containing_slashes(tmp_path):
    repo = _init_repo(tmp_path)
    sha = _commit(repo)
    feature_branch = repo.create_head("feature/security-scan")
    feature_branch.checkout()
    # Precondition, checked independently of the function under test: proves the
    # fixture is actually checked out onto this branch, so a failure below is
    # attributable to extract_git_metadata, not to a broken test setup.
    assert repo.active_branch.name == "feature/security-scan"

    result = extract_git_metadata(load_repository(str(tmp_path)))

    assert result.branch == "feature/security-scan"
    assert result.commit_hash == sha


def test_nested_subdirectory_is_not_treated_as_the_parent_repository(tmp_path):
    # search_parent_directories=False must prevent walking up to an ancestor .git -
    # verifies that deliberate choice rather than assuming it.
    repo = _init_repo(tmp_path)
    _commit(repo)
    nested = tmp_path / "src" / "nested"
    nested.mkdir(parents=True)

    result = extract_git_metadata(load_repository(str(nested)))

    assert result == GitMetadata(is_git_repository=False, branch=None, commit_hash=None)


def test_bare_repository(tmp_path):
    repo = _init_repo(tmp_path / "source")
    sha = _commit(repo)
    bare_path = tmp_path / "bare.git"
    repo.clone(str(bare_path), bare=True)

    result = extract_git_metadata(load_repository(str(bare_path)))

    assert result.is_git_repository is True
    assert result.branch == repo.active_branch.name
    assert result.commit_hash == sha


def test_malformed_repository_missing_head_is_not_a_crash(tmp_path):
    git.Repo.init(tmp_path)
    (tmp_path / ".git" / "HEAD").unlink()

    result = extract_git_metadata(load_repository(str(tmp_path)))

    assert result == GitMetadata(is_git_repository=False, branch=None, commit_hash=None)


def test_missing_git_binary_propagates_instead_of_being_swallowed(tmp_path, monkeypatch):
    repo = _init_repo(tmp_path)
    _commit(repo)
    monkeypatch.setenv("PATH", "/nonexistent")

    with pytest.raises(git.exc.GitCommandNotFound):
        extract_git_metadata(load_repository(str(tmp_path)))
