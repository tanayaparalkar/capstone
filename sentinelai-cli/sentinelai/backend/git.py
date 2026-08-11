"""
Git metadata extraction - current branch and commit hash for a
LoadedRepository.

Uses GitPython (a real project dependency, added for this file) rather
than hand-parsing .git/HEAD, refs/, packed-refs, worktrees, and
submodules: that surface is git's own internal format, which is
intricate and evolves - a hand-rolled parser would gradually become a
partial, maintenance-prone reimplementation of git internals as edge
cases accumulate. GitPython is a mature, widely-used library that
already handles this correctly.

The loader already validated the filesystem; this module validates
git state only, and treats "not a git repository" and "no commits yet"
as normal, expected states, not errors.

Exceptions are triaged by what they represent, not just what they are:
InvalidGitRepositoryError and NoSuchPathError describe repository
state (this path isn't a usable git repo) and are caught, converted
into is_git_repository=False. GitCommandNotFound describes execution
environment configuration (the `git` binary itself is missing) and is
deliberately left to propagate uncaught - the repository could be
perfectly valid; the environment inspecting it is not, and reporting
is_git_repository=False in that case would misrepresent which of the
two is actually broken.
"""
from dataclasses import dataclass
from typing import Optional

from git import InvalidGitRepositoryError, NoSuchPathError, Repo

from .loader import LoadedRepository


@dataclass(frozen=True)
class GitMetadata:
    is_git_repository: bool
    branch: Optional[str]
    commit_hash: Optional[str]


def extract_git_metadata(repository: LoadedRepository) -> GitMetadata:
    try:
        repo = Repo(repository.absolute_path, search_parent_directories=False)
    except (InvalidGitRepositoryError, NoSuchPathError):
        return GitMetadata(is_git_repository=False, branch=None, commit_hash=None)

    # Detached HEAD has no branch name; a fresh repo with no commits still has one
    # (HEAD points to it symbolically) even though there's no commit to name yet.
    branch = None if repo.head.is_detached else repo.active_branch.name

    try:
        commit_hash = repo.head.commit.hexsha
    except ValueError:
        # HEAD's branch ref exists but no commit has been made yet.
        commit_hash = None
    # repo.head.commit shells out to `git` internally; a missing binary raises
    # GitCommandNotFound here, deliberately left uncaught - see module docstring.

    return GitMetadata(is_git_repository=True, branch=branch, commit_hash=commit_hash)
