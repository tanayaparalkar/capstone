"""
Repository safety inspection.

Answers one question before a patch is written: does this repository already
contain changes the user has not committed? If it does, applying a patch mixes
SentinelAI's edit into work in progress, and telling the two apart afterwards is
the user's problem rather than the tool's.

This module OBSERVES and never ACTS. It does not print, log, prompt, abort, or
raise on a dirty tree. It returns a frozen value object describing what it found,
and whoever called it decides what that means. A dirty repository is a warning
for a future CLI to surface, not an error - refusing to patch because a file is
uncommitted would be a policy decision this layer has no standing to make.

Not being a git repository, and git not being installed, are likewise not errors.
Plenty of scanned directories are neither, and a backup is taken regardless, so
the absence of git costs the user nothing.

Uses GitPython rather than hand-rolled subprocess handling, matching the decision
backend/git.py already documents for this same problem: its typed exceptions
distinguish "not a repository" from "git is missing" without parsing stderr, and
`repo.git.status("--porcelain")` runs exactly that command. This is the only
module in patching/ that shells out at all, which is why it is separate from
backup.py - keeping the dependency in one small, replaceable place.
"""
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Tuple

from git import InvalidGitRepositoryError, NoSuchPathError, Repo
from git.exc import GitCommandError, GitCommandNotFound


class RepositoryState(str, Enum):
    """What was found when the repository was inspected.

    Four outcomes, deliberately distinct. Only DIRTY is a reason to warn; the
    other three are all ordinary conditions under which patching proceeds
    normally, and collapsing them would lose the reason a check was skipped.
    """

    CLEAN = "clean"
    DIRTY = "dirty"
    NOT_A_REPOSITORY = "not_a_repository"
    GIT_UNAVAILABLE = "git_unavailable"
    NOT_CHECKED = "not_checked"


@dataclass(frozen=True)
class RepositoryStatus:
    """Result of inspecting a repository before writing.

    `changed_paths` holds the porcelain entries verbatim, in git's own order, so
    a caller can show the user exactly what git reported rather than a
    paraphrase. Empty for every state except DIRTY.

    `should_warn` is the single question a caller needs answered. It is derived
    rather than stored, so it cannot contradict `state`.
    """

    state: RepositoryState
    changed_paths: Tuple[str, ...] = ()
    detail: str = ""

    @property
    def should_warn(self) -> bool:
        return self.state is RepositoryState.DIRTY

    @property
    def is_clean(self) -> bool:
        return self.state is RepositoryState.CLEAN


def inspect_repository(root: Path) -> RepositoryStatus:
    """Report whether `root` is a git repository with uncommitted changes.

    Never raises for an absent, broken, or non-git directory: every such case is
    a state, not an exception. The only thing that could propagate is an
    unexpected git failure, which is reported as NOT_A_REPOSITORY with the
    detail attached rather than crashing a patch run over a status query.
    """
    try:
        repo = Repo(Path(root), search_parent_directories=False)
    except (InvalidGitRepositoryError, NoSuchPathError) as exc:
        return RepositoryStatus(state=RepositoryState.NOT_A_REPOSITORY, detail=str(exc))
    except GitCommandNotFound as exc:
        return RepositoryStatus(state=RepositoryState.GIT_UNAVAILABLE, detail=str(exc))

    try:
        porcelain = repo.git.status("--porcelain")
    except GitCommandNotFound as exc:
        return RepositoryStatus(state=RepositoryState.GIT_UNAVAILABLE, detail=str(exc))
    except GitCommandError as exc:
        return RepositoryStatus(state=RepositoryState.NOT_A_REPOSITORY, detail=str(exc))

    entries = tuple(line for line in porcelain.splitlines() if line.strip())
    if not entries:
        return RepositoryStatus(state=RepositoryState.CLEAN)

    return RepositoryStatus(
        state=RepositoryState.DIRTY,
        changed_paths=entries,
        detail=(
            f"{len(entries)} uncommitted change(s) present; a patch applied now will be "
            "mixed in with work already in progress"
        ),
    )
