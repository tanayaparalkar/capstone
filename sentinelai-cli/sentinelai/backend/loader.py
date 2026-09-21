"""
Repository loading and validation - the foundation of the backend.

Confirms a filesystem path is a usable directory and records the basic
facts about it. Knows nothing about AI, RAG, scanners, the CLI,
reporting, statistics, or even git-ness/languages/dependencies - those
are later backend modules that take a LoadedRepository as their input.
"""
import logging
from dataclasses import dataclass
from pathlib import Path


class RepositoryError(Exception):
    """Base class for backend repository loading/validation errors. Not a subclass of core.errors.SentinelAIError."""


class RepositoryNotFoundError(RepositoryError):
    """The given path does not exist, or could not be resolved to a real location (e.g. a symlink loop)."""


class RepositoryNotADirectoryError(RepositoryError):
    """The given path exists but is not a directory."""


logger = logging.getLogger("sentinelai")


@dataclass(frozen=True)
class LoadedRepository:
    # original_path is kept, not derivable from absolute_path: this object is meant to be the
    # one piece of state later backend modules (git.py, context_builder.py, live_provider.py, ...)
    # carry forward, so it needs to be complete rather than requiring callers to separately track
    # what was originally given alongside it.
    original_path: Path
    absolute_path: Path
    name: str


def validate_repository(path: str) -> Path:
    """Resolve `path` to an absolute Path, raising if it doesn't exist or isn't a directory."""
    try:
        absolute_path = Path(path).resolve()
    except RuntimeError as exc:
        # pathlib raises RuntimeError, not OSError, for a detected symlink loop.
        raise RepositoryNotFoundError(f"Repository path could not be resolved: {path}") from exc

    if not absolute_path.exists():
        raise RepositoryNotFoundError(f"Repository path does not exist: {absolute_path}")
    if not absolute_path.is_dir():
        raise RepositoryNotADirectoryError(f"Repository path is not a directory: {absolute_path}")
    return absolute_path


def load_repository(path: str) -> LoadedRepository:
    """Validate `path` and return a LoadedRepository describing it. Raises RepositoryError on invalid input."""
    absolute_path = validate_repository(path)
    logger.debug("repository loaded: name=%s path=%s", absolute_path.name, absolute_path)
    return LoadedRepository(original_path=Path(path), absolute_path=absolute_path, name=absolute_path.name)
