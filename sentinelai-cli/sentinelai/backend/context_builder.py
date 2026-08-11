"""
Repository context - a single immutable snapshot bundling the outputs of
every other backend module for one LoadedRepository.

Pure orchestration: this module contains no filesystem walking, parsing,
scanning, or AI logic of its own. It calls each of the four existing
extractors exactly once and bundles their results. List outputs are
wrapped in tuples so the whole context is immutable end-to-end, not just
at the top level - a container-type change for the frozen contract, not
a transformation of the data itself.

RepositoryContext also carries the LoadedRepository it was built from,
per loader.py's own stated design: LoadedRepository is meant to be "the
one piece of state later backend modules... carry forward" - a context
with no way to know which repository it describes would be incomplete.

Exception behavior is inherited, not reimplemented: each extractor
already represents its own "expected absence" states through its return
value (GitMetadata.is_git_repository=False for a non-git directory, an
empty list for a repository with no dependencies, etc.), so there is
nothing for this module to catch. Any exception an extractor does raise
(e.g. tomllib.TOMLDecodeError from a malformed pyproject.toml, or
GitCommandNotFound from a missing git binary) is a genuine, unexpected
failure and propagates unchanged - this module adds no try/except.
"""
from dataclasses import dataclass
from typing import Tuple

from .dependencies import Dependency, extract_dependencies
from .git import GitMetadata, extract_git_metadata
from .languages import LanguageInfo, detect_languages
from .loader import LoadedRepository
from .snippets import CodeSnippet, extract_snippets


@dataclass(frozen=True)
class RepositoryContext:
    repository: LoadedRepository
    git_metadata: GitMetadata
    languages: Tuple[LanguageInfo, ...]
    dependencies: Tuple[Dependency, ...]
    snippets: Tuple[CodeSnippet, ...]


def build_repository_context(repository: LoadedRepository) -> RepositoryContext:
    """Call each backend extractor exactly once and bundle their results for `repository`."""
    git_metadata = extract_git_metadata(repository)
    languages = detect_languages(repository)
    dependencies = extract_dependencies(repository)
    snippets = extract_snippets(repository)

    return RepositoryContext(
        repository=repository,
        git_metadata=git_metadata,
        languages=tuple(languages),
        dependencies=tuple(dependencies),
        snippets=tuple(snippets),
    )
