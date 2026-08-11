"""Backend: repository loading and analysis, independent of AI, scanners, and the CLI."""
from .context_builder import RepositoryContext, build_repository_context
from .dependencies import Dependency, extract_dependencies
from .git import GitMetadata, extract_git_metadata
from .languages import LanguageInfo, detect_languages
from .loader import LoadedRepository, RepositoryError, RepositoryNotADirectoryError, RepositoryNotFoundError, load_repository, validate_repository
from .snippets import CodeSnippet, extract_snippets

__all__ = [
    "LoadedRepository",
    "RepositoryError",
    "RepositoryNotFoundError",
    "RepositoryNotADirectoryError",
    "load_repository",
    "validate_repository",
    "GitMetadata",
    "extract_git_metadata",
    "LanguageInfo",
    "detect_languages",
    "Dependency",
    "extract_dependencies",
    "CodeSnippet",
    "extract_snippets",
    "RepositoryContext",
    "build_repository_context",
]
