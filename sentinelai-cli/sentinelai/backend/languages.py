"""
Language detection - which programming languages a repository contains,
by file extension only.

Deliberately extension-based, not content-based: sniffing file contents
(shebangs, syntax heuristics) is a much larger and fuzzier problem than
this MVP needs, and every later consumer (RepositoryContext, scanners)
can be built against a simple, deterministic signal first.

Directory pruning uses os.walk rather than a pathlib-only approach so
that ignored directories (.git, node_modules, ...) are never descended
into - not just filtered out afterward. pathlib.Path.walk() would give
the same pruning, but only exists on Python 3.12+; this project supports
3.9+, and Path.rglob() has no hook to stop descending into a matched
directory, so it would still walk .git/node_modules/etc. before
filtering. os.walk is the only stdlib option that is both 3.9-compatible
and prunes without wasted traversal.
"""
import logging
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Final, List, Mapping, Set, Tuple

from .loader import LoadedRepository

IGNORED_DIRECTORIES = {
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "node_modules",
    "dist",
    "build",
    ".cache",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
}

_EXTENSION_LANGUAGE_MAP: Final[Mapping[str, str]] = {
    ".py": "Python",
    ".js": "JavaScript",
    ".ts": "TypeScript",
    ".java": "Java",
    ".php": "PHP",
    ".go": "Go",
    ".c": "C/C++",
    ".cpp": "C/C++",
    ".cc": "C/C++",
    ".cxx": "C/C++",
    ".h": "C/C++",
    ".hpp": "C/C++",
}


@dataclass(frozen=True)
class LanguageInfo:
    name: str
    file_count: int
    extensions: Tuple[str, ...]


logger = logging.getLogger("sentinelai")


def detect_languages(repository: LoadedRepository) -> List[LanguageInfo]:
    """Walk `repository.absolute_path` and count files by recognized extension."""
    file_counts: Counter = Counter()
    extensions_seen: Dict[str, Set[str]] = {}

    for dirpath, dirnames, filenames in os.walk(repository.absolute_path):
        dirnames[:] = [d for d in dirnames if d not in IGNORED_DIRECTORIES]
        for filename in filenames:
            extension = Path(filename).suffix.lower()
            if not extension:
                continue
            language = _EXTENSION_LANGUAGE_MAP.get(extension)
            if language is None:
                continue
            file_counts[language] += 1
            extensions_seen.setdefault(language, set()).add(extension)

    languages = [
        LanguageInfo(name=language, file_count=count, extensions=tuple(sorted(extensions_seen[language])))
        for language, count in file_counts.items()
    ]
    languages.sort(key=lambda info: (-info.file_count, info.name))
    logger.debug(
        "language detection: languages=%d names=%s",
        len(languages),
        ",".join(language.name for language in languages) or "-",
    )
    return languages
