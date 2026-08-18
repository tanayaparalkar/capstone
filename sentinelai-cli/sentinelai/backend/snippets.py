"""
Code snippet extraction - short, fixed-size excerpts of source code read
from a repository's supported source files.

A future RAG/context consumer needs small, retrievable units of actual
code, not whole files; this module produces those units without any
opinion about which lines matter (that requires findings, which don't
exist yet - a scanner concern, not this module's).

Directory pruning reuses languages.py's IGNORED_DIRECTORIES directly
(the exact set, not a redefinition that could drift). The supported
extension set below intentionally duplicates languages.py's language map
instead of importing it: that map is a private, underscore-prefixed
implementation detail of languages.py, and importing a private name
across modules would couple the two beyond what the module boundary
promises. Both lists must be updated together if language support
changes.

Design decisions:

- Chunk size is a fixed 20 lines per snippet (the last chunk of a file
  may be shorter). No consumer has asked for a different or configurable
  size yet, so one constant is used rather than a speculative parameter.
- Snippet content joins lines with "\\n", regardless of the file's
  original line-ending style. Snippets are for reading/embedding, not
  reproducing a file byte-for-byte, so normalizing CRLF/LF here is a
  simplification, not a loss of anything a consumer needs.
- Files that fail strict UTF-8 decoding (binary files, or text in another
  encoding) are skipped entirely, not decoded with a lossy fallback
  (e.g. errors="replace"). A snippet containing replacement characters
  would be worse than no snippet - misleading to a reader and useless to
  an embedding model - so skipping is the safer failure mode.
- A file that disappears or becomes unreadable between directory listing
  and read (e.g. a broken symlink, a permission error) is skipped rather
  than aborting the whole extraction; one bad file should not prevent
  reporting snippets for the rest of the repository.
- An empty file (zero lines) produces zero snippets - there is nothing to
  extract.
"""
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

from .languages import IGNORED_DIRECTORIES
from .loader import LoadedRepository

# Intentionally duplicates languages.py's extension set - see module docstring.
_SUPPORTED_EXTENSIONS = {
    ".py",
    ".js",
    ".ts",
    ".java",
    ".php",
    ".go",
    ".c",
    ".cpp",
    ".cc",
    ".cxx",
    ".h",
    ".hpp",
}

_LINES_PER_SNIPPET = 20


@dataclass(frozen=True)
class CodeSnippet:
    file_path: str
    start_line: int
    end_line: int
    content: str


def extract_snippets(repository: LoadedRepository) -> List[CodeSnippet]:
    """Read supported source files under `repository.absolute_path` and split each into fixed-size line chunks."""
    snippets: List[CodeSnippet] = []

    for path in _discover_source_files(repository.absolute_path):
        snippets.extend(_read_snippets(path, repository.absolute_path))

    # Explicit final sort: discovery order is already deterministic (see
    # _discover_source_files), but sorting here makes that guarantee visible
    # at the point the result is returned, rather than implicit in how the
    # list happens to have been built.
    snippets.sort(key=lambda snippet: (snippet.file_path, snippet.start_line))
    return snippets


def _discover_source_files(root: Path) -> List[Path]:
    discovered = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in IGNORED_DIRECTORIES]
        for filename in filenames:
            if Path(filename).suffix.lower() in _SUPPORTED_EXTENSIONS:
                discovered.append(Path(dirpath) / filename)

    # os.walk's traversal order is filesystem-dependent, not guaranteed
    # alphabetical - sort explicitly so output ordering doesn't vary by platform.
    discovered.sort()
    return discovered


def _read_snippets(path: Path, repository_root: Path) -> List[CodeSnippet]:
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return []

    lines = text.splitlines()
    if not lines:
        return []

    file_path = str(path.relative_to(repository_root))
    snippets = []
    for chunk_start in range(0, len(lines), _LINES_PER_SNIPPET):
        chunk_lines = lines[chunk_start : chunk_start + _LINES_PER_SNIPPET]
        snippets.append(
            CodeSnippet(
                file_path=file_path,
                start_line=chunk_start + 1,
                end_line=chunk_start + len(chunk_lines),
                content="\n".join(chunk_lines),
            )
        )
    return snippets
