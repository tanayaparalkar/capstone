"""
Finding-anchored source context - the lines surrounding a scanner
finding's own location, read from disk.

Distinct from snippets.py, and deliberately not built on it. That module
chunks whole repositories into fixed 20-line windows for retrieval and
says so explicitly: it has "no opinion about which lines matter (that
requires findings, which don't exist yet)". Findings exist now, so this
module is that opinion - one window, centred on one finding, sized
relative to it. Reusing snippets.py would mean re-reading and re-chunking
an entire repository to locate a single window, and the chunk boundaries
would not align with the finding anyway.

Why this exists at all: the AI layer's prompt previously showed only the
scanner's own `raw_evidence` - typically the single matched line - plus a
file:line string. That is enough to explain a finding but not to rewrite
one: the remediation analyst cannot see the enclosing function, the
imports, the indentation, or the surrounding statements a real fix has to
fit into, which is why its prompt currently tells it not to guess at
surrounding code.

No repository root is required, and none is accepted. Semgrep, Bandit and
GitLeaks all emit ABSOLUTE paths in ScannerFinding.file (verified against
a live scan of the benchmark fixture), so a finding already carries
everything needed to locate its own source. Taking a root here would add
a parameter every caller would have to thread through the AI layer for no
gain, and would put a filesystem path into the one place the project has
been explicit it should not go.

Failure is always None, never an exception. This feeds a prompt; a
missing, unreadable, binary, or since-deleted file means the model sees
what it saw before this module existed, which is a degraded prompt rather
than a failed scan. Dependency findings (Trivy, OSV-Scanner) have
file-but-no-line or no file at all and legitimately return None - there is
no source line to surround.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

# Matches snippets.py's window size. Chosen to agree with the existing
# retrieval chunk size rather than to be independently tuned: a reviewer
# comparing a retrieved chunk against this context sees comparable spans.
DEFAULT_CONTEXT_LINES = 20

# Read guard. A minified bundle or a vendored data file can be megabytes on
# one line; reading it to extract twenty lines around line 1 would cost far
# more than the context is worth, and the result would be unreadable as
# context anyway. Files above this are treated as unavailable.
_MAX_FILE_BYTES = 2_000_000


@dataclass(frozen=True)
class CodeContext:
    """A window of source lines around a finding, with the finding's own span marked.

    `lines` holds the window only - not the whole file - and carries no line
    numbers of its own. Numbering is presentation, and belongs to whoever
    renders this (ai/prompt_builder.py), not to the reader.

    start_line/end_line describe where the window sits in the file; focus_start/
    focus_end describe the finding's own span within it. Both are 1-based and
    inclusive, matching ScannerFinding.line_start/line_end, so no caller has to
    convert between conventions.
    """

    file_path: str
    start_line: int
    end_line: int
    focus_start: int
    focus_end: int
    lines: Tuple[str, ...]


def extract_code_context(
    file_path: Optional[str],
    line_start: Optional[int],
    line_end: Optional[int] = None,
    context_lines: int = DEFAULT_CONTEXT_LINES,
) -> Optional[CodeContext]:
    """Read `context_lines` lines either side of a finding's span, or None if unavailable.

    Takes the finding's fields rather than a ScannerFinding so that backend/
    keeps not importing contracts/ - the same leaf discipline the rest of this
    package follows - and so this is callable with a bare path and line number
    from a test.

    `context_lines=0` is meaningful and allowed: it yields the finding's own
    lines with no surrounding context, which is the narrowest useful window
    rather than an empty one.
    """
    if not file_path or line_start is None or line_start < 1 or context_lines < 0:
        return None

    path = Path(file_path)
    try:
        if not path.is_file() or path.stat().st_size > _MAX_FILE_BYTES:
            return None
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        # Same fallback as snippets.py._read_snippets: unreadable and
        # undecodable files are skipped, not reported.
        return None

    lines: List[str] = text.splitlines()
    if not lines or line_start > len(lines):
        # A line number past EOF means the file changed since the scan. Showing
        # the wrong lines would be worse than showing none.
        return None

    # line_end is advisory: scanners disagree about whether it is set, and a
    # value below line_start (or past EOF) is clamped rather than trusted.
    focus_end = line_end if line_end is not None and line_end >= line_start else line_start
    focus_end = min(focus_end, len(lines))

    start_line = max(1, line_start - context_lines)
    end_line = min(len(lines), focus_end + context_lines)

    return CodeContext(
        file_path=file_path,
        start_line=start_line,
        end_line=end_line,
        focus_start=line_start,
        focus_end=focus_end,
        lines=tuple(lines[start_line - 1 : end_line]),
    )
