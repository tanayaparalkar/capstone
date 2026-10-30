"""
Bounded code context around a finding's line.

A standalone copy of the windowing in backend/code_context.py
(extract_code_context, which ai/prompt_builder.py uses today), left in
place until Phase 7 rewires the prompt onto this module. It differs on
purpose in four ways:

- Two ceilings, not one. `before`/`after` bound the window in lines;
  `max_chars` bounds it in characters. A ten-line window of minified
  JavaScript is still a 40 KB payload, and payload size is a privacy and
  cost concern. When max_chars forces a cut, whole context lines are
  dropped evenly from both ends so the target line stays in the middle,
  and truncated is set. If the target line alone is over the limit, it
  is cut to its first max_chars characters: there is no column to centre
  on, and a scanner's own evidence starts from the line's beginning too.
  Hitting the line window is not truncation - that window is what the
  caller asked for - so truncated means only that max_chars cut something.
- Failure raises CodeSliceError instead of returning None. A missing,
  unreadable or since-shortened file is an operational fault the caller
  records; an empty slice passed along silently would look like code.
- Lines are split on "\\n" only, like the scanners count them, not on
  str.splitlines()'s wider set (form feed, \\u2028, ...), which can shift
  every later line number away from the finding's.
- Bytes that are not UTF-8 are decoded as U+FFFD instead of rejecting the
  file, so a Latin-1 source still yields context with correct numbering.

The file is streamed and each line is read with a byte cap, so a
multi-megabyte single line costs at most a few times max_chars in
memory, and reading stops once the window's last line is in hand.

Imports nothing from the rest of ai/. The text is unredacted source:
callers pass it through ai/redaction before it goes anywhere near a
prompt.
"""
from pathlib import Path
from typing import BinaryIO

from pydantic import BaseModel, ConfigDict, Field, model_validator

# UTF-8 needs at most 4 bytes a character, so a line longer than this many
# bytes is certainly longer than max_chars characters.
_BYTES_PER_CHAR = 4


class CodeSliceError(Exception):
    """The requested slice could not be produced: file missing or unreadable, or the line is past its end."""


class CodeSlice(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str = Field(..., description="The slice's lines joined with '\\n', at most max_chars characters.")
    path: str = Field(..., description="The path the slice was read from, as given.")
    line_start: int = Field(..., ge=1, description="First line included, 1-based.")
    line_end: int = Field(..., ge=1, description="Last line included, 1-based and inclusive.")
    truncated: bool = Field(..., description="True if max_chars cut lines or characters from the requested window.")

    @model_validator(mode="after")
    def _validate_line_range(self) -> "CodeSlice":
        if self.line_end < self.line_start:
            raise ValueError("line_end cannot be before line_start")
        return self


def slice_around(
    path: Path, line: int, *, before: int = 5, after: int = 5, max_chars: int = 2000
) -> CodeSlice:
    """Return up to `before` lines above and `after` lines below `line`, within `max_chars` characters."""
    if line < 1:
        raise ValueError("line must be >= 1")
    if before < 0 or after < 0:
        raise ValueError("before and after must be >= 0")
    if max_chars < 1:
        raise ValueError("max_chars must be >= 1")

    first = max(1, line - before)
    window, lines_read = _read_lines(Path(path), first, line + after, max_chars * _BYTES_PER_CHAR + _BYTES_PER_CHAR)
    if lines_read < line:
        raise CodeSliceError(f"{path}: line {line} is past the end of the file ({lines_read} lines)")
    last = first + len(window) - 1

    target = window[line - first]
    if len(target) > max_chars:
        return CodeSlice(text=target[:max_chars], path=str(path), line_start=line, line_end=line, truncated=True)

    # Grow outward from the target one line per side at a time; a side stops at the
    # first line that does not fit, since a slice must stay contiguous.
    lo = hi = line
    used = len(target)
    grow_up, grow_down = lo > first, hi < last
    while grow_up or grow_down:
        if grow_up:
            cost = len(window[lo - 1 - first]) + 1
            if used + cost <= max_chars:
                lo, used = lo - 1, used + cost
                grow_up = lo > first
            else:
                grow_up = False
        if grow_down:
            cost = len(window[hi + 1 - first]) + 1
            if used + cost <= max_chars:
                hi, used = hi + 1, used + cost
                grow_down = hi < last
            else:
                grow_down = False

    return CodeSlice(
        text="\n".join(window[lo - first : hi - first + 1]),
        path=str(path),
        line_start=lo,
        line_end=hi,
        truncated=(lo, hi) != (first, last),
    )


def _read_lines(path: Path, first: int, last: int, byte_cap: int) -> tuple[list[str], int]:
    """Lines first..last (1-based, inclusive) of path, each cut to byte_cap bytes, and how many lines were read.

    Fewer lines come back if the file ends sooner; the count is then the file's length."""
    lines: list[str] = []
    number = 0
    try:
        with path.open("rb") as handle:
            while number < last:
                raw, more = _read_capped_line(handle, byte_cap)
                if not raw and not more:
                    break
                number += 1
                if number >= first:
                    lines.append(_decode(raw))
    except OSError as exc:
        raise CodeSliceError(f"{path}: cannot read source ({exc.strerror or type(exc).__name__})") from exc
    return lines, number


def _read_capped_line(handle: BinaryIO, byte_cap: int) -> tuple[bytes, bool]:
    """One line without its terminator, cut to byte_cap; the rest of an over-long line is skipped unread into memory.

    Returns (line, ended_with_newline). (b"", False) means end of file.
    """
    raw = handle.readline(byte_cap)
    if raw.endswith(b"\n"):
        return raw[:-1], True
    if len(raw) == byte_cap:
        while True:
            rest = handle.readline(byte_cap)
            if not rest or rest.endswith(b"\n"):
                return raw, bool(rest)
    return raw, False


def _decode(raw: bytes) -> str:
    if raw.endswith(b"\r"):
        raw = raw[:-1]
    return raw.decode("utf-8", errors="replace")
