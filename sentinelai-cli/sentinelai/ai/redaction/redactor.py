"""
Secret redaction over code slices, before they reach a prompt.

Runs over EVERY slice, not only GitLeaks findings: a slice built around
a Semgrep SQL injection can carry a credential on a neighbouring line.

Two sources of spans are merged into one pass over the original text:

- redact_text(): the regex patterns in ai/redaction/patterns.py.
- SecretIndex.redact_slice(): additionally, the positions (file, line)
  GitLeaks already flagged, plus any unredacted literal it was given.

GitLeaks runs with --redact (scanners/gitleaks.py), so a finding's
raw_evidence is normally the match with the secret replaced by the
word REDACTED, e.g. `api = "REDACTED"`. That evidence is used as a
template to locate the secret on its flagged line; when the template
does not fit (file edited since the scan, multi-line finding), the
whole line is masked instead. Fail closed: a flagged line never passes
through unmasked.

Shape is preserved: each secret becomes «REDACTED:<pattern-name>»,
and a masked span keeps its newlines, so line numbers in the slice
still line up with the scanner's.

Never logs, never puts a secret into an exception message, and never
returns one: the second element of every result is pattern names only,
one per masked secret, in text order. Pure: no I/O of any kind.
"""
import re
from dataclasses import dataclass
from typing import Iterable, Iterator, Optional

from sentinelai.contracts import ScannerFinding

from .patterns import MARKER_PREFIX, PATTERNS, redaction_marker

_MARKER = re.compile(re.escape(MARKER_PREFIX) + r"[^»\s]*»")
_GITLEAKS_REDACTED = "REDACTED"
# Shorter literals would mask unrelated text all over a slice.
_MIN_LITERAL_LENGTH = 4
_INDEX_RANK = 0


@dataclass(frozen=True)
class _Span:
    start: int
    end: int
    name: str
    rank: int


def redact_text(text: str) -> tuple[str, list[str]]:
    """Returns (redacted_text, pattern_names_hit).
    NEVER returns, logs, or raises the secret value itself."""
    return _redact(text, ())


def _pattern_spans(text: str) -> Iterator[_Span]:
    for rank, pattern in enumerate(PATTERNS, start=_INDEX_RANK + 1):
        group = "s" if "s" in pattern.regex.groupindex else 0
        for match in pattern.regex.finditer(text):
            start, end = match.span(group)
            value = text[start:end]
            if not _MARKER.sub("", value).strip():
                continue  # empty, or only already-redacted markers: keeps redaction idempotent
            if pattern.accept is not None and not pattern.accept(value):
                continue
            yield _Span(start, end, pattern.name, rank)


def _redact(text: str, extra_spans: Iterable[_Span]) -> tuple[str, list[str]]:
    spans = sorted([*_pattern_spans(text), *extra_spans], key=lambda s: (s.start, -s.end))
    merged: list[_Span] = []
    for span in spans:
        if merged and span.start < merged[-1].end:
            last = merged[-1]
            best = last if last.rank <= span.rank else span
            merged[-1] = _Span(last.start, max(last.end, span.end), best.name, best.rank)
        else:
            merged.append(span)

    pieces: list[str] = []
    cursor = 0
    for span in merged:
        pieces.append(text[cursor : span.start])
        pieces.append(redaction_marker(span.name) + "\n" * text.count("\n", span.start, span.end))
        cursor = span.end
    pieces.append(text[cursor:])
    return "".join(pieces), [span.name for span in merged]


@dataclass(frozen=True)
class _FlaggedRange:
    path: tuple[str, ...]
    line_start: int
    line_end: int
    name: str
    evidence: Optional[str]


class SecretIndex:
    """Repo-scoped record of what GitLeaks flagged, so a known secret is masked in any slice that contains it."""

    def __init__(self) -> None:
        self._ranges: list[_FlaggedRange] = []
        self._literals: dict[str, str] = {}

    def __repr__(self) -> str:
        return f"SecretIndex(ranges={len(self._ranges)}, literals={len(self._literals)})"

    def add_from_findings(self, findings: list[ScannerFinding]) -> None:
        for finding in findings:
            if finding.scanner.lower() != "gitleaks":
                continue
            name = _index_pattern_name(finding.rule_id)
            evidence = finding.raw_evidence or None
            if evidence and _GITLEAKS_REDACTED not in evidence and len(evidence) >= _MIN_LITERAL_LENGTH:
                self._literals.setdefault(evidence, name)
            if finding.file and finding.line_start is not None:
                self._ranges.append(
                    _FlaggedRange(
                        path=_path_key(finding.file),
                        line_start=finding.line_start,
                        line_end=finding.line_end or finding.line_start,
                        name=name,
                        evidence=evidence,
                    )
                )

    def redact_slice(self, text: str, path: str, line_start: int) -> tuple[str, list[str]]:
        """Redact a slice of `path` whose first line is `line_start` (1-based). Same guarantees as redact_text()."""
        if line_start < 1:
            raise ValueError("line_start must be >= 1")
        spans = [*self._range_spans(text, path, line_start), *self._literal_spans(text)]
        return _redact(text, spans)

    def _literal_spans(self, text: str) -> Iterator[_Span]:
        for literal, name in self._literals.items():
            start = text.find(literal)
            while start != -1:
                yield _Span(start, start + len(literal), name, _INDEX_RANK)
                start = text.find(literal, start + len(literal))

    def _range_spans(self, text: str, path: str, line_start: int) -> Iterator[_Span]:
        slice_path = _path_key(path)
        lines = _line_bounds(text)
        line_last = line_start + len(lines) - 1
        for flagged in self._ranges:
            if not _same_file(flagged.path, slice_path):
                continue
            first = max(flagged.line_start, line_start)
            last = min(flagged.line_end, line_last)
            if first > last:
                continue
            if flagged.line_start == flagged.line_end:
                located = list(_locate_on_line(text, lines[first - line_start], flagged))
                if located:
                    yield from located
                    continue
            start = _first_non_blank(text, *lines[first - line_start])
            end = lines[last - line_start][1]
            if start < end:
                yield _Span(start, end, flagged.name, _INDEX_RANK)


def _locate_on_line(text: str, bounds: tuple[int, int], flagged: _FlaggedRange) -> Iterator[_Span]:
    """Find the exact secret on its flagged line from the GitLeaks evidence; yields nothing if it doesn't fit."""
    evidence = flagged.evidence
    if not evidence or "\n" in evidence:
        return
    start = _first_non_blank(text, *bounds)
    line = text[start : bounds[1]]
    if _GITLEAKS_REDACTED in evidence:
        parts = evidence.split(_GITLEAKS_REDACTED)
        regex = re.escape(parts[0])
        for index, part in enumerate(parts[1:], start=1):
            last = index == len(parts) - 1
            regex += "(.+)" if last and not part else "(.+?)" + re.escape(part)
        match = re.search(regex, line)
        if match:
            for group in range(1, len(parts)):
                yield _Span(start + match.start(group), start + match.end(group), flagged.name, _INDEX_RANK)
    else:
        found = line.find(evidence)
        if found != -1:
            yield _Span(start + found, start + found + len(evidence), flagged.name, _INDEX_RANK)


def _line_bounds(text: str) -> list[tuple[int, int]]:
    """(start, end) of each line's content, excluding its line terminator. Splits on \\n only, like the scanners count."""
    bounds = []
    position = 0
    while True:
        newline = text.find("\n", position)
        end = len(text) if newline == -1 else newline
        content_end = end - 1 if end > position and text[end - 1] == "\r" else end
        bounds.append((position, content_end))
        if newline == -1:
            return bounds
        position = newline + 1


def _first_non_blank(text: str, start: int, end: int) -> int:
    while start < end and text[start] in " \t":
        start += 1
    return start


def _path_key(path: str) -> tuple[str, ...]:
    # Case-folded on every platform: over-matching a path only masks more.
    return tuple(part for part in path.replace("\\", "/").casefold().split("/") if part not in ("", "."))


def _same_file(a: tuple[str, ...], b: tuple[str, ...]) -> bool:
    """Scanner paths are absolute, slice paths may be relative: match when one is a path-suffix of the other."""
    if not a or not b:
        return False
    n = min(len(a), len(b))
    return a[-n:] == b[-n:]


def _index_pattern_name(rule_id: str) -> str:
    slug = re.sub(r"[^a-z0-9._-]+", "-", rule_id.lower()).strip("-")
    return f"gitleaks:{slug or 'unknown'}"
