"""
Deterministic pre-flight validation of a generated patch.

Pure computation: no model call, no subprocess, no `git apply`, no filesystem
access, no network. The same patch text always produces the same result. This is
the deterministic half of the split the rest of this project keeps - a model may
*propose* a patch, and everything after that proposal, starting here, is ordinary
code that can be read and tested without a model running.

This module REPORTS and never REPAIRS. A hunk header whose counts disagree with
its body is rejected and returned as a HUNK_COUNT_MISMATCH; it is not corrected,
and no caller gets a "fixed" diff back. That restraint is deliberate: silently
rewriting a patch would mean applying something the model never actually
proposed, and would hide a generation defect precisely where it most needs to be
visible.

Scope is syntax, not applicability. A valid result says the diff is well formed
and any Python replacement parses. It says nothing about whether the context
lines match the real file, whether the target exists, or whether the change is
correct - all of which need the target file and belong to a later phase.

Unified diff grammar enforced here, per the format's own rules:

    --- <old path>
    +++ <new path>
    @@ -<old_start>[,<old_count>] +<new_start>[,<new_count>] @@ [heading]
    <body lines, each prefixed ' ', '-', '+', or '\\'>

An omitted count means 1, which is the format's documented default rather than a
convenience added here. The '\\' prefix is the "\\ No newline at end of file"
marker; it is a note about the preceding line and counts toward neither side.
"""
import ast
import re
import textwrap
from typing import List, Optional, Tuple

from sentinelai.contracts import StructuredPatch

from .models import PatchIssue, PatchIssueCode, PatchValidation

# Trailing text after the closing @@ is a section heading, which the format
# permits and which carries no meaning for validation.
_HUNK_HEADER = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")

_CONTEXT, _REMOVED, _ADDED, _NO_NEWLINE = " ", "-", "+", "\\"
_LEGAL_PREFIXES = (_CONTEXT, _REMOVED, _ADDED, _NO_NEWLINE)

_PYTHON_SUFFIX = ".py"


def validate_structured_patch(patch: StructuredPatch) -> PatchValidation:
    """Validate every part of a StructuredPatch that can be checked without the target file.

    The diff is always validated. The replacement block is validated only when
    present and only when the target is a Python file - this project cannot parse
    other languages, and reporting a Go snippet as a Python syntax error would be
    a false rejection.
    """
    issues: List[PatchIssue] = list(validate_unified_diff(patch.diff).issues)

    if patch.replacement is not None and patch.file.endswith(_PYTHON_SUFFIX):
        issues.extend(validate_python_replacement(patch.replacement).issues)

    return PatchValidation(issues=tuple(issues))


def validate_python_replacement(source: str) -> PatchValidation:
    """Check a replacement block parses as Python, via ast.parse().

    A replacement is usually a fragment lifted from inside a function, so it
    arrives indented and `ast.parse` rejects it with IndentationError even when
    the code is fine. Dedenting and re-parsing distinguishes "indented fragment"
    from "genuinely malformed". This changes nothing about the patch - only how
    it is inspected - and the original string is what a later phase would use.
    """
    try:
        ast.parse(source)
        return PatchValidation()
    except SyntaxError as first_error:
        dedented = textwrap.dedent(source)
        if dedented != source:
            try:
                ast.parse(dedented)
                return PatchValidation()
            except SyntaxError:
                pass
        return PatchValidation(
            issues=(
                PatchIssue(
                    code=PatchIssueCode.REPLACEMENT_SYNTAX_ERROR,
                    message=f"replacement block is not valid Python: {first_error.msg}",
                    line=first_error.lineno,
                ),
            )
        )


def validate_unified_diff(diff: str) -> PatchValidation:
    """Validate the structure of a unified diff. Reports every issue it can reach."""
    if not diff or not diff.strip():
        return _single(PatchIssueCode.EMPTY_DIFF, "diff is empty")

    lines = diff.splitlines()

    header_issues = _validate_file_headers(lines)
    if header_issues:
        # Without both file headers there is no reliable point to start reading
        # hunks from, so further findings would be guesses rather than facts.
        return PatchValidation(issues=tuple(header_issues))

    return PatchValidation(issues=tuple(_validate_hunks(lines)))


def _single(code: PatchIssueCode, message: str, line: Optional[int] = None) -> PatchValidation:
    return PatchValidation(issues=(PatchIssue(code=code, message=message, line=line),))


def _validate_file_headers(lines: List[str]) -> List[PatchIssue]:
    issues: List[PatchIssue] = []
    if not lines[0].startswith("--- "):
        issues.append(
            PatchIssue(
                code=PatchIssueCode.MISSING_OLD_FILE_HEADER,
                message="first line must be an old-file header beginning '--- '",
                line=1,
            )
        )
    if len(lines) < 2 or not lines[1].startswith("+++ "):
        issues.append(
            PatchIssue(
                code=PatchIssueCode.MISSING_NEW_FILE_HEADER,
                message="second line must be a new-file header beginning '+++ '",
                line=2,
            )
        )
    return issues


def _validate_hunks(lines: List[str]) -> List[PatchIssue]:
    issues: List[PatchIssue] = []
    index = 2
    hunks_seen = 0
    changed_lines_seen = False

    while index < len(lines):
        match = _HUNK_HEADER.match(lines[index])
        if match is None:
            issues.append(
                PatchIssue(
                    code=PatchIssueCode.MALFORMED_HUNK_HEADER,
                    message=(
                        "expected a hunk header of the form '@@ -old_start,old_count "
                        f"+new_start,new_count @@', found {lines[index]!r}"
                    ),
                    line=index + 1,
                )
            )
            # Everything after an unreadable header is unattributable to a hunk.
            return issues

        hunks_seen += 1
        header_line = index + 1
        declared_old, declared_new = _declared_counts(match)

        index, counted_old, counted_new, changed, prefix_issues = _scan_hunk_body(lines, index + 1)
        issues.extend(prefix_issues)
        changed_lines_seen = changed_lines_seen or changed

        if prefix_issues:
            # A body containing an illegal prefix has no trustworthy line counts,
            # so a count comparison here would report a second, derived failure
            # for what is really one problem.
            continue

        if (counted_old, counted_new) != (declared_old, declared_new):
            issues.append(
                PatchIssue(
                    code=PatchIssueCode.HUNK_COUNT_MISMATCH,
                    message=(
                        f"hunk header declares {declared_old} old and {declared_new} new lines, "
                        f"but the body contains {counted_old} old and {counted_new} new"
                    ),
                    line=header_line,
                )
            )

    if hunks_seen == 0:
        issues.append(
            PatchIssue(
                code=PatchIssueCode.NO_HUNKS,
                message="diff contains file headers but no '@@' hunk",
                line=None,
            )
        )
    elif not changed_lines_seen and not issues:
        issues.append(
            PatchIssue(
                code=PatchIssueCode.NO_CHANGES,
                message=(
                    "diff contains no '-' or '+' line, so it would change nothing; a patch "
                    "made only of context lines is structurally well formed but cannot be a fix"
                ),
                line=None,
            )
        )
    return issues


def _declared_counts(match: "re.Match") -> Tuple[int, int]:
    """An omitted count means 1 - the unified diff format's documented default."""
    old = int(match.group(2)) if match.group(2) is not None else 1
    new = int(match.group(4)) if match.group(4) is not None else 1
    return old, new


def _scan_hunk_body(lines: List[str], start: int) -> Tuple[int, int, int, bool, List[PatchIssue]]:
    """Read one hunk's body. Returns (next_index, old_count, new_count, had_change, issues)."""
    issues: List[PatchIssue] = []
    old_count = new_count = 0
    had_change = False
    index = start

    while index < len(lines) and not lines[index].startswith("@@"):
        line = lines[index]
        prefix = line[:1]

        if prefix not in _LEGAL_PREFIXES:
            issues.append(
                PatchIssue(
                    code=PatchIssueCode.MALFORMED_LINE_PREFIX,
                    message=(
                        "every line in a hunk body must begin with ' ', '-', '+', or '\\'; "
                        f"found {line!r}"
                        + (" (a blank context line must be written as a single space)" if line == "" else "")
                    ),
                    line=index + 1,
                )
            )
        elif prefix == _CONTEXT:
            old_count += 1
            new_count += 1
        elif prefix == _REMOVED:
            old_count += 1
            had_change = True
        elif prefix == _ADDED:
            new_count += 1
            had_change = True
        # _NO_NEWLINE annotates the previous line and counts toward neither side.

        index += 1

    return index, old_count, new_count, had_change, issues
