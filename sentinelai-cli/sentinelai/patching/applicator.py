"""
Deterministic patch application.

The first component in this project that writes into a scanned repository.
Everything before it - scanners, correlation, the AI layer, validation - is
read-only, so the guarantees here are stated explicitly rather than assumed.

No model call, no subprocess, no `git apply`, no network. Given the same file
and the same patch, this module produces the same bytes every time. A model may
have *proposed* the patch; deciding whether it fits and writing it are ordinary
code.

Atomicity. The updated contents are written to a temporary file in the target's
own directory, flushed, fsync'd, and then renamed over the target with
os.replace(), which is atomic on POSIX and Windows. A reader therefore observes
either the original file or the complete new one, never a half-written mix. The
temporary file shares a directory with the target because a rename is only
atomic within one filesystem; /tmp is frequently a different one. If anything
fails before the rename, the temporary file is removed and the target is
untouched.

This module REFUSES rather than REPAIRS, in both directions. It will not apply a
patch that validation rejected, and it will not apply a hunk whose context does
not match the file - no fuzzy matching, no offset searching, no "close enough"
application. A patch that does not fit is a patch that does not get applied.

Hunk parsing lives here rather than being imported from validator.py. The two
modules ask different questions - "is this well formed" versus "which lines does
this change" - and validator.py's regex is a private implementation detail.
backend/snippets.py sets the precedent that importing a private name across a
module boundary is coupling, not reuse. Validation still runs first, so parsing
here can rely on structure that has already been checked.

Out of scope by design, and belonging to later phases: backups, snapshots,
rollback, undo, git integration, CLI wiring, and any repair of malformed
patches.
"""
import os
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from sentinelai.contracts import StructuredPatch

from ._filesystem import atomic_replace
from .backup import BackupManager, BackupRecord
from .exceptions import (
    PatchError,
    PatchApplicationError,
    PatchTargetNotFoundError,
    PatchValidationError,
    PatchVerificationError,
    PatchWriteError,
    ReplacementMismatchError,
    RollbackError,
)
from .models import PatchValidation
from .rollback import RollbackManager, RollbackRecord
from .safety import RepositoryState, RepositoryStatus, inspect_repository
from .validator import validate_structured_patch

_HUNK_HEADER = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")

_CONTEXT, _REMOVED, _ADDED, _NO_NEWLINE = " ", "-", "+", "\\"

_DEFAULT_NEWLINE = "\n"

# Names the temporary file so crash debris is attributable to the operation
# that created it. Unchanged from the previous inline implementation.
_WRITE_SUFFIX = ".sentinelai-tmp"


class PatchStrategy(str, Enum):
    """Which representation of the fix to apply.

    StructuredPatch.diff is required by the contract and StructuredPatch.
    replacement is optional, so the diff is the default. The choice is an
    explicit parameter rather than inferred from which fields happen to be
    populated: a patch carrying both would otherwise be applied one way or the
    other depending on generation luck, which is exactly the kind of implicit
    behaviour this project keeps out of its deterministic half.
    """

    UNIFIED_DIFF = "unified_diff"
    REPLACEMENT = "replacement"


@dataclass(frozen=True)
class PatchApplicationResult:
    """Everything one application produced, for a caller that needs more than the text.

    apply() still returns just the updated contents, which is what Phase 2.1
    callers expect and what most callers want. This richer view exists so a
    future CLI can report where the backup went and whether the repository was
    dirty without the applicator holding state between calls or growing an
    output parameter.
    """

    contents: str
    backup: Optional[BackupRecord]
    repository: RepositoryStatus


class PatchApplicator:
    """Applies a validated StructuredPatch to a file on disk.

    Stateless apart from its configuration, so one instance may be reused across
    findings and two instances behave identically.

    `root`, when given, bounds where this object is willing to write: the
    resolved target must sit inside it. It is optional because a StructuredPatch
    carries an absolute path already and no repository root is threaded through
    the AI layer, but supplying one is strongly preferred for anything acting on
    model-proposed paths - this is the one component in the project that writes,
    and a path is the one field a model controls.
    """

    def __init__(
        self,
        root: Optional[Path] = None,
        backup_manager: Optional[BackupManager] = None,
    ) -> None:
        self._root = root.resolve() if root is not None else None
        # Orchestrated, not owned: this class decides WHEN a backup happens, and
        # BackupManager decides what a backup IS. An explicitly supplied manager
        # wins, otherwise one is derived from the root, and with neither there is
        # no repository layout to mirror so no backup is taken.
        if backup_manager is not None:
            self._backups: Optional[BackupManager] = backup_manager
        elif self._root is not None:
            self._backups = BackupManager(self._root)
        else:
            self._backups = None
        # Restoration is delegated, never reimplemented here. The rollback
        # manager shares the backup manager so a custom backup directory is
        # restored from the same place it was written to.
        self._rollback: Optional[RollbackManager] = (
            RollbackManager(self._root or self._backups.backup_root.parent, self._backups)
            if self._backups is not None
            else None
        )

    # --- public API -------------------------------------------------------------------------------------------------

    def apply(
        self,
        patch: StructuredPatch,
        validation: Optional[PatchValidation] = None,
        strategy: PatchStrategy = PatchStrategy.UNIFIED_DIFF,
        target: Optional[Path] = None,
        dry_run: bool = False,
    ) -> str:
        """Apply `patch` to its target file and return the new contents.

        `validation` is the verdict from patching/validator.py. It is accepted
        rather than always recomputed so a caller that has already validated
        does not pay for it twice; when omitted it is computed here. Either way
        an invalid verdict raises before the target is opened.

        Returns the full updated file contents, which is also what was written.
        """
        verdict = validation if validation is not None else validate_structured_patch(patch)
        if not verdict.is_valid:
            raise PatchValidationError(
                "refusing to apply a patch that failed validation: "
                + "; ".join(f"{issue.code.value}: {issue.message}" for issue in verdict.issues)
            )

        return self.apply_with_result(patch, verdict, strategy, target, dry_run).contents

    def apply_with_result(
        self,
        patch: StructuredPatch,
        validation: Optional[PatchValidation] = None,
        strategy: PatchStrategy = PatchStrategy.UNIFIED_DIFF,
        target: Optional[Path] = None,
        dry_run: bool = False,
    ) -> "PatchApplicationResult":
        """Apply `patch`, returning the contents together with backup and repository state.

        The execution order here is the contract of this phase and is not
        rearrangeable: validate, then inspect, then read, then render, then back
        up, then write. Backup sits immediately before the write and after the
        render so that a patch which cannot be rendered leaves no backup behind,
        and so no write can happen without a preserved original - if the backup
        raises, the write is never reached.
        """
        verdict = validation if validation is not None else validate_structured_patch(patch)
        if not verdict.is_valid:
            raise PatchValidationError(
                "refusing to apply a patch that failed validation: "
                + "; ".join(f"{issue.code.value}: {issue.message}" for issue in verdict.issues)
            )

        path = self._resolve_target(patch, target)
        repository = self._inspect_repository()
        original = self._read(path)
        updated = self.render(patch, original, strategy=strategy)

        if dry_run:
            # Everything above this line is computation: validate, inspect, read,
            # render. Everything below it touches the repository. Returning here
            # is what makes a dry run a preview rather than a simulation - the
            # contents reported are the same bytes a real run would write,
            # produced by the same render() call, not by a parallel code path.
            #
            # The backup is skipped deliberately rather than taken "just in
            # case": writing into .sentinelai/backups/ would itself modify the
            # repository, which is exactly what a dry run promises not to do.
            # Nothing is written, so there is nothing to verify and nothing to
            # roll back.
            return PatchApplicationResult(contents=updated, backup=None, repository=repository)

        record = self._backups.backup(path) if self._backups is not None else None

        try:
            self._atomic_write(path, updated)
            self._verify_written(path, updated)
        except PatchError as exc:
            # The repository must never be left partially modified. A failed
            # _atomic_write already leaves the target untouched - the rename is
            # atomic - so this path matters most for a verification failure,
            # where the write reported success but the file does not hold what
            # was intended. Restoring unconditionally is still correct: with an
            # untouched target the restore is a no-op in content terms.
            exc.rollback = self._rollback_after_failure(path, record)
            raise

        return PatchApplicationResult(contents=updated, backup=record, repository=repository)

    def _rollback_after_failure(
        self, path: Path, record: Optional[BackupRecord]
    ) -> Optional[RollbackRecord]:
        """Restore `path` from its backup, if one was taken. Never swallows a rollback failure.

        A rollback that itself fails is the worst outcome this package has: the
        patch failed AND the original could not be put back. It is raised rather
        than reported, chained from the failure that triggered it so neither
        cause is lost.
        """
        if record is None or self._rollback is None:
            return None
        return self._rollback.restore(path)

    @staticmethod
    def _verify_written(path: Path, expected: str) -> None:
        """Read the file back and confirm it holds exactly what was written.

        Cheap insurance against a write that reported success without landing -
        a full disk that surfaced late, a filesystem that truncated, or another
        process writing the same path. Without this the caller's only evidence
        that the patch applied is that no exception was raised.
        """
        try:
            with open(path, "r", encoding="utf-8", newline="") as handle:
                actual = handle.read()
        except (OSError, UnicodeDecodeError) as exc:
            raise PatchVerificationError(
                f"could not read '{path}' back after writing it: {exc}"
            ) from exc

        if actual != expected:
            raise PatchVerificationError(
                f"'{path}' does not match what was written: expected {len(expected)} characters, "
                f"found {len(actual)}"
            )

    def _inspect_repository(self) -> RepositoryStatus:
        """Observe repository state. Never aborts, never warns - the caller decides."""
        if self._root is None:
            return RepositoryStatus(
                state=RepositoryState.NOT_CHECKED,
                detail="no repository root configured, so no git inspection was attempted",
            )
        return inspect_repository(self._root)

    def render(
        self,
        patch: StructuredPatch,
        original: str,
        strategy: PatchStrategy = PatchStrategy.UNIFIED_DIFF,
    ) -> str:
        """Compute the updated contents without touching the filesystem.

        Separated from apply() so the transformation can be tested, diffed and
        reasoned about with no file involved, and so a later phase can preview a
        change before committing to it.
        """
        if strategy is PatchStrategy.REPLACEMENT:
            return self._render_replacement(patch, original)
        return self._render_unified_diff(patch, original)

    # --- target resolution ------------------------------------------------------------------------------------------

    def _resolve_target(self, patch: StructuredPatch, target: Optional[Path]) -> Path:
        path = (target if target is not None else Path(patch.file)).resolve()

        if self._root is not None and self._root not in path.parents and path != self._root:
            raise PatchTargetNotFoundError(
                f"refusing to write outside the configured root: '{path}' is not inside '{self._root}'"
            )
        if not path.exists():
            raise PatchTargetNotFoundError(f"patch target does not exist: '{path}'")
        if not path.is_file():
            raise PatchTargetNotFoundError(f"patch target is not a regular file: '{path}'")
        return path

    @staticmethod
    def _read(path: Path) -> str:
        # newline="" disables universal-newline translation, so a CRLF file is
        # not silently rewritten with LF endings by the round trip.
        try:
            with open(path, "r", encoding="utf-8", newline="") as handle:
                return handle.read()
        except (OSError, UnicodeDecodeError) as exc:
            raise PatchTargetNotFoundError(f"could not read patch target '{path}': {exc}") from exc

    # --- unified diff -----------------------------------------------------------------------------------------------

    def _render_unified_diff(self, patch: StructuredPatch, original: str) -> str:
        lines = original.splitlines(keepends=True)
        newline = _detect_newline(original)
        output: List[str] = []
        cursor = 0

        for header_index, start, body in _parse_hunks(patch.diff):
            hunk_start = max(start - 1, 0)
            if hunk_start < cursor:
                raise PatchApplicationError(
                    f"hunk at diff line {header_index} starts at file line {start}, which the "
                    "previous hunk already consumed; overlapping hunks cannot be applied"
                )
            output.extend(lines[cursor:hunk_start])
            cursor = hunk_start

            for offset, body_line in enumerate(body):
                prefix, content = body_line[:1], body_line[1:]

                if prefix == _NO_NEWLINE:
                    continue

                if prefix == _ADDED:
                    output.append(content + newline)
                    continue

                # Context and removed lines must both be present in the file.
                if cursor >= len(lines):
                    raise PatchApplicationError(
                        f"hunk at diff line {header_index} expects a line at file line "
                        f"{cursor + 1}, but the file ends at line {len(lines)}"
                    )
                actual = _strip_terminator(lines[cursor])
                if actual != content:
                    raise PatchApplicationError(
                        f"hunk at diff line {header_index} does not match the file at line "
                        f"{cursor + 1}: expected {content!r}, found {actual!r}"
                    )
                if prefix == _CONTEXT:
                    output.append(lines[cursor])
                cursor += 1

        output.extend(lines[cursor:])
        return "".join(output)

    # --- replacement block ------------------------------------------------------------------------------------------

    def _render_replacement(self, patch: StructuredPatch, original: str) -> str:
        """Replace lines [start_line, end_line] with the replacement block.

        Known limitation, not compensated for here: StructuredPatch.replacement
        is a NonBlankStr, whose strip_whitespace=True removes the block's leading
        indentation before this method sees it. An indented fragment therefore
        arrives dedented, and for a multi-line block only the first line is
        affected, which leaves the result inconsistently indented. Re-indenting
        would mean inventing code the model never sent, so it is not done.
        Prefer PatchStrategy.UNIFIED_DIFF, which carries indentation intact and
        is the default for this reason.
        """
        if patch.replacement is None:
            raise ReplacementMismatchError(
                "cannot apply a replacement: the patch carries no replacement block"
            )
        if patch.start_line is None or patch.end_line is None:
            raise ReplacementMismatchError(
                "cannot apply a replacement: the patch carries no start_line/end_line, so the "
                "span to replace cannot be located"
            )
        if patch.end_line < patch.start_line:
            raise ReplacementMismatchError(
                f"cannot apply a replacement: end_line {patch.end_line} precedes "
                f"start_line {patch.start_line}"
            )

        lines = original.splitlines(keepends=True)
        if patch.start_line > len(lines) or patch.end_line > len(lines):
            raise ReplacementMismatchError(
                f"cannot apply a replacement: lines {patch.start_line}-{patch.end_line} fall "
                f"outside the file, which has {len(lines)} line(s)"
            )

        newline = _detect_newline(original)
        replacement = [line + newline for line in patch.replacement.splitlines()]
        return "".join(lines[: patch.start_line - 1] + replacement + lines[patch.end_line :])

    # --- atomic write -----------------------------------------------------------------------------------------------

    @staticmethod
    def _atomic_write(path: Path, contents: str) -> None:
        """Write `contents` over `path` atomically, preserving the file's own permissions.

        The byte-level sequence lives in _filesystem.atomic_replace, shared with
        rollback.py so the guarantee has one implementation. What stays here is
        policy this class is the one to decide: a patch preserves the mode the
        target already has, and an OSError from the primitive means
        PatchWriteError to this class's callers.

        Encoding to UTF-8 here rather than writing text through the primitive
        keeps the primitive byte-oriented. It is the same bytes either way: the
        previous implementation opened the temporary file with newline="", so no
        line-ending translation was applied then and none is applied now.
        """
        try:
            atomic_replace(
                path,
                contents.encode("utf-8"),
                mode=os.stat(path).st_mode & 0o7777,
                suffix=_WRITE_SUFFIX,
            )
        except OSError as exc:
            raise PatchWriteError(f"could not write patched contents to '{path}': {exc}") from exc


# --- helpers ------------------------------------------------------------------------------------------------------


def _parse_hunks(diff: str) -> List[Tuple[int, int, Sequence[str]]]:
    """Return (diff_line_number, old_start, body_lines) for each hunk, in order.

    Assumes the diff already passed validation, so a malformed header here would
    be a contradiction rather than a user-facing error; such a hunk is skipped
    rather than guessed at.
    """
    hunks: List[Tuple[int, int, Sequence[str]]] = []
    lines = diff.splitlines()
    index = 0
    while index < len(lines):
        match = _HUNK_HEADER.match(lines[index])
        if match is None:
            index += 1
            continue
        header_index = index + 1
        old_start = int(match.group(1))
        index += 1
        body: List[str] = []
        while index < len(lines) and not lines[index].startswith("@@"):
            body.append(lines[index])
            index += 1
        hunks.append((header_index, old_start, body))
    return hunks


def _strip_terminator(line: str) -> str:
    """Remove only the line terminator, preserving any other trailing whitespace.

    Trailing spaces are significant when matching a diff against a file, so
    rstrip() would make two different lines compare equal.
    """
    if line.endswith("\r\n"):
        return line[:-2]
    if line.endswith("\n") or line.endswith("\r"):
        return line[:-1]
    return line


def _detect_newline(text: str) -> str:
    """Use the file's own first line terminator for lines this module adds."""
    carriage = text.find("\r\n")
    if carriage != -1 and (text.find("\n") == carriage + 1):
        return "\r\n"
    return _DEFAULT_NEWLINE
