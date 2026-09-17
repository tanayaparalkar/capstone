"""
Deterministic restoration from the backup tree.

The mirror of backup.py, and deliberately a separate module: creating a
preserved copy and putting it back are different operations with different
failure modes, and only one of them writes into the repository under recovery.
backup.py stays backup-only; this module never writes into the backup tree.

Byte-for-byte. Restoration copies exactly what was preserved, with no parsing,
re-rendering, diffing or interpretation of any kind. Whatever bytes backup.py
stored are the bytes that come back, which is what makes "restore" a fact rather
than a best effort.

No model call, no subprocess, no network. A restore is a copy and a rename.

Composition with Phase 2.2. Backups are addressed by path, with no manifest to
consult - repo/app/main.py is preserved at repo/.sentinelai/backups/app/main.py -
so restoration is the same mapping read in the other direction. Two consequences
follow for free. Because backups are never overwritten, a restore returns the
file to its state before the FIRST modification rather than the most recent one.
And because restoring does not consume the backup, repeated restores are
idempotent: the second produces the same bytes as the first.

Atomicity matches applicator.py's write path, for the same reason. The restored
contents are written to a temporary file in the target's own directory, flushed,
fsync'd, and renamed over the target with os.replace(). A reader sees either the
patched file or the fully restored one, never a mix, and a failure before the
rename leaves the target as it was. The temporary file is removed on every path.

Out of scope, and belonging to later phases or to no phase at all: pruning,
versioning, deleting backups, multiple rollback generations, interactive
confirmation, logging, notification, and background cleanup.
"""
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ._filesystem import atomic_replace
from .backup import BackupManager
from .exceptions import RollbackFailedError, RollbackNotFoundError

# Distinct from the applicator's suffix so crash debris names the operation
# that created it. Unchanged from the previous inline implementation.
_RESTORE_SUFFIX = ".sentinelai-restore"


@dataclass(frozen=True)
class RollbackRecord:
    """Deterministic description of one restoration.

    `succeeded` is True on every record this module returns: a restoration that
    fails raises RollbackFailedError rather than returning succeeded=False. That
    asymmetry is deliberate. A failed rollback means the repository may hold
    content nobody intended, which is the one outcome in this package that must
    not be possible to ignore by forgetting to check a boolean. The field exists
    so a caller logging or reporting outcomes has a uniform shape to read, not as
    a channel for signalling failure.

    `restored_at` is the one non-reproducible field, by necessity - it records
    when the copy was put back.
    """

    restored_path: Path
    backup_path: Path
    restored_at: datetime
    succeeded: bool = True


class RollbackManager:
    """Restores files from the backup tree created by BackupManager.

    Stateless apart from its root, so instances are interchangeable. The backup
    manager is injectable so a caller that configured a non-default backup
    directory restores from the same place it backed up to - otherwise a custom
    layout would back up correctly and restore from nowhere.
    """

    def __init__(self, root: Path, backup_manager: Optional[BackupManager] = None) -> None:
        self._root = Path(root).resolve()
        self._backups = backup_manager if backup_manager is not None else BackupManager(self._root)

    @property
    def backup_manager(self) -> BackupManager:
        return self._backups

    def backup_path_for(self, target: Path) -> Path:
        """Where `target`'s backup would live. Pure computation, creates nothing."""
        return self._backups.backup_path_for(target)

    def can_restore(self, target: Path) -> bool:
        """Whether a usable backup exists. Lets a caller ask without provoking an exception."""
        try:
            return self._backups.backup_path_for(target).is_file()
        except Exception:
            # backup_path_for raises for a target outside the root, which is a
            # perfectly good answer to "can this be restored" - no.
            return False

    def restore(self, target: Path) -> RollbackRecord:
        """Restore `target` from its backup, atomically, and return what was done.

        The backup is opened read-only and is never modified, moved or removed,
        so a failed restore leaves the preserved copy available for a retry.
        """
        resolved = Path(target).resolve()
        backup_path = self._backups.backup_path_for(resolved)

        if not backup_path.exists():
            raise RollbackNotFoundError(
                f"no backup exists for '{resolved}' (looked in '{backup_path}')"
            )
        if not backup_path.is_file():
            raise RollbackNotFoundError(
                f"backup for '{resolved}' is not a regular file: '{backup_path}'"
            )

        try:
            contents = backup_path.read_bytes()
            mode = os.stat(backup_path).st_mode & 0o7777
        except OSError as exc:
            raise RollbackFailedError(
                f"could not read the backup for '{resolved}' from '{backup_path}': {exc}"
            ) from exc

        self._atomic_restore(resolved, contents, mode)

        return RollbackRecord(
            restored_path=resolved,
            backup_path=backup_path,
            restored_at=datetime.now(tz=timezone.utc),
            succeeded=True,
        )

    @staticmethod
    def _atomic_restore(target: Path, contents: bytes, mode: int) -> None:
        """Write `contents` over `target` atomically, carrying the backup's permissions.

        The byte-level sequence lives in _filesystem.atomic_replace, shared with
        applicator.py. What stays here is policy: permissions come from the
        BACKUP rather than from the current file, so restoring undoes a change a
        patch made to the mode as well as to the content, and an OSError from
        the primitive means RollbackFailedError to this class's callers.
        """
        try:
            atomic_replace(target, contents, mode=mode, suffix=_RESTORE_SUFFIX)
        except OSError as exc:
            raise RollbackFailedError(f"could not restore '{target}': {exc}") from exc
