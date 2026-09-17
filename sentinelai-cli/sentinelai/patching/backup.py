"""
Pre-write backup storage.

Preserves a file's original contents before PatchApplicator modifies it, so a
later phase can restore it. Nothing here reads a backup back or restores
anything - creating the copy and restoring from it are separate operations with
separate failure modes, and only the first belongs to this phase.

Pure filesystem and stdlib: no model call, no subprocess, no network, no git. A
backup is a byte-for-byte copy at a deterministic path, so the same target under
the same root always maps to the same backup location.

Layout mirrors the repository, so a backup is findable by inspection rather than
by consulting an index:

    repo/app/main.py  ->  repo/.sentinelai/backups/app/main.py

There is no manifest, no database and no naming scheme to decode; the path IS
the key. That is what lets a restore be a copy in the other direction.

Never overwritten, and that is the load-bearing property. A backup records the
state before the FIRST modification, so applying two patches to one file leaves
the original recoverable rather than the intermediate. The second call reports
`already_existed=True` and copies nothing. The no-overwrite guarantee is enforced
by os.link() against an existing destination rather than by an exists() check
followed by a write, so two concurrent callers cannot both decide the backup is
missing and race to create it.

Backups live inside the repository (under .sentinelai/) rather than in a temp
directory so they survive a crash, a reboot, and the end of the process, and so
the backup shares a filesystem with its target - a requirement for the atomic
link used below.
"""
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .exceptions import BackupError

# Relative to the repository root. A dotted directory keeps it out of ordinary
# listings, and the nesting leaves room for sibling state later without a second
# top-level directory appearing in someone's repository.
BACKUP_DIRECTORY = Path(".sentinelai") / "backups"


@dataclass(frozen=True)
class BackupRecord:
    """Deterministic description of one backup.

    Carries exactly what a restore needs - where the original lives and where its
    preserved copy lives - so a later phase needs no lookup, index or heuristic
    to find the counterpart.

    `created_at` is the one field that is not reproducible, by necessity: it
    records when the copy was taken. For a backup that already existed it is that
    file's modification time, not the time of the call, so the record always
    describes the backup itself rather than the attempt to make one.

    `already_existed` True means this call copied nothing and an earlier backup
    is being reported. Treating that as success rather than as an error is
    deliberate - it is what preserves the pre-FIRST-modification state.
    """

    original_path: Path
    backup_path: Path
    created_at: datetime
    already_existed: bool


class BackupManager:
    """Creates backups under `root`/.sentinelai/backups, mirroring the repository layout.

    Stateless apart from its root, so instances are interchangeable and nothing
    accumulates between calls.
    """

    def __init__(self, root: Path, backup_directory: Path = BACKUP_DIRECTORY) -> None:
        self._root = Path(root).resolve()
        self._backup_root = self._root / backup_directory

    @property
    def backup_root(self) -> Path:
        """Absolute path of the directory backups are written under."""
        return self._backup_root

    def backup_path_for(self, target: Path) -> Path:
        """Where `target`'s backup lives. Pure computation - creates nothing.

        Exposed so a caller (and a future restore) can locate a backup without
        creating one, and so the mapping is testable on its own.
        """
        resolved = Path(target).resolve()
        try:
            relative = resolved.relative_to(self._root)
        except ValueError as exc:
            raise BackupError(
                f"cannot back up '{resolved}': it is outside the backup root '{self._root}'"
            ) from exc
        return self._backup_root / relative

    def backup(self, target: Path) -> BackupRecord:
        """Copy `target` into the backup tree, unless a backup already exists.

        Returns a record either way. Raises BackupError if the target cannot be
        read or the copy cannot be written - a failure here must stop the caller
        before it writes, since an unbacked-up write is the one outcome this
        layer exists to prevent.
        """
        resolved = Path(target).resolve()
        destination = self.backup_path_for(resolved)

        if destination.exists():
            return BackupRecord(
                original_path=resolved,
                backup_path=destination,
                created_at=_modified_at(destination),
                already_existed=True,
            )

        if not resolved.is_file():
            raise BackupError(f"cannot back up '{resolved}': not a regular file")

        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            created = self._write_copy(resolved, destination)
        except OSError as exc:
            raise BackupError(f"could not back up '{resolved}' to '{destination}': {exc}") from exc

        return BackupRecord(
            original_path=resolved,
            backup_path=destination,
            created_at=created,
            already_existed=False,
        )

    def _write_copy(self, source: Path, destination: Path) -> datetime:
        """Write a complete temporary copy, then link it into place.

        os.link() fails with FileExistsError if the destination appeared in the
        meantime, which is what makes "never overwrite" a guarantee rather than a
        check that could be raced. The temporary file is removed either way, so a
        failed link leaves no partial backup behind.
        """
        handle = None
        temporary = None
        try:
            descriptor, temporary = tempfile.mkstemp(
                dir=str(destination.parent), prefix=f".{destination.name}.", suffix=".partial"
            )
            handle = os.fdopen(descriptor, "wb")
            handle.write(source.read_bytes())
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
            handle = None

            os.chmod(temporary, os.stat(source).st_mode & 0o7777)
            try:
                os.link(temporary, destination)
            except FileExistsError:
                # Lost the race; the existing backup is the one that counts.
                return _modified_at(destination)
            return _modified_at(destination)
        finally:
            if handle is not None:
                handle.close()
            if temporary is not None and os.path.exists(temporary):
                os.unlink(temporary)


def _modified_at(path: Path) -> datetime:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
