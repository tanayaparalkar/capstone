"""
The atomic file-replacement primitive, owned in one place.

Package-internal (leading underscore, not exported from __init__.py). It exists
because applicator.py and rollback.py previously carried two copies of the same
seven-step sequence, and every step in it is a correctness requirement rather
than a style choice:

- the temporary file is created in the TARGET's own directory, because
  os.replace() is only atomic within a single filesystem and a temp directory is
  frequently a different one;
- contents are flushed and fsync'd BEFORE the rename, because a rename can
  otherwise outlive the data it points at across a crash;
- permissions are applied to the temporary file before the rename, since
  mkstemp() creates 0600 and renaming without this would silently tighten the
  target;
- os.replace() is the only step a reader can observe, and it is atomic, so the
  target is never seen partially written;
- the temporary file is removed on every exit path, so a failure leaves no
  debris beside the file it failed to replace.

Two copies meant two places for any of those to be wrong, and a fix to one
leaving the other broken. One copy makes the guarantee a property of this
function rather than a coincidence of two code paths agreeing.

Deliberately NOT shared with backup.py. That module's final step is os.link()
rather than os.replace(), because its guarantee is the opposite one - never
overwrite an existing file. Folding it in here would mean a flag selecting
between "replace" and "refuse to replace", which is two primitives wearing one
name.

This function owns mechanism only. It takes the bytes to write and the mode to
apply, and raises OSError. Deciding WHICH mode is right (the target's own, for a
patch; the backup's, for a restore) and which domain exception an OSError means
(PatchWriteError or RollbackFailedError) are policy, and stay with the callers
that have the context to answer them.
"""
import os
import tempfile
from pathlib import Path
from typing import Optional


def atomic_replace(target: Path, data: bytes, mode: int, suffix: str) -> None:
    """Replace `target`'s contents with `data`, atomically, applying `mode`.

    `suffix` names the temporary file so that debris surviving a hard crash is
    attributable to the operation that created it; callers pass distinct values.

    Raises OSError, which callers translate into their own domain exception.
    Whatever the failure, the target is left either wholly original or wholly
    replaced, and no temporary file remains.
    """
    handle = None
    temporary: Optional[str] = None
    try:
        descriptor, temporary = tempfile.mkstemp(
            dir=str(target.parent), prefix=f".{target.name}.", suffix=suffix
        )
        handle = os.fdopen(descriptor, "wb")
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        handle = None

        os.chmod(temporary, mode)
        os.replace(temporary, target)
        temporary = None
    finally:
        if handle is not None:
            handle.close()
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)
