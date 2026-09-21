"""
Deterministic failure modes of patch application.

Every class here subclasses core.errors.SentinelAIError, the hierarchy the CLI
already knows how to render and map to an exit code. No parallel hierarchy is
introduced: a future `sentinelai fix` command should be able to catch
SentinelAIError and behave sensibly without knowing this module exists.

PatchError is the single base a caller can catch to mean "applying a patch
failed" without enumerating reasons. The five subclasses exist because the
reasons call for different responses: a validation failure means the generated
patch was malformed, a missing target means the repository moved under the
scan, and a context mismatch means the file changed since the evidence was
gathered. Collapsing them would make a caller parse English to tell those
apart.

None of these indicate partial writes. The applicator either replaces a file
completely or leaves it untouched, so an exception raised from this module
always means the target is in its original state.
"""
from sentinelai.core.errors import SentinelAIError


class PatchError(SentinelAIError):
    """Base for every patch-application failure. Catch this to mean 'the patch was not applied'.

    `rollback` is None unless an automatic rollback ran while this error was
    being raised, in which case it carries the RollbackRecord describing what was
    restored. Declared here as a plain class attribute rather than set
    dynamically, so a caller can read `error.rollback` unconditionally, and so
    exceptions.py needs no import of rollback.py (which imports this module).
    """

    rollback = None


class PatchValidationError(PatchError):
    """The patch did not pass deterministic validation, so application was refused.

    Raised before the target file is read. The applicator does not re-derive
    validity - it consumes patching/validator.py's verdict and declines to act
    on a rejected one.
    """


class PatchTargetNotFoundError(PatchError):
    """The file the patch targets does not exist, or is not a regular file."""


class PatchApplicationError(PatchError):
    """The patch is well formed but does not fit the target file.

    The usual cause is that a context or removed line in the diff does not match
    the file at the line the hunk header points to - the file changed after the
    evidence was gathered, or the hunk's line numbers are wrong. Nothing is
    written and nothing is guessed at; a diff that does not fit is refused
    rather than applied approximately.
    """


class ReplacementMismatchError(PatchError):
    """A replacement block cannot be located in the target file.

    Distinct from PatchApplicationError because the failure is about the span
    rather than the content: the patch supplied no line range, or a range that
    falls outside the file.
    """


class BackupError(PatchError):
    """The original file could not be preserved before writing.

    Raised before any modification. A backup failure must stop the caller rather
    than be tolerated: writing without a preserved original is the single
    outcome the backup layer exists to prevent, so proceeding would trade a
    recoverable change for an unrecoverable one.
    """


class PatchVerificationError(PatchError):
    """The file on disk does not match what was written.

    Raised after a write that reported success but whose read-back differs from
    the intended contents. Distinct from PatchWriteError because the write call
    did NOT fail - the discrepancy appeared afterwards, which is the case
    automatic rollback exists for: a failed write leaves the target untouched
    thanks to the atomic rename, whereas a failed verification means the target
    may hold something nobody intended.
    """


class RollbackError(PatchError):
    """Base for restoration failures. Catch this to mean 'the file was not restored'."""


class RollbackNotFoundError(RollbackError):
    """No backup exists for the requested file, so there is nothing to restore from.

    Not a failure of restoration but of its precondition. Kept separate because a
    caller can act on it - a file with no backup was never patched by SentinelAI,
    which is information rather than an error condition to retry.
    """


class RollbackFailedError(RollbackError):
    """A backup was found but the file could not be restored from it.

    The most serious error in this package: it means a patch failed AND the
    original could not be put back, so the target may hold modified content. The
    backup itself is never modified during restoration, so the preserved copy
    remains available for a manual retry.
    """


class PatchWriteError(PatchError):
    """The updated contents could not be written.

    Raised only from the atomic write step. Because the write completes by
    renaming a fully-written temporary file over the target, this failing means
    the target still holds its original contents.
    """
