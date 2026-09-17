"""Deterministic patch validation and application. No LLM, no subprocess, no network."""
from .applicator import PatchApplicationResult, PatchApplicator, PatchStrategy
from .backup import BACKUP_DIRECTORY, BackupManager, BackupRecord
from .exceptions import (
    BackupError,
    PatchApplicationError,
    PatchVerificationError,
    RollbackError,
    RollbackFailedError,
    RollbackNotFoundError,
    PatchError,
    PatchTargetNotFoundError,
    PatchValidationError,
    PatchWriteError,
    ReplacementMismatchError,
)
from .models import PatchIssue, PatchIssueCode, PatchValidation
from .rollback import RollbackManager, RollbackRecord
from .runner import PatchAttempt, PatchOutcome, PatchRunResult, run_patches
from .safety import RepositoryState, RepositoryStatus, inspect_repository
from .validator import (
    validate_python_replacement,
    validate_structured_patch,
    validate_unified_diff,
)

__all__ = [
    # validation (Phase 1.3)
    "PatchIssue",
    "PatchIssueCode",
    "PatchValidation",
    "validate_python_replacement",
    "validate_structured_patch",
    "validate_unified_diff",
    # application (Phase 2.1)
    "PatchApplicationResult",
    "PatchApplicator",
    "PatchStrategy",
    # backup and safety (Phase 2.2)
    "BACKUP_DIRECTORY",
    "BackupManager",
    "BackupRecord",
    "RepositoryState",
    "RepositoryStatus",
    "inspect_repository",
    # rollback (Phase 2.3)
    "RollbackManager",
    "RollbackRecord",
    # CLI-facing orchestration (Phase 3.1)
    "PatchAttempt",
    "PatchOutcome",
    "PatchRunResult",
    "run_patches",
    # errors
    "PatchError",
    "BackupError",
    "PatchValidationError",
    "PatchTargetNotFoundError",
    "PatchApplicationError",
    "ReplacementMismatchError",
    "PatchWriteError",
    "PatchVerificationError",
    "RollbackError",
    "RollbackNotFoundError",
    "RollbackFailedError",
]
