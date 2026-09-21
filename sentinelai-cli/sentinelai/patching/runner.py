"""
Per-finding patch application, and the record of what happened.

Orchestration only. This module calls PatchApplicator once per finding and
writes down the outcome; it does not validate, read, write, back up or restore
anything itself. PatchApplicator remains the only component that writes into a
repository and RollbackManager the only one that restores, reached here only
through the applicator's own automatic-rollback path.

Best effort, per finding. A patch that cannot be applied is a fact about that
finding, not about the scan, so every failure is caught, classified and
recorded, and the loop moves on. Nothing here raises, and nothing here can
change a scan's exit code - a repository that produced twenty findings and
applied none of them still completes its scan normally.

Deterministic. Findings are processed in the order given and results come back
in that same order, so two runs over the same input produce the same report with
the same entries in the same positions.

No model call. This is downstream of generation: the patches it applies were
proposed by a model earlier in the pipeline, and every step from here is
ordinary code.
"""
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from sentinelai.contracts import AIEnrichedFinding

from .applicator import PatchApplicator
from .exceptions import (
    PatchApplicationError,
    PatchError,
    PatchTargetNotFoundError,
    PatchValidationError,
    PatchVerificationError,
    PatchWriteError,
    ReplacementMismatchError,
    RollbackFailedError,
)


class PatchOutcome(str, Enum):
    """What happened to one finding's patch. Stable identifiers - callers may branch on these."""

    APPLIED = "applied"
    SKIPPED_NO_PATCH = "skipped_no_patch"
    VALIDATION_FAILED = "validation_failed"
    NOT_APPLICABLE = "not_applicable"
    ROLLED_BACK = "rolled_back"
    ROLLBACK_FAILED = "rollback_failed"
    ERROR = "error"


@dataclass(frozen=True)
class PatchAttempt:
    """One finding's outcome.

    `rollback_executed` and `rollback_failed` are carried as separate booleans
    rather than folded into `outcome` because they answer a different question.
    The outcome says what became of the patch; these say what became of the file.
    A reader needs both to know whether a repository was left modified.
    """

    finding_id: str
    file: Optional[str]
    outcome: PatchOutcome
    detail: str = ""
    rollback_executed: bool = False
    rollback_failed: bool = False

    @property
    def succeeded(self) -> bool:
        return self.outcome is PatchOutcome.APPLIED


@dataclass(frozen=True)
class PatchRunResult:
    """Every attempt from one run, in the order the findings were given.

    `attempts` is the source of truth; everything else here is derived from it.

    The partitions below are the operational questions a caller running patches
    asks - did anything apply, did anything fail, was anything rolled back - and
    they exist because an operator needs them, not because a report does.
    Presentation statistics are deliberately NOT here: reporting derives its own
    counts from `attempts` in one place (PatchApplicationSummary.from_run), so
    this type does not grow a property every time a report wants a new figure.
    """

    attempts: Tuple[PatchAttempt, ...] = field(default_factory=tuple)
    dry_run: bool = False

    @property
    def applied(self) -> Tuple[PatchAttempt, ...]:
        return tuple(a for a in self.attempts if a.outcome is PatchOutcome.APPLIED)

    @property
    def skipped(self) -> Tuple[PatchAttempt, ...]:
        return tuple(a for a in self.attempts if a.outcome is PatchOutcome.SKIPPED_NO_PATCH)

    @property
    def failed(self) -> Tuple[PatchAttempt, ...]:
        return tuple(
            a
            for a in self.attempts
            if a.outcome not in (PatchOutcome.APPLIED, PatchOutcome.SKIPPED_NO_PATCH)
        )

    @property
    def rolled_back(self) -> Tuple[PatchAttempt, ...]:
        return tuple(a for a in self.attempts if a.rollback_executed)

    @property
    def attempted(self) -> int:
        """Findings that carried a patch, so were actually tried."""
        return len(self.attempts) - len(self.skipped)


def run_patches(
    findings: Sequence[AIEnrichedFinding],
    root: Path,
    applicator: Optional[PatchApplicator] = None,
    dry_run: bool = False,
) -> PatchRunResult:
    """Attempt every finding's structured patch, recording each outcome.

    `applicator` is injectable for testing and for a caller that has configured
    a custom backup location; by default one is constructed bounded to `root`,
    which is what gives both the write-containment check and the backup layout.
    """
    engine = applicator if applicator is not None else PatchApplicator(root=Path(root))
    attempts: List[PatchAttempt] = []

    for finding in findings:
        patch = finding.structured_patch
        if patch is None:
            attempts.append(
                PatchAttempt(
                    finding_id=finding.finding_id,
                    file=None,
                    outcome=PatchOutcome.SKIPPED_NO_PATCH,
                    detail="finding carries no structured patch",
                )
            )
            continue

        attempts.append(_attempt(engine, finding.finding_id, patch, dry_run))

    return PatchRunResult(attempts=tuple(attempts), dry_run=dry_run)


def _attempt(engine: PatchApplicator, finding_id: str, patch, dry_run: bool = False) -> PatchAttempt:
    try:
        engine.apply_with_result(patch, dry_run=dry_run)
    except PatchError as exc:
        return _classify(finding_id, patch.file, exc)
    except Exception as exc:  # noqa: BLE001 - see below
        # A patch failure must never abort a scan, so an unexpected error from
        # anywhere beneath is recorded rather than propagated. Caught broadly for
        # the same reason main.py catches broadly around AI enrichment: the
        # alternative is a traceback that loses every other finding's result.
        return PatchAttempt(
            finding_id=finding_id,
            file=patch.file,
            outcome=PatchOutcome.ERROR,
            detail=f"{type(exc).__name__}: {exc}",
        )

    # APPLIED regardless of mode. Execution mode is an orthogonal axis carried by
    # PatchRunResult.dry_run, not a second meaning folded into this enum, which
    # enumerates what the apply step did rather than whether it was written.
    # Renderers surface the mode alongside the table, so an "applied" row is never
    # shown without it.
    return PatchAttempt(finding_id=finding_id, file=patch.file, outcome=PatchOutcome.APPLIED)


def _classify(finding_id: str, file: str, exc: PatchError) -> PatchAttempt:
    """Map a patch failure onto an outcome.

    Ordered most-severe first. RollbackFailedError is checked before anything
    else because it is the only case where the repository may still hold modified
    content, and it must not be absorbed into a milder classification.
    """
    detail = f"{type(exc).__name__}: {exc}"
    rollback_executed = getattr(exc, "rollback", None) is not None

    if isinstance(exc, RollbackFailedError):
        return PatchAttempt(
            finding_id=finding_id,
            file=file,
            outcome=PatchOutcome.ROLLBACK_FAILED,
            detail=detail,
            rollback_executed=True,
            rollback_failed=True,
        )

    if isinstance(exc, PatchValidationError):
        return PatchAttempt(finding_id, file, PatchOutcome.VALIDATION_FAILED, detail)

    if isinstance(exc, (PatchWriteError, PatchVerificationError)):
        return PatchAttempt(
            finding_id=finding_id,
            file=file,
            outcome=PatchOutcome.ROLLED_BACK if rollback_executed else PatchOutcome.ERROR,
            detail=detail,
            rollback_executed=rollback_executed,
        )

    if isinstance(
        exc, (PatchApplicationError, ReplacementMismatchError, PatchTargetNotFoundError)
    ):
        return PatchAttempt(finding_id, file, PatchOutcome.NOT_APPLICABLE, detail)

    return PatchAttempt(finding_id, file, PatchOutcome.ERROR, detail)
