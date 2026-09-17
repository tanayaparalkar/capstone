"""
The bridge between the deterministic patch pipeline and the reporting layer.

Why PatchRunResult remains the source of truth
---------------------------------------------
A patch run's facts are the attempts it produced: which finding, which file,
what became of it, whether the file was restored. Everything a report shows is
recomputed from that tuple, and nothing is stored twice. This matters because
the alternative - having the runner hand over pre-counted figures - creates two
records of one event that can drift: a bug fixed in the counting would then have
to be fixed in the runner too, and a report could disagree with the run it
describes.

The division is by responsibility, not convenience. PatchRunResult keeps the
partitions an OPERATOR needs while patches are being applied - did anything
apply, did anything fail, was anything rolled back - because those questions
have answers whether or not a report is ever produced. It deliberately does not
carry presentation figures such as "how many failed validation", because those
exist only so a report can print them; PatchApplicationSummary.from_run() counts
those from `attempts` instead. Without that line, PatchRunResult would grow a
property every time a renderer wanted a new number.

Why reporting never consumes PatchApplicator
--------------------------------------------
By the time anything here runs, every decision has been made and every file has
already been written or left alone. Reporting is meant to describe a completed
run rather than cause one. Holding a PatchApplicator would hand a renderer the
ability to write into a repository - a capability this project keeps confined to
a single component - and would make generating a report twice something other
than a read-only operation.

So the dependency runs in one direction only: reporting -> patching -> contracts.
A renderer receives a frozen result object and only reads from it. patching/ has
no knowledge of reports, which is what lets the patch pipeline be tested with no
renderer loaded and renderers be tested with no repository on disk. A test
asserts that no renderer imports or names PatchApplicator.

What lives here
---------------
build_patch_application_report() decides only whether there is a patch stage to
describe at all; the projection itself belongs to the report models, which own
the shape they declare - see PatchApplicationReport.from_run(). The row helpers
below exist so the Markdown, HTML and terminal renderers present the same fields
in the same order without three copies of that ordering.
"""
from typing import List, Optional, Tuple

from sentinelai.patching import PatchRunResult

from .models import PatchApplicationReport, PatchAttemptReport


def build_patch_application_report(run: Optional[PatchRunResult]) -> Optional[PatchApplicationReport]:
    """Project a PatchRunResult into its reportable form, or None when patching was not requested.

    None in, None out: a scan run without --apply-patches has no patch stage to
    describe, and inventing an empty one would make "not requested" and "nothing
    was eligible" indistinguishable.
    """
    if run is None:
        return None

    return PatchApplicationReport.from_run(run)


# Label and attribute name for each summary count, in the order every renderer
# shows them: outcomes first, worst last, so a reader scanning down the list
# reaches "rollback failed" - the only entry meaning a file may still be
# modified - at the end rather than buried in the middle.
_SUMMARY_FIELDS: Tuple[Tuple[str, str], ...] = (
    ("Total findings", "total_findings"),
    ("Carrying a structured patch", "with_structured_patch"),
    ("Attempted", "attempted"),
    ("Applied", "applied"),
    ("Skipped (no patch)", "skipped"),
    ("Validation failed", "validation_failed"),
    ("Not applicable", "not_applicable"),
    ("Rolled back", "rolled_back"),
    ("Rollback failed", "rollback_failed"),
    ("Unexpected errors", "errors"),
)

ATTEMPT_COLUMNS: Tuple[str, ...] = ("Finding", "File", "Outcome", "Rolled back", "Detail")


def summary_rows(report: PatchApplicationReport) -> List[Tuple[str, int]]:
    """(label, count) pairs in the shared order. One definition, four renderers."""
    return [(label, getattr(report.summary, attribute)) for label, attribute in _SUMMARY_FIELDS]


def format_rollback_status(attempt: PatchAttemptReport) -> str:
    """Render one attempt's rollback state as "no", "yes" or "failed".

    Two booleans collapse into one column, and the order of the checks is what
    makes that lossless: rollback_failed implies a rollback was attempted, so
    testing it first reports the outcome that matters. Reading rollback_executed
    first would show "yes" for a restoration that did not succeed - the one case
    where a file may still hold modified content.

    Takes a PatchAttemptReport rather than a patching.PatchAttempt: this is the
    reporting layer, and what reaches it has already been projected.
    """
    if attempt.rollback_failed:
        return "failed"
    if attempt.rollback_executed:
        return "yes"
    return "no"


def attempt_rows(report: PatchApplicationReport) -> List[Tuple[str, str, str, str, str]]:
    """One display row per attempt, columns matching ATTEMPT_COLUMNS."""
    return [
        (
            attempt.finding_id,
            attempt.file or "-",
            attempt.outcome,
            format_rollback_status(attempt),
            attempt.detail or "-",
        )
        for attempt in report.attempts
    ]
