"""
Canonical JSON report envelope.

Wraps the existing ScanResult data (repository/scan metadata + both
finding layers) and the existing ScanStatistics into one stable,
versioned structure - this is a composition of existing contracts, not a
second definition of finding fields. ScannerFinding, AIEnrichedFinding,
RepositoryInfo, and ScanMetadata remain the only source of truth for
their own fields; this model only arranges them for the report.

schema_version is a plain string (not tied to the package version) so the
report format can be versioned independently of sentinelai's own release
number - a report-shape change bumps this, a CLI bugfix release doesn't.

`findings.correlated` was added without bumping schema_version, which is
deliberate and not an oversight. reporting/loader.py rejects any report
whose schema_version is not exactly REPORT_SCHEMA_VERSION, so bumping to
"1.1" would make every previously-saved "1.0" report unloadable - it
would break backward compatibility rather than express it. The addition
is compatible in both directions without a bump: a new reader supplies
the default [] for an old report that has no `correlated` key, and an
older reader ignores the extra key (this model does not set
extra="forbid"). A bump is warranted when a change would actually make
an old report unreadable - renaming or removing a field, or changing a
field's meaning - which this does not.

`patch_application` was added the same way and for the same reasons.

Bumping to "1.1" would not have expressed "this report may contain patch
results". It would have made every previously saved "1.0" report fail to load,
because loader.py compares the version with strict equality rather than a range.
A version bump is a statement that old readers CANNOT understand new reports and
new readers CANNOT understand old ones; neither is true here.

The addition is compatible without one. A reader handed a report written before
the field existed supplies its default of None, which every renderer treats as
"patch application was not requested" - indistinguishable from a scan run
without --apply-patches, which is exactly right, because that is what such a
report describes. The guarantee that matters is this one: an optional field with
a default adds information to a report without changing how any field already in
it is read, so the meaning of a "1.0" report is the same before and after.

The bar for bumping stays where it was: a change that makes an old report
genuinely unreadable - renaming a field, removing one, or changing what an
existing field means. Adding an optional field with a default does none of
those. A bump here would have broken backwards compatibility in the name of
declaring it.
"""
from typing import TYPE_CHECKING, Optional

from pydantic import BaseModel, Field

from ..contracts import (
    AIEnrichedFinding,
    CorrelatedFinding,
    RepositoryInfo,
    ScanMetadata,
    ScannerFinding,
)
from ..statistics import ScanStatistics

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..patching import PatchAttempt, PatchRunResult

REPORT_SCHEMA_VERSION = "1.0"


class ReportFindings(BaseModel):
    scanner: list[ScannerFinding] = Field(default_factory=list)
    ai_enriched: list[AIEnrichedFinding] = Field(default_factory=list)
    correlated: list[CorrelatedFinding] = Field(
        default_factory=list,
        description=(
            "Cross-scanner correlation groups. An index over `scanner`, never a substitute for "
            "it: every raw finding above is still present and unmodified. Defaults to [] so a "
            "report written before correlation existed still loads."
        ),
    )


# Summary field to count, keyed by the PatchOutcome value each attempt carries.
# Compared by VALUE rather than by importing PatchOutcome, so this module keeps no
# runtime dependency on patching/ - which transitively loads a git client that
# report rendering has no business pulling in. These strings are already part of
# the report contract (PatchAttemptReport.outcome), so coupling to them couples to
# something public, and the parametrized outcome tests fail loudly if one is
# renamed. `rolled_back` and `rollback_failed` are absent by design: they come
# from the per-attempt booleans, because a rollback can accompany more than one
# outcome.
_SUMMARY_FIELD_BY_OUTCOME = {
    "applied": "applied",
    "skipped_no_patch": "skipped",
    "validation_failed": "validation_failed",
    "not_applicable": "not_applicable",
    "error": "errors",
}


class PatchAttemptReport(BaseModel):
    """One finding's patch outcome, as it appears in a report.

    A faithful projection of patching.PatchAttempt, not a reinterpretation:
    `outcome` carries PatchOutcome's own string value, so a consumer branching on
    it is branching on the same identifiers the deterministic pipeline produced.
    """

    finding_id: str
    file: Optional[str] = Field(None, description="Target file, or null for a finding carrying no patch")
    outcome: str = Field(..., description="PatchOutcome value, e.g. 'applied', 'validation_failed'")
    rollback_executed: bool = False
    rollback_failed: bool = False
    detail: str = ""

    @classmethod
    def from_attempt(cls, attempt: "PatchAttempt") -> "PatchAttemptReport":
        """Project one PatchAttempt. The only place that mapping is written."""
        return cls(
            finding_id=attempt.finding_id,
            file=attempt.file,
            outcome=attempt.outcome.value,
            rollback_executed=attempt.rollback_executed,
            rollback_failed=attempt.rollback_failed,
            detail=attempt.detail,
        )


class PatchApplicationSummary(BaseModel):
    """Counts over one patch run, derived in exactly one place.

    Every figure here is computed by from_run() from the run's `attempts`, and
    nowhere else. No renderer counts anything for itself, which is what makes it
    impossible for the JSON and Markdown views of one scan to disagree about how
    many patches applied - the same reason statistics/ exists as a single
    calculator rather than as arithmetic repeated per format.

    These counts live on the report model rather than on PatchRunResult because
    they are presentation figures: they exist so a report can show them, not
    because applying patches requires them. PatchRunResult keeps only what an
    operator needs during a run. Drawing the line here means a renderer wanting a
    new number adds a field to this model instead of growing the domain type.
    """

    total_findings: int = 0
    with_structured_patch: int = 0
    attempted: int = 0
    applied: int = 0
    skipped: int = 0
    validation_failed: int = 0
    not_applicable: int = 0
    rolled_back: int = 0
    rollback_failed: int = 0
    errors: int = 0

    @classmethod
    def from_run(cls, run: "PatchRunResult") -> "PatchApplicationSummary":
        """Derive every count from one PatchRunResult's attempts.

        Lives on the model rather than on PatchRunResult because these are
        presentation figures: they exist so a report can show them, not because
        patch execution needs them. PatchRunResult stays the source of truth
        through `attempts`, and this is the single place those attempts are
        counted - no renderer counts anything for itself.
        """
        counts = {field: 0 for field in _SUMMARY_FIELD_BY_OUTCOME.values()}
        rolled_back = 0
        rollback_failed = 0

        # One pass over attempts, which is the run's source of truth. Counting
        # here rather than on PatchRunResult keeps presentation figures out of the
        # domain type, and doing it once keeps every renderer agreeing.
        for attempt in run.attempts:
            field = _SUMMARY_FIELD_BY_OUTCOME.get(attempt.outcome.value)
            if field is not None:
                counts[field] += 1
            if attempt.rollback_executed:
                rolled_back += 1
            if attempt.rollback_failed:
                rollback_failed += 1

        return cls(
            total_findings=len(run.attempts),
            with_structured_patch=run.attempted,
            attempted=run.attempted,
            rolled_back=rolled_back,
            rollback_failed=rollback_failed,
            **counts,
        )


class PatchApplicationReport(BaseModel):
    """The deterministic patch stage, as a serializable projection of one run.

    Why this exists rather than serializing PatchRunResult directly
    --------------------------------------------------------------
    PatchRunResult is a frozen dataclass built for the pipeline: it holds enum
    members, derives its partitions on access, and is shaped around the questions
    an operator asks mid-run. None of that survives a file. This model is the
    same information in a form that can be written to JSON, read back a week
    later by a different version of the tool, and validated on the way in -
    enums flattened to their string values, counts materialised rather than
    recomputed, every field declared and defaulted.

    It also decouples the two lifetimes. `sentinelai report` re-renders a SAVED
    report and never has a PatchRunResult at all; there is no run in progress,
    only a file. Both paths converge here, which is what lets a saved report
    render to HTML exactly as the original scan would have. Serializing the
    domain type directly would have tied the on-disk format to the internal one,
    so every refactor of the runner would have become a report-format change.

    PatchRunResult stays the source of truth: this is derived from it in exactly
    one place (from_run below) and is never written to by anything else.

    Why patch_application is a sibling of `findings`, not nested inside it
    ---------------------------------------------------------------------
    A finding is something a scanner or a model CLAIMED about the code. A patch
    outcome is something SentinelAI DID to the code. They are different kinds of
    assertion, produced by different halves of the system, and the whole contract
    design keeps such things adjacent rather than merged - the same reason
    ScannerFinding and AIEnrichedFinding are separate objects joined by identity
    instead of one merged record.

    Nesting would have caused three concrete problems. A reader could no longer
    tell whether a field came from analysis or from execution. Findings that
    carry no patch would need a null patch field, and findings skipped for having
    no patch would have nowhere to be recorded at all - yet they are exactly what
    the summary's `skipped` count describes. And the patch stage's own summary,
    which is about the run rather than about any one finding, would have no
    natural home.

    The same separation holds in every format: Markdown and HTML give it its own
    section, and SARIF puts it under the run's property bag while adding nothing
    to `results`, because a patch outcome is not a finding and must not be
    counted as one.
    """

    summary: PatchApplicationSummary
    dry_run: bool = Field(
        False,
        description=(
            "True when this run computed patches without writing them. Carried on the report "
            "so a saved artifact states for itself whether any file was modified, rather than "
            "leaving a reader to infer it from the outcome values."
        ),
    )
    attempts: list[PatchAttemptReport] = Field(
        default_factory=list, description="One entry per finding considered, in the order processed"
    )

    @classmethod
    def from_run(cls, run: "PatchRunResult") -> "PatchApplicationReport":
        """Project a whole run. Attempts keep their processing order.

        Not sorted: that order is itself deterministic and is what makes two
        reports of the same scan diffable line by line.
        """
        return cls(
            summary=PatchApplicationSummary.from_run(run),
            dry_run=run.dry_run,
            attempts=[PatchAttemptReport.from_attempt(attempt) for attempt in run.attempts],
        )


class JSONReport(BaseModel):
    schema_version: str = REPORT_SCHEMA_VERSION
    repository: RepositoryInfo
    scan: ScanMetadata
    statistics: ScanStatistics
    findings: ReportFindings
    patch_application: Optional[PatchApplicationReport] = Field(
        None,
        description=(
            "Outcome of --apply-patches, or null when patch application was not requested. "
            "Optional and defaulted for the same reason `findings.correlated` was added without "
            "a schema bump: a reader supplies the default for an older report, and an older "
            "reader ignores the key."
        ),
    )
