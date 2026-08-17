"""
Cross-scanner correlation contract.

A CorrelatedFinding is a group of raw ScannerFindings that three
independent tools appear to be reporting about the same underlying
issue. It is an *index over* the raw findings, never a replacement for
them: ScanResult.scanner_findings still carries every finding exactly as
its scanner emitted it, and every raw finding belongs to exactly one
group. Correlation adds a view; it never discards evidence.

Grouping is deterministic and conservative. Two findings correlate only
when all of the following hold:

    1. same file (compared after path normalization);
    2. both carry a CWE, extracted to its bare `CWE-<digits>` form;
    3. those normalized CWEs are equal;
    4. their line ranges overlap, or are within _LINE_TOLERANCE lines.

Nothing else merges. Matching severity, matching scanner name, or a
matching generic category (Semgrep reports `security` for almost every
rule; Bandit reports `blacklist` for four unrelated checks) is explicitly
not evidence that two findings describe one issue - a file can hold two
different medium-severity problems, and frequently does.

A finding with no CWE, or no line information, is always a singleton.
That is why a GitLeaks hit and a Semgrep CWE-798 hit on the same
hardcoded credential stay separate today: GitLeaks emits no CWE at all,
so rule 2 fails. Secret-family correlation, which would need its own
matching policy rather than a CWE, is deliberately left to future work
rather than bolted on as an exception here.

The line tolerance is a documented heuristic, not proof that two
findings describe the same defect. It exists because scanners disagree
about which line to blame: for one SQL injection in the benchmark
repository, Bandit reports the line building the query string while
Semgrep reports the following line that executes it. The tolerance
captures that class of disagreement and nothing wider - see
sentinelai/correlation/correlator.py for the measured safety margin
against this corpus.
"""
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from .common import Severity


class CorrelationRule(str, Enum):
    """Which rule produced a group. Recorded so a merge is always explainable."""

    SAME_CWE_SAME_LINE = "same_cwe_same_line"
    SAME_CWE_ADJACENT_LINES = "same_cwe_adjacent_lines"
    SINGLETON = "singleton"


class CorrelatedFinding(BaseModel):
    """One logical issue, backed by one or more raw ScannerFindings."""

    correlation_id: str = Field(
        ...,
        description=(
            "Stable identifier for this group, e.g. 'CORR-001'. Deterministic across input "
            "orderings: groups are sorted by (file, line_start, cwe, canonical_finding_id) "
            "before numbering, so the same scan always yields the same ids regardless of the "
            "order scanners happened to run in."
        ),
    )
    canonical_finding_id: str = Field(
        ...,
        description=(
            "The one raw finding_id chosen to represent this group. Deterministically the "
            "lowest-sorting source id, so it does not depend on scanner order. This is the id "
            "AIEnrichedFinding.finding_id keeps pointing at, which is what lets existing "
            "raw-finding -> AI joins keep working unchanged."
        ),
    )
    source_finding_ids: list[str] = Field(
        ...,
        description=(
            "Every raw ScannerFinding.finding_id in this group, sorted. Always contains at "
            "least the canonical id. Never lossy: the union of these across all groups is "
            "exactly the set of raw findings in the scan."
        ),
    )
    scanners: list[str] = Field(
        ..., description="Every contributing scanner, sorted and de-duplicated."
    )
    file: Optional[str] = Field(None, description="File all sources agree on, if any.")
    line_start: Optional[int] = Field(None, ge=1, description="Earliest line across the sources.")
    line_end: Optional[int] = Field(None, ge=1, description="Latest line across the sources.")
    cwe: Optional[str] = Field(
        None,
        description=(
            "Normalized CWE shared by the group, e.g. 'CWE-89'. Normalized form only - the raw "
            "scanner strings (Semgrep emits 'CWE-89: Improper Neutralization of...') are "
            "preserved untouched on the raw findings themselves."
        ),
    )
    severity: Severity = Field(
        ...,
        description=(
            "Highest severity among the sources, ranked by the project's own SEVERITY_RANK "
            "ordering rather than by string comparison."
        ),
    )
    category: str = Field(
        ..., description="The canonical source finding's category, carried through unchanged."
    )
    rule: CorrelationRule = Field(..., description="Which correlation rule produced this group.")
    correlation_reason: str = Field(
        ...,
        description=(
            "Human-readable justification naming the scanners, the shared CWE, the location, and "
            "whether the match was on the same/overlapping line or on adjacent lines within "
            "tolerance. Rendered in reports so a merge is never opaque."
        ),
    )

    @property
    def is_multi_scanner(self) -> bool:
        """True when more than one distinct scanner contributed - i.e. independent corroboration."""
        return len(self.scanners) > 1
