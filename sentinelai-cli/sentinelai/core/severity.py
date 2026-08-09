"""
Centralized severity ordering and severity-based policy decisions.

SEVERITY_RANK is the single source of truth for "how severe is X
relative to Y" - used by CLI severity filtering (--severity), CI gating
(--fail-on), and anywhere else a numeric comparison between Severity
values is needed. Everything else (e.g. presentation styling, fixed
low->high enumeration order in a report template) is free to reference
Severity directly, but any *ranking/comparison* logic should import
SEVERITY_RANK from here rather than redefining it, so the ordering can
never drift out of sync between the CLI, presentation, and reporting
layers.

filter_by_severity() and exceeds_fail_on_threshold() live here rather
than in main.py because they're pure severity-comparison policy with no
CLI/Typer/console dependency - the CLI only orchestrates calling them.
Keeping them here also makes them independently testable and reusable
(e.g. by a future `report --fail-on`) without importing sentinelai.main.
"""
from ..contracts import ScanResult, Severity
from ..statistics import ScanStatistics

SEVERITY_RANK = {
    Severity.LOW: 0,
    Severity.MEDIUM: 1,
    Severity.HIGH: 2,
    Severity.CRITICAL: 3,
}


def filter_by_severity(result: ScanResult, min_severity: Severity) -> ScanResult:
    """Return a copy of `result` containing only findings at or above `min_severity`."""
    threshold = SEVERITY_RANK[min_severity]
    return result.model_copy(
        update={
            "scanner_findings": [f for f in result.scanner_findings if SEVERITY_RANK[f.severity] >= threshold],
            "ai_findings": [f for f in result.ai_findings if SEVERITY_RANK[f.severity] >= threshold],
        }
    )


def exceeds_fail_on_threshold(stats: ScanStatistics, fail_on: str) -> bool:
    """
    True if any finding at or above the `fail_on` severity is present.

    `fail_on` is one of "low", "medium", "high", "critical", "none".
    --fail-on is independent of --severity: --severity controls what's
    included/displayed; --fail-on only decides whether the CLI should
    report a security-policy failure for whatever result is being
    reported (i.e. after any --severity filtering has already been
    applied), so a CI pipeline gates on exactly what it can also see in
    the output.
    """
    if fail_on == "none":
        return False
    threshold = SEVERITY_RANK[Severity(fail_on)]
    counts_by_rank = {
        SEVERITY_RANK[Severity.CRITICAL]: stats.critical_findings,
        SEVERITY_RANK[Severity.HIGH]: stats.high_findings,
        SEVERITY_RANK[Severity.MEDIUM]: stats.medium_findings,
        SEVERITY_RANK[Severity.LOW]: stats.low_findings,
    }
    return any(count > 0 for rank, count in counts_by_rank.items() if rank >= threshold)
