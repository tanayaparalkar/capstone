"""
Deterministic cross-scanner correlation.

Groups raw ScannerFindings that appear to describe one underlying issue,
without discarding any of them. Pure computation: no I/O, no model call,
no configuration lookup, no randomness. The same findings in any order
produce byte-identical groups.

Why CWE and location, and nothing else
--------------------------------------
CWE is the only cross-scanner vocabulary these three tools actually
share. Categories do not survive the crossing: Semgrep reports
`security` for nearly every rule, and Bandit reports `blacklist` for
four unrelated checks (import pickle, import subprocess, pickle.loads,
eval). Grouping on category would merge an unsafe-deserialization
finding with an eval finding purely because Bandit files both under the
same internal label. Severity is likewise not identity - one file
routinely holds several medium-severity problems.

CWE normalization is not optional. The three scanners emit three shapes
for the same concept:

    bandit   "CWE-327"
    semgrep  "CWE-327: Use of a Broken or Risky Cryptographic Algorithm"
    gitleaks  None

so raw string equality would correlate nothing at all. _normalize_cwe
extracts the bare `CWE-<digits>` form. The raw strings stay untouched on
the raw findings; only CorrelatedFinding carries the normalized value.

The line tolerance
------------------
_LINE_TOLERANCE exists for one measured reason: scanners disagree about
which line to blame. In the benchmark repository a single SQL injection
is reported by Bandit at the line building the query string and by
Semgrep at the next line executing it - the same defect, one line apart.
Requiring identical lines would treat that as two issues.

The value is 2, and the margin was measured rather than guessed. Within
this corpus the nearest same-CWE findings that are genuinely *different*
issues are 7 lines apart (CWE-502: `import pickle` at line 6 vs
`pickle.loads` at line 13) and 10 lines apart (CWE-78: `import
subprocess` at line 7 vs the shell call at line 17). A tolerance of 2
therefore has roughly five lines of headroom before it would merge
something it should not.

This is a documented, configurable heuristic - not semantic proof that
two findings are the same defect. Widening it trades precision for
recall, and anything approaching 7 would start merging distinct issues
in this very corpus.

Transitivity
------------
Grouping is transitive within a run: if A correlates with B and B with
C, all three become one group, even where A and C alone would fall
outside the tolerance. This is union-find over the pairwise relation,
which keeps the result independent of the order findings are compared
in - the property that makes correlation ids stable.
"""
import re
from typing import Dict, List, Optional, Sequence, Tuple

from sentinelai.contracts import CorrelatedFinding, CorrelationRule, ScannerFinding, Severity
from sentinelai.core import SEVERITY_RANK

# Documented heuristic - see the module docstring for the measured margin.
_LINE_TOLERANCE = 2

_CWE_PATTERN = re.compile(r"CWE-(\d+)", re.IGNORECASE)


def _normalize_cwe(raw: Optional[str]) -> Optional[str]:
    """Extract the bare `CWE-<digits>` form, or None when the finding carries no CWE."""
    if not raw:
        return None
    match = _CWE_PATTERN.search(raw)
    return f"CWE-{match.group(1)}" if match else None


def _correlatable(finding: ScannerFinding) -> bool:
    """A finding can only join a group if it has both a file, a line, and a CWE.

    Anything missing one of those is a singleton by construction rather than by
    accident - there is no evidence available to match it on conservatively.
    """
    return bool(finding.file) and finding.line_start is not None and _normalize_cwe(finding.cwe) is not None


def _line_gap(a: ScannerFinding, b: ScannerFinding) -> int:
    """Distance between two line ranges; 0 when they touch or overlap."""
    a_start, a_end = a.line_start, a.line_end or a.line_start
    b_start, b_end = b.line_start, b.line_end or b.line_start
    if a_start <= b_end and b_start <= a_end:
        return 0
    return b_start - a_end if b_start > a_end else a_start - b_end


def _pair_rule(a: ScannerFinding, b: ScannerFinding) -> Optional[CorrelationRule]:
    """Return the rule correlating `a` and `b`, or None if they do not correlate."""
    if a.file != b.file:
        return None
    cwe_a, cwe_b = _normalize_cwe(a.cwe), _normalize_cwe(b.cwe)
    if cwe_a is None or cwe_a != cwe_b:
        return None
    gap = _line_gap(a, b)
    if gap == 0:
        return CorrelationRule.SAME_CWE_SAME_LINE
    if gap <= _LINE_TOLERANCE:
        return CorrelationRule.SAME_CWE_ADJACENT_LINES
    return None


class _Union:
    """Minimal union-find, so grouping is order-independent and transitive."""

    def __init__(self, keys: Sequence[str]) -> None:
        self._parent: Dict[str, str] = {key: key for key in keys}

    def find(self, key: str) -> str:
        while self._parent[key] != key:
            self._parent[key] = self._parent[self._parent[key]]
            key = self._parent[key]
        return key

    def union(self, a: str, b: str) -> None:
        root_a, root_b = self.find(a), self.find(b)
        if root_a != root_b:
            # Lower id always becomes the root, so the structure - and therefore
            # the canonical finding - never depends on the order unions happened.
            low, high = sorted((root_a, root_b))
            self._parent[high] = low


def _highest_severity(findings: Sequence[ScannerFinding]) -> Severity:
    """Highest severity by the project's own ordering, never by string comparison."""
    return max((f.severity for f in findings), key=lambda severity: SEVERITY_RANK[severity])


def _reason(
    rule: CorrelationRule, scanners: List[str], cwe: Optional[str], location: str, findings: Sequence[ScannerFinding]
) -> str:
    if rule is CorrelationRule.SINGLETON:
        return f"Reported only by {scanners[0]} at {location}; no other finding matched."

    joined = " and ".join(scanners) if len(scanners) > 1 else scanners[0]
    if rule is CorrelationRule.SAME_CWE_SAME_LINE:
        return (
            f"{joined} report {cwe} at the same location ({location}); "
            "grouped on matching CWE and overlapping lines."
        )
    lines = sorted({f.line_start for f in findings if f.line_start is not None})
    return (
        f"{joined} report {cwe} on adjacent lines ({location}, lines {lines}); "
        f"grouped on matching CWE within the configured tolerance of {_LINE_TOLERANCE} lines. "
        "Line tolerance is a heuristic for scanners blaming different lines of one statement, "
        "not proof the findings describe the same defect."
    )


def _location(findings: Sequence[ScannerFinding]) -> str:
    first = findings[0]
    if not first.file:
        return "an unspecified location"
    starts = [f.line_start for f in findings if f.line_start is not None]
    return f"{first.file}:{min(starts)}" if starts else first.file


def correlate_findings(scanner_findings: Sequence[ScannerFinding]) -> List[CorrelatedFinding]:
    """Group raw findings into correlated issues. Pure, deterministic, and lossless.

    Every input finding appears in exactly one returned group, so
    `sum(len(g.source_finding_ids)) == len(scanner_findings)` always holds.
    """
    if not scanner_findings:
        return []

    # Sort first so every downstream step - unions, canonical selection, id
    # numbering - is independent of the order scanners ran in.
    ordered = sorted(
        scanner_findings,
        key=lambda f: (f.file or "", f.line_start or 0, _normalize_cwe(f.cwe) or "", f.finding_id),
    )
    by_id = {f.finding_id: f for f in ordered}

    union = _Union([f.finding_id for f in ordered])
    rules: Dict[Tuple[str, str], CorrelationRule] = {}
    for i, a in enumerate(ordered):
        if not _correlatable(a):
            continue
        for b in ordered[i + 1 :]:
            if not _correlatable(b):
                continue
            rule = _pair_rule(a, b)
            if rule is not None:
                union.union(a.finding_id, b.finding_id)
                rules[(a.finding_id, b.finding_id)] = rule

    grouped: Dict[str, List[ScannerFinding]] = {}
    for finding in ordered:
        grouped.setdefault(union.find(finding.finding_id), []).append(finding)

    groups: List[CorrelatedFinding] = []
    for members in grouped.values():
        source_ids = sorted(f.finding_id for f in members)
        canonical_id = source_ids[0]
        canonical = by_id[canonical_id]
        scanners = sorted({f.scanner for f in members})
        starts = [f.line_start for f in members if f.line_start is not None]
        ends = [f.line_end or f.line_start for f in members if f.line_start is not None]

        if len(members) == 1:
            rule = CorrelationRule.SINGLETON
        else:
            member_ids = {f.finding_id for f in members}
            applied = [r for (x, y), r in rules.items() if x in member_ids and y in member_ids]
            # Same-line is the stronger signal; report it when any pair matched that way.
            rule = (
                CorrelationRule.SAME_CWE_SAME_LINE
                if CorrelationRule.SAME_CWE_SAME_LINE in applied
                else CorrelationRule.SAME_CWE_ADJACENT_LINES
            )

        cwe = _normalize_cwe(canonical.cwe)
        groups.append(
            CorrelatedFinding(
                correlation_id="",  # assigned below, after deterministic ordering
                canonical_finding_id=canonical_id,
                source_finding_ids=source_ids,
                scanners=scanners,
                file=canonical.file,
                line_start=min(starts) if starts else None,
                line_end=max(ends) if ends else None,
                cwe=cwe,
                severity=_highest_severity(members),
                # The canonical finding's category, carried through unchanged. No
                # "prefer the non-generic one" rule: with Semgrep reporting
                # `security` and Bandit reporting `blacklist`, "less generic" is not
                # reliably "more accurate", and picking between them would be a
                # judgement this deterministic layer has no basis to make.
                category=canonical.category,
                rule=rule,
                correlation_reason=_reason(rule, scanners, cwe, _location(members), members),
            )
        )

    # Number after sorting on stable content fields, never on iteration order.
    groups.sort(key=lambda g: (g.file or "", g.line_start or 0, g.cwe or "", g.canonical_finding_id))
    return [g.model_copy(update={"correlation_id": f"CORR-{i:03d}"}) for i, g in enumerate(groups, start=1)]
