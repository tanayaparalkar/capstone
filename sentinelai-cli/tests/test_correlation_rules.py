"""
Correlation rules: what merges, what does not, and the invariants.

The no-merge cases carry more weight than the merge cases. Correlation
that is too eager silently hides findings behind one another, which in a
security tool is worse than reporting a duplicate - so every rule that
could over-merge (same severity, same scanner, same generic category,
same file) has an explicit test proving it does not.

Every scenario is built from real shapes observed in the benchmark
repository, including the three CWE formats the scanners actually emit.
"""
import random

import pytest

from sentinelai.contracts import CorrelationRule, ScannerFinding, Severity
from sentinelai.correlation import correlate_findings
from sentinelai.correlation.correlator import _LINE_TOLERANCE, _normalize_cwe

FILE_A = "app/tasks.py"
FILE_B = "app/db.py"


def _f(finding_id, *, scanner="bandit", cwe="CWE-78", file=FILE_A, line=10,
       line_end=None, severity=Severity.MEDIUM, category="blacklist") -> ScannerFinding:
    return ScannerFinding(
        finding_id=finding_id, scanner=scanner, category=category, severity=severity,
        file=file, line_start=line, line_end=line_end or line,
        rule_id="R1", message="m", raw_evidence="e", cwe=cwe,
    )


def _groups_by_size(findings):
    groups = correlate_findings(findings)
    return groups, {len(g.source_finding_ids) for g in groups}


# --- CWE normalization -------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("CWE-327", "CWE-327"),
        ("CWE-327: Use of a Broken or Risky Cryptographic Algorithm", "CWE-327"),
        ("CWE-78: Improper Neutralization of Special Elements used in an OS Command", "CWE-78"),
        ("cwe-89", "CWE-89"),
        (None, None),
        ("", None),
        ("no cwe here", None),
    ],
)
def test_cwe_normalization(raw, expected):
    # The three scanners emit three shapes for one concept; raw equality would
    # correlate nothing at all.
    assert _normalize_cwe(raw) == expected


def test_differently_formatted_same_cwe_correlates():
    findings = [
        _f("bandit-0", scanner="bandit", cwe="CWE-327", line=6),
        _f("semgrep-0", scanner="semgrep", cwe="CWE-327: Use of a Broken or Risky Cryptographic Algorithm", line=6),
    ]

    groups = correlate_findings(findings)

    assert len(groups) == 1
    assert groups[0].cwe == "CWE-327"  # normalized form only
    assert groups[0].scanners == ["bandit", "semgrep"]


# --- MERGE cases -------------------------------------------------------------------------------------------------------


def test_same_cwe_same_line_merges():
    groups = correlate_findings([
        _f("bandit-6", scanner="bandit", line=21),
        _f("semgrep-3", scanner="semgrep", line=21),
    ])

    assert len(groups) == 1
    assert groups[0].rule is CorrelationRule.SAME_CWE_SAME_LINE
    assert "same location" in groups[0].correlation_reason


def test_overlapping_line_ranges_merge():
    groups = correlate_findings([
        _f("a", scanner="bandit", line=10, line_end=14),
        _f("b", scanner="semgrep", line=12, line_end=13),
    ])

    assert len(groups) == 1
    assert groups[0].rule is CorrelationRule.SAME_CWE_SAME_LINE


def test_adjacent_lines_within_tolerance_merge():
    # The real db.py case: bandit blames the assignment, semgrep the execution.
    groups = correlate_findings([
        _f("bandit-1", scanner="bandit", cwe="CWE-89", file=FILE_B, line=6),
        _f("semgrep-1", scanner="semgrep", cwe="CWE-89: SQL Injection", file=FILE_B, line=7),
    ])

    assert len(groups) == 1
    assert groups[0].rule is CorrelationRule.SAME_CWE_ADJACENT_LINES
    assert "adjacent lines" in groups[0].correlation_reason
    assert str(_LINE_TOLERANCE) in groups[0].correlation_reason
    assert "not proof" in groups[0].correlation_reason  # heuristic, stated as such


def test_exactly_at_tolerance_merges():
    groups = correlate_findings([_f("a", line=10), _f("b", scanner="semgrep", line=10 + _LINE_TOLERANCE)])
    assert len(groups) == 1


def test_three_scanners_form_one_group_transitively():
    groups = correlate_findings([
        _f("a", scanner="bandit", line=10),
        _f("b", scanner="semgrep", line=11),
        _f("c", scanner="gitleaks", line=12),
    ])

    assert len(groups) == 1
    assert groups[0].scanners == ["bandit", "gitleaks", "semgrep"]
    assert groups[0].source_finding_ids == ["a", "b", "c"]


# --- NO-MERGE cases (the ones that matter most) ---------------------------------------------------------------------------


def test_different_cwe_at_the_same_line_does_not_merge():
    # The real tasks.py:29 case - bandit says CWE-78, semgrep says CWE-95.
    groups = correlate_findings([
        _f("bandit-9", scanner="bandit", cwe="CWE-78", line=29),
        _f("semgrep-4", scanner="semgrep", cwe="CWE-95: Eval Injection", line=29),
    ])

    assert len(groups) == 2


def test_missing_cwe_never_merges():
    # The real settings.py:14 case - GitLeaks emits no CWE at all. Secret-family
    # correlation is deliberately future work, not an exception bolted on here.
    groups = correlate_findings([
        _f("gitleaks-0", scanner="gitleaks", cwe=None, line=14, category="generic-api-key"),
        _f("semgrep-5", scanner="semgrep", cwe="CWE-798: Hard-coded Credentials", line=14),
    ])

    assert len(groups) == 2
    assert all(g.rule is CorrelationRule.SINGLETON for g in groups)


def test_beyond_tolerance_does_not_merge():
    # The real CWE-502 case: `import pickle` at 6 vs `pickle.loads` at 13.
    groups = correlate_findings([
        _f("a", cwe="CWE-502", line=6),
        _f("b", scanner="semgrep", cwe="CWE-502", line=13),
    ])

    assert len(groups) == 2


def test_same_severity_alone_does_not_merge():
    groups = correlate_findings([
        _f("a", cwe="CWE-78", line=10, severity=Severity.HIGH),
        _f("b", cwe="CWE-502", line=10, severity=Severity.HIGH),
    ])

    assert len(groups) == 2


def test_same_scanner_alone_does_not_merge():
    groups = correlate_findings([
        _f("a", scanner="bandit", cwe="CWE-78", line=10),
        _f("b", scanner="bandit", cwe="CWE-502", line=10),
    ])

    assert len(groups) == 2


def test_same_generic_category_alone_does_not_merge():
    # Bandit files four unrelated checks under `blacklist`; Semgrep files nearly
    # everything under `security`. Category is not identity.
    groups = correlate_findings([
        _f("a", cwe="CWE-502", line=6, category="blacklist"),
        _f("b", cwe="CWE-78", line=7, category="blacklist"),
    ])

    assert len(groups) == 2


def test_same_file_alone_does_not_merge():
    groups = correlate_findings([
        _f("a", cwe="CWE-78", line=10),
        _f("b", cwe="CWE-78", line=100),
    ])

    assert len(groups) == 2


def test_different_files_never_merge():
    groups = correlate_findings([
        _f("a", cwe="CWE-78", file=FILE_A, line=10),
        _f("b", cwe="CWE-78", file=FILE_B, line=10),
    ])

    assert len(groups) == 2


def test_missing_line_information_never_merges():
    a = ScannerFinding(finding_id="a", scanner="osv", category="dep", severity=Severity.HIGH,
                       file=FILE_A, rule_id="R", message="m", cwe="CWE-78")
    b = _f("b", scanner="semgrep", cwe="CWE-78", line=10)

    groups = correlate_findings([a, b])

    assert len(groups) == 2


# --- canonical selection, severity, category ------------------------------------------------------------------------------


def test_canonical_is_the_lowest_sorting_source_id():
    groups = correlate_findings([
        _f("zzz", scanner="semgrep", line=10),
        _f("aaa", scanner="bandit", line=10),
    ])

    assert groups[0].canonical_finding_id == "aaa"
    assert groups[0].canonical_finding_id in groups[0].source_finding_ids


def test_severity_uses_project_ordering_not_string_comparison():
    # Lexicographically "medium" > "high"; by SEVERITY_RANK, high > medium.
    groups = correlate_findings([
        _f("a", scanner="bandit", line=10, severity=Severity.MEDIUM),
        _f("b", scanner="semgrep", line=10, severity=Severity.HIGH),
    ])

    assert groups[0].severity is Severity.HIGH


def test_critical_beats_high():
    groups = correlate_findings([
        _f("a", scanner="bandit", line=10, severity=Severity.HIGH),
        _f("b", scanner="semgrep", line=10, severity=Severity.CRITICAL),
    ])

    assert groups[0].severity is Severity.CRITICAL


def test_category_is_the_canonical_findings_category():
    # Documented rule: carry the canonical category through unchanged rather than
    # guessing which scanner's label is "less generic".
    groups = correlate_findings([
        _f("aaa", scanner="bandit", line=10, category="blacklist"),
        _f("zzz", scanner="semgrep", line=10, category="security"),
    ])

    assert groups[0].canonical_finding_id == "aaa"
    assert groups[0].category == "blacklist"


# --- invariants ------------------------------------------------------------------------------------------------------------


def _corpus():
    return [
        _f("bandit-0", scanner="bandit", cwe="CWE-327", file="a.py", line=6, severity=Severity.HIGH),
        _f("semgrep-0", scanner="semgrep", cwe="CWE-327: Broken Crypto", file="a.py", line=6),
        _f("bandit-1", scanner="bandit", cwe="CWE-89", file=FILE_B, line=6),
        _f("semgrep-1", scanner="semgrep", cwe="CWE-89: SQLi", file=FILE_B, line=7),
        _f("gitleaks-0", scanner="gitleaks", cwe=None, file="s.py", line=14),
        _f("lonely", scanner="bandit", cwe="CWE-20", file="c.py", line=99),
    ]


def test_every_raw_finding_appears_exactly_once():
    findings = _corpus()

    groups = correlate_findings(findings)
    all_ids = sorted(i for g in groups for i in g.source_finding_ids)

    assert all_ids == sorted(f.finding_id for f in findings)
    assert sum(len(g.source_finding_ids) for g in groups) == len(findings)


def test_correlation_is_order_independent():
    findings = _corpus()
    expected = [(g.correlation_id, g.canonical_finding_id, g.source_finding_ids) for g in correlate_findings(findings)]

    for seed in range(10):
        shuffled = findings[:]
        random.Random(seed).shuffle(shuffled)
        actual = [(g.correlation_id, g.canonical_finding_id, g.source_finding_ids) for g in correlate_findings(shuffled)]
        assert actual == expected, f"ordering changed results (seed {seed})"


def test_correlation_ids_are_sequential_and_stable():
    groups = correlate_findings(_corpus())

    assert [g.correlation_id for g in groups] == [f"CORR-{i:03d}" for i in range(1, len(groups) + 1)]


def test_empty_input_returns_no_groups():
    assert correlate_findings([]) == []


def test_singletons_are_labelled_and_explained():
    groups = correlate_findings([_f("only", cwe="CWE-20", line=5)])

    assert len(groups) == 1
    assert groups[0].rule is CorrelationRule.SINGLETON
    assert groups[0].source_finding_ids == ["only"]
    assert not groups[0].is_multi_scanner
    assert "no other finding matched" in groups[0].correlation_reason


def test_multi_scanner_flag_reflects_distinct_scanners():
    same_scanner = correlate_findings([_f("a", scanner="bandit", line=10), _f("b", scanner="bandit", line=10)])
    cross = correlate_findings([_f("a", scanner="bandit", line=10), _f("b", scanner="semgrep", line=10)])

    assert same_scanner[0].is_multi_scanner is False  # one tool twice is not corroboration
    assert cross[0].is_multi_scanner is True
