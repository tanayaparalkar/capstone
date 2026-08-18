"""
Precise unit tests for --fail-on gating (sentinelai.core.severity.exceeds_fail_on_threshold)
and the exit-code/severity infrastructure it relies on.

These use synthetic ScanStatistics rather than the CLI's fixed mock
dataset, because the mock data always contains critical findings - which
makes it impossible to construct a CLI-level scenario where a high (or
lower) --fail-on threshold is *not* crossed. Testing the pure function
directly lets us cover the full severity-tier matrix precisely,
including cases (e.g. "no findings at all", "only medium present")
that can't be produced through tests/test_cli.py against the real
mock provider.

exceeds_fail_on_threshold() lives in sentinelai.core.severity (not
sentinelai.main) - it's pure severity-comparison policy with no CLI
dependency, so main.py just imports and calls it. See Milestone 12's
audit notes in core/severity.py for why it was moved there.
"""
from sentinelai.contracts import RepositoryInfo, ScanMetadata, ScanMode
from sentinelai.core import SEVERITY_RANK, ExitCode, exceeds_fail_on_threshold
from sentinelai.core.severity import SEVERITY_RANK as SEVERITY_RANK_DIRECT
from sentinelai.statistics import ScanStatistics


def _stats(critical=0, high=0, medium=0, low=0) -> ScanStatistics:
    return ScanStatistics(
        total_findings=critical + high + medium + low,
        critical_findings=critical,
        high_findings=high,
        medium_findings=medium,
        low_findings=low,
        ai_enrichment_status="unavailable",
        repository_name="demo",
        scan_mode="standard",
        timestamp="2026-08-09T12:00:00Z",
    )


# --- exit codes -----------------------------------------------------------------


def test_exit_code_values_are_stable_and_distinct():
    assert ExitCode.SUCCESS == 0
    assert ExitCode.SECURITY_FINDINGS == 1
    assert ExitCode.INVALID_INPUT == 2
    assert ExitCode.PROVIDER_ERROR == 3
    assert ExitCode.INTERNAL_ERROR == 4
    values = [ExitCode.SUCCESS, ExitCode.SECURITY_FINDINGS, ExitCode.INVALID_INPUT, ExitCode.PROVIDER_ERROR, ExitCode.INTERNAL_ERROR]
    assert len(set(values)) == len(values)  # all distinct


# --- centralized severity ranking -----------------------------------------------------------------


def test_severity_rank_is_monotonic_low_to_critical():
    from sentinelai.contracts import Severity

    assert SEVERITY_RANK[Severity.LOW] < SEVERITY_RANK[Severity.MEDIUM]
    assert SEVERITY_RANK[Severity.MEDIUM] < SEVERITY_RANK[Severity.HIGH]
    assert SEVERITY_RANK[Severity.HIGH] < SEVERITY_RANK[Severity.CRITICAL]


def test_severity_rank_is_a_single_shared_object():
    # sentinelai.core exports the same dict core.severity defines - not a copy.
    assert SEVERITY_RANK is SEVERITY_RANK_DIRECT


# --- --fail-on: exact matrix from the milestone spec -----------------------------------------------------------------


class TestFailOnHigh:
    def test_low_finding_does_not_fail(self):
        assert exceeds_fail_on_threshold(_stats(low=1), "high") is False

    def test_medium_finding_does_not_fail(self):
        assert exceeds_fail_on_threshold(_stats(medium=1), "high") is False

    def test_high_finding_fails(self):
        assert exceeds_fail_on_threshold(_stats(high=1), "high") is True

    def test_critical_finding_fails(self):
        assert exceeds_fail_on_threshold(_stats(critical=1), "high") is True


class TestFailOnCritical:
    def test_high_finding_does_not_fail(self):
        assert exceeds_fail_on_threshold(_stats(high=1), "critical") is False

    def test_medium_finding_does_not_fail(self):
        assert exceeds_fail_on_threshold(_stats(medium=1), "critical") is False

    def test_low_finding_does_not_fail(self):
        assert exceeds_fail_on_threshold(_stats(low=1), "critical") is False

    def test_critical_finding_fails(self):
        assert exceeds_fail_on_threshold(_stats(critical=1), "critical") is True


class TestFailOnMedium:
    def test_low_finding_does_not_fail(self):
        assert exceeds_fail_on_threshold(_stats(low=1), "medium") is False

    def test_medium_finding_fails(self):
        assert exceeds_fail_on_threshold(_stats(medium=1), "medium") is True

    def test_high_finding_fails(self):
        assert exceeds_fail_on_threshold(_stats(high=1), "medium") is True

    def test_critical_finding_fails(self):
        assert exceeds_fail_on_threshold(_stats(critical=1), "medium") is True


class TestFailOnLow:
    def test_low_finding_fails(self):
        assert exceeds_fail_on_threshold(_stats(low=1), "low") is True

    def test_any_higher_severity_also_fails(self):
        assert exceeds_fail_on_threshold(_stats(critical=1), "low") is True


class TestFailOnNone:
    def test_never_fails_even_with_critical_findings(self):
        assert exceeds_fail_on_threshold(_stats(critical=4, high=4, medium=2), "none") is False

    def test_never_fails_with_no_findings(self):
        assert exceeds_fail_on_threshold(_stats(), "none") is False


class TestFailOnNoFindings:
    def test_no_findings_never_fails_regardless_of_threshold(self):
        for threshold in ("low", "medium", "high", "critical"):
            assert exceeds_fail_on_threshold(_stats(), threshold) is False
