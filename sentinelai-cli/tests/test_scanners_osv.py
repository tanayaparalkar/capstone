"""
Tests for sentinelai.scanners.osv - the OSV-Scanner wrapper.
subprocess.run is always mocked; no real OSV-Scanner installation is
invoked and no request is ever made to osv.dev.

Fixtures reproduce the real shape of `osv-scanner scan source --format
json` output from OSV-Scanner 2.5.1 against the project's benchmark
repository - in particular the PYSEC/GHSA group pairing that makes
group-level emission necessary (50 raw vulnerabilities, 25 groups), and
the second `source.type: "unknown"` result holding transitive packages.
"""
import json
import subprocess
from unittest.mock import patch

import pytest

from sentinelai.backend.context_builder import build_repository_context
from sentinelai.backend.loader import load_repository
from sentinelai.contracts import ScannerFinding, Severity
from sentinelai.correlation import correlate_findings
from sentinelai.scanners.exceptions import ScannerExecutionError
from sentinelai.scanners.osv import OSVScanner

MANIFEST = "requirements.txt"


def _context(tmp_path):
    return build_repository_context(load_repository(str(tmp_path)))


def _completed(stdout="", stderr="", returncode=0):
    return subprocess.CompletedProcess(args=["osv-scanner"], returncode=returncode, stdout=stdout, stderr=stderr)


def _pysec(**overrides):
    """A PYSEC record: no database_specific, therefore no CWE - as observed in 25/25 real groups."""
    record = {
        "id": "PYSEC-2018-66",
        "aliases": ["CVE-2018-1000656", "GHSA-562c-5r94-xh97"],
        "details": "The Pallets Project flask is affected by a denial of service.",
        "affected": [
            {
                "package": {"name": "flask", "ecosystem": "PyPI"},
                "ranges": [{"type": "ECOSYSTEM", "events": [{"introduced": "0"}, {"fixed": "0.12.3"}]}],
            }
        ],
        "references": [{"type": "WEB", "url": "https://github.com/pallets/flask/pull/2691"}],
    }
    record.update(overrides)
    return record


def _ghsa(**overrides):
    """A GHSA record: this is where cwe_ids actually live."""
    record = {
        "id": "GHSA-562c-5r94-xh97",
        "aliases": ["CVE-2018-1000656", "PYSEC-2018-66"],
        "summary": "Improper Input Validation in Flask",
        "details": "Flask before 0.12.3 contains a CWE-20 vulnerability.",
        "database_specific": {"cwe_ids": ["CWE-20"], "severity": "HIGH"},
        "affected": [
            {
                "package": {"name": "flask", "ecosystem": "PyPI"},
                "ranges": [{"type": "ECOSYSTEM", "events": [{"introduced": "0"}, {"fixed": "0.12.3"}]}],
            }
        ],
        "references": [{"type": "ADVISORY", "url": "https://github.com/advisories/GHSA-562c-5r94-xh97"}],
    }
    record.update(overrides)
    return record


def _package(
    *,
    name="flask",
    version="0.12.2",
    ecosystem="PyPI",
    vulnerabilities=None,
    groups=None,
    max_severity="8.7",
):
    vulnerabilities = [_pysec(), _ghsa()] if vulnerabilities is None else vulnerabilities
    groups = (
        [{"ids": ["PYSEC-2018-66", "GHSA-562c-5r94-xh97"], "max_severity": max_severity}]
        if groups is None
        else groups
    )
    return {
        "package": {"name": name, "version": version, "ecosystem": ecosystem},
        "vulnerabilities": vulnerabilities,
        "groups": groups,
    }


def _payload(*packages, source_path=MANIFEST, source_type="lockfile", extra_results=()):
    results = [{"source": {"path": source_path, "type": source_type}, "packages": list(packages) or [_package()]}]
    results.extend(extra_results)
    return json.dumps({"results": results})


def _scan(tmp_path, stdout, returncode=1, stderr=""):
    """Default returncode 1: that is what OSV returns on a successful scan *with* findings."""
    with patch(
        "sentinelai.scanners.osv.subprocess.run",
        return_value=_completed(stdout=stdout, returncode=returncode, stderr=stderr),
    ):
        return OSVScanner().scan(_context(tmp_path))


# --- command construction -------------------------------------------------------------------------------------------


def test_command_uses_the_v2_scan_source_form(tmp_path):
    """OSV-Scanner 2.x removed the v1 top-level `--lockfile` invocation."""
    with patch("sentinelai.scanners.osv.subprocess.run", return_value=_completed(stdout=_payload())) as run:
        OSVScanner().scan(_context(tmp_path))

    command = run.call_args[0][0]
    assert command[:5] == ["osv-scanner", "scan", "source", "--format", "json"]
    assert command[5] == str(tmp_path.resolve())
    assert not any(arg.startswith("--lockfile") for arg in command)


def test_custom_executable_is_honoured(tmp_path):
    with patch("sentinelai.scanners.osv.subprocess.run", return_value=_completed(stdout=_payload())) as run:
        OSVScanner(executable="/opt/bin/osv-scanner").scan(_context(tmp_path))

    assert run.call_args[0][0][0] == "/opt/bin/osv-scanner"


# --- exit codes -----------------------------------------------------------------------------------------------------


def test_exit_one_with_findings_is_success(tmp_path):
    """OSV's convention: 1 means the scan succeeded and found vulnerabilities."""
    findings = _scan(tmp_path, _payload(), returncode=1)

    assert len(findings) == 1
    assert findings[0].scanner == "osv-scanner"


def test_exit_zero_with_no_findings_is_success(tmp_path):
    assert _scan(tmp_path, json.dumps({"results": []}), returncode=0) == []


@pytest.mark.parametrize("returncode", [2, 127, 128])
def test_other_exit_codes_are_failures(tmp_path, returncode):
    with pytest.raises(ScannerExecutionError, match=f"osv-scanner exited with status {returncode}"):
        _scan(tmp_path, "", returncode=returncode, stderr="something went wrong")


def test_no_package_sources_is_reported_by_its_real_cause_not_a_decode_error(tmp_path):
    """At exit 128 stdout is empty; parsing first would hide the real message."""
    with pytest.raises(ScannerExecutionError, match="No package sources found"):
        _scan(tmp_path, "", returncode=128, stderr="No package sources found, --help for usage information.")


def test_failure_preserves_actionable_stderr(tmp_path):
    with pytest.raises(ScannerExecutionError, match="failed to resolve path"):
        _scan(tmp_path, "", returncode=127, stderr="failed to resolve path: stat /nope: no such file or directory")


# --- group / alias deduplication ------------------------------------------------------------------------------------


def test_one_finding_per_group_not_per_vulnerability(tmp_path):
    """The real benchmark has 50 vulnerabilities in 25 groups - every group is a PYSEC/GHSA pair."""
    findings = _scan(tmp_path, _payload(_package()))

    assert len(findings) == 1  # two raw records, one logical advisory


def test_representative_is_deterministic_and_carries_the_cwe(tmp_path):
    """sorted(ids)[0] selects the GHSA record, which is where cwe_ids live in 25/25 real groups."""
    finding = _scan(tmp_path, _payload(_package()))[0]

    assert finding.rule_id == "GHSA-562c-5r94-xh97"
    assert finding.cwe == "CWE-20"


def test_cwe_falls_back_across_group_members(tmp_path):
    """OSV's own ids[0] is the PYSEC record, which carries no CWE at all - 0/25 real groups."""
    package = _package(
        vulnerabilities=[_pysec(id="AAA-1", aliases=[]), _ghsa(id="ZZZ-9")],
        groups=[{"ids": ["AAA-1", "ZZZ-9"], "max_severity": "8.7"}],
    )

    finding = _scan(tmp_path, _payload(package))[0]

    assert finding.rule_id == "AAA-1"  # representative has no CWE of its own
    assert finding.cwe == "CWE-20"  # recovered from the other member


def test_all_group_ids_and_aliases_are_preserved(tmp_path):
    finding = _scan(tmp_path, _payload(_package()))[0]

    assert "PYSEC-2018-66" in finding.message
    assert "GHSA-562c-5r94-xh97" in finding.message
    assert "group ids: GHSA-562c-5r94-xh97, PYSEC-2018-66" in finding.raw_evidence
    assert "CVE-2018-1000656" in finding.raw_evidence  # the alias a reader will recognize


def test_group_with_no_ids_is_skipped_rather_than_invented(tmp_path):
    package = _package(groups=[{"ids": [], "max_severity": "8.7"}])

    assert _scan(tmp_path, _payload(package)) == []


def test_two_distinct_groups_stay_two_findings(tmp_path):
    package = _package(
        vulnerabilities=[_pysec(), _ghsa(), _pysec(id="PYSEC-2019-179"), _ghsa(id="GHSA-5wv5-4vpf-pj6m")],
        groups=[
            {"ids": ["PYSEC-2018-66", "GHSA-562c-5r94-xh97"], "max_severity": "8.7"},
            {"ids": ["PYSEC-2019-179", "GHSA-5wv5-4vpf-pj6m"], "max_severity": "4.3"},
        ],
    )

    findings = _scan(tmp_path, _payload(package))

    assert len(findings) == 2
    assert {f.rule_id for f in findings} == {"GHSA-562c-5r94-xh97", "GHSA-5wv5-4vpf-pj6m"}


# --- severity -------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "max_severity,expected",
    [
        ("9.8", Severity.CRITICAL),
        ("9.0", Severity.CRITICAL),
        ("8.7", Severity.HIGH),
        ("7.0", Severity.HIGH),
        ("6.9", Severity.MEDIUM),
        ("4.0", Severity.MEDIUM),
        ("3.9", Severity.LOW),
        ("", Severity.LOW),
        (None, Severity.LOW),
        ("not-a-number", Severity.LOW),
    ],
)
def test_cvss_score_maps_through_the_standard_bands(tmp_path, max_severity, expected):
    """OSV exposes no severity enum anywhere - only a CVSS score string on the group."""
    package = _package(groups=[{"ids": ["GHSA-562c-5r94-xh97"], "max_severity": max_severity}])

    assert _scan(tmp_path, _payload(package))[0].severity is expected


# --- fixed versions -------------------------------------------------------------------------------------------------


def test_fixed_version_is_extracted_from_ecosystem_ranges(tmp_path):
    finding = _scan(tmp_path, _payload(_package()))[0]

    assert "Fixed in 0.12.3" in finding.message
    assert "fixed: 0.12.3" in finding.raw_evidence


def test_git_ranges_are_excluded_so_commit_hashes_are_not_offered_as_versions(tmp_path):
    """Real case: requests' PYSEC-2018-28 lists both a commit hash and 2.20.0 under `fixed`."""
    record = _pysec(
        affected=[
            {
                "package": {"name": "requests", "ecosystem": "PyPI"},
                "ranges": [
                    {"type": "GIT", "events": [{"introduced": "0"}, {"fixed": "c45d7c49ea75133e52ab22a8e9e13173"}]},
                    {"type": "ECOSYSTEM", "events": [{"introduced": "0"}, {"fixed": "2.20.0"}]},
                ],
            }
        ]
    )
    package = _package(vulnerabilities=[record], groups=[{"ids": ["PYSEC-2018-66"], "max_severity": "7.5"}])

    finding = _scan(tmp_path, _payload(package))[0]

    assert "fixed: 2.20.0" in finding.raw_evidence
    assert "c45d7c49" not in finding.raw_evidence


def test_multiple_fixed_versions_are_sorted_for_determinism(tmp_path):
    record = _pysec(
        affected=[
            {
                "package": {"name": "flask", "ecosystem": "PyPI"},
                "ranges": [{"type": "ECOSYSTEM", "events": [{"fixed": "2.3.2"}, {"fixed": "2.2.5"}]}],
            }
        ]
    )
    package = _package(vulnerabilities=[record], groups=[{"ids": ["PYSEC-2018-66"], "max_severity": "5.0"}])

    assert "fixed: 2.2.5, 2.3.2" in _scan(tmp_path, _payload(package))[0].raw_evidence


def test_absent_fix_is_stated_explicitly(tmp_path):
    record = _pysec(affected=[{"package": {"name": "flask", "ecosystem": "PyPI"}, "ranges": []}])
    package = _package(vulnerabilities=[record], groups=[{"ids": ["PYSEC-2018-66"], "max_severity": "5.0"}])

    finding = _scan(tmp_path, _payload(package))[0]

    assert "No fixed version is available." in finding.message
    assert "fixed: none available" in finding.raw_evidence


# --- transitive packages --------------------------------------------------------------------------------------------


def test_transitive_packages_from_the_unknown_source_are_reported(tmp_path):
    """OSV resolves dependencies that never appear in the manifest - idna and urllib3 in the benchmark."""
    transitive = {
        "source": {"path": MANIFEST, "type": "unknown"},
        "packages": [
            _package(
                name="urllib3",
                version="1.23.0",
                vulnerabilities=[_pysec(id="PYSEC-2019-132"), _ghsa(id="GHSA-r64q-w8jr-g9qp")],
                groups=[{"ids": ["PYSEC-2019-132", "GHSA-r64q-w8jr-g9qp"], "max_severity": "6.1"}],
            )
        ],
    }

    findings = _scan(tmp_path, _payload(_package(), extra_results=(transitive,)))

    assert len(findings) == 2
    transitive_finding = next(f for f in findings if "urllib3" in f.finding_id)
    assert transitive_finding.severity is Severity.MEDIUM
    assert transitive_finding.file == MANIFEST
    assert "urllib3 1.23.0 (PyPI)" in transitive_finding.message


# --- no line numbers ------------------------------------------------------------------------------------------------


def test_no_line_numbers_are_ever_synthesized(tmp_path):
    findings = _scan(tmp_path, _payload(_package()))

    assert all(f.line_start is None and f.line_end is None for f in findings)


def test_dependency_findings_remain_singletons_under_the_existing_correlation_rules(tmp_path):
    """Two same-CWE advisories on one manifest must not merge - they are distinct issues."""
    package = _package(
        vulnerabilities=[_ghsa(id="GHSA-aaaa"), _ghsa(id="GHSA-bbbb")],
        groups=[
            {"ids": ["GHSA-aaaa"], "max_severity": "9.8"},
            {"ids": ["GHSA-bbbb"], "max_severity": "9.8"},
        ],
    )

    findings = _scan(tmp_path, _payload(package))
    groups = correlate_findings(findings)

    assert len(findings) == 2
    assert {f.cwe for f in findings} == {"CWE-20"}  # same CWE, same file, no lines
    assert len(groups) == 2
    assert all(len(g.source_finding_ids) == 1 for g in groups)


# --- deterministic ids ----------------------------------------------------------------------------------------------


def test_finding_id_is_derived_not_positional(tmp_path):
    finding = _scan(tmp_path, _payload(_package()))[0]

    assert finding.finding_id == "osv-requirements.txt:flask@0.12.2:GHSA-562c-5r94-xh97"


def test_finding_ids_are_unique_across_packages_and_groups(tmp_path):
    package = _package(
        vulnerabilities=[_ghsa(id="GHSA-aaaa"), _ghsa(id="GHSA-bbbb")],
        groups=[{"ids": ["GHSA-aaaa"], "max_severity": "5"}, {"ids": ["GHSA-bbbb"], "max_severity": "5"}],
    )
    other = _package(name="requests", version="2.19.1", vulnerabilities=[_ghsa(id="GHSA-aaaa")],
                     groups=[{"ids": ["GHSA-aaaa"], "max_severity": "5"}])

    findings = _scan(tmp_path, _payload(package, other))

    assert len({f.finding_id for f in findings}) == len(findings) == 3


def test_duplicate_groups_do_not_collapse_finding_ids(tmp_path):
    package = _package(groups=[{"ids": ["GHSA-x"], "max_severity": "5"}, {"ids": ["GHSA-x"], "max_severity": "5"}])

    findings = _scan(tmp_path, _payload(package))

    assert len({f.finding_id for f in findings}) == len(findings) == 2


def test_output_is_deterministic_across_calls(tmp_path):
    payload = _payload(_package())

    assert _scan(tmp_path, payload) == _scan(tmp_path, payload)


# --- manifest path normalization ------------------------------------------------------------------------------------


def test_absolute_source_path_is_made_relative(tmp_path):
    """OSV reports source.path absolute; the finding records it relative to the repository root."""
    absolute = str(tmp_path.resolve() / "requirements.txt")

    assert _scan(tmp_path, _payload(_package(), source_path=absolute))[0].file == "requirements.txt"


def test_source_path_outside_the_repository_is_left_alone(tmp_path):
    assert _scan(tmp_path, _payload(_package(), source_path="/elsewhere/req.txt"))[0].file == "/elsewhere/req.txt"


# --- failures (fail-closed) -----------------------------------------------------------------------------------------


def test_missing_executable_raises_scanner_execution_error(tmp_path):
    with patch("sentinelai.scanners.osv.subprocess.run", side_effect=FileNotFoundError("no osv-scanner")):
        with pytest.raises(ScannerExecutionError, match="Failed to run osv-scanner"):
            OSVScanner().scan(_context(tmp_path))


def test_invalid_json_raises_scanner_execution_error(tmp_path):
    with pytest.raises(ScannerExecutionError, match="Failed to parse osv-scanner JSON output"):
        _scan(tmp_path, "not json at all")


def test_unsupported_output_shape_raises_scanner_execution_error(tmp_path):
    payload = json.dumps({"results": [{"source": {"path": MANIFEST}, "packages": "nope"}]})

    with pytest.raises(ScannerExecutionError, match="Failed to parse osv-scanner JSON output"):
        _scan(tmp_path, payload)


def test_unexpected_subprocess_failure_propagates_with_cause(tmp_path):
    with patch("sentinelai.scanners.osv.subprocess.run", side_effect=OSError("boom")):
        with pytest.raises(ScannerExecutionError) as excinfo:
            OSVScanner().scan(_context(tmp_path))

    assert isinstance(excinfo.value.__cause__, OSError)


def test_missing_optional_fields_use_sensible_defaults(tmp_path):
    payload = json.dumps(
        {"results": [{"source": {"path": MANIFEST}, "packages": [{"groups": [{"ids": ["GHSA-x"]}]}]}]}
    )

    finding = _scan(tmp_path, payload)[0]

    assert finding.rule_id == "GHSA-x"
    assert finding.severity is Severity.LOW
    assert finding.cwe is None
    assert "unknown" in finding.message


def test_scanner_returns_only_scanner_finding_objects(tmp_path):
    assert all(isinstance(f, ScannerFinding) for f in _scan(tmp_path, _payload(_package())))
