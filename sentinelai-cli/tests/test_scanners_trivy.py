"""
Tests for sentinelai.scanners.trivy - the Trivy scanner wrapper.
subprocess.run is always mocked; no real Trivy installation is ever
invoked and no vulnerability database is ever downloaded.

Fixtures are trimmed copies of genuine `trivy fs --scanners vuln --format
json` output from Trivy 0.74.0 run against the project's benchmark
repository, including the two empty shapes that are easy to get wrong:
`Results: null` for a tree with no manifest, and a Result carrying
`Packages` but no `Vulnerabilities` key when a manifest is clean.
"""
import json
import subprocess
from unittest.mock import patch

import pytest

from sentinelai.backend.context_builder import build_repository_context
from sentinelai.backend.loader import load_repository
from sentinelai.contracts import ScannerFinding, Severity
from sentinelai.scanners.exceptions import ScannerExecutionError
from sentinelai.scanners.trivy import TrivyScanner


def _context(tmp_path):
    return build_repository_context(load_repository(str(tmp_path)))


def _completed(stdout="", stderr="", returncode=0):
    return subprocess.CompletedProcess(args=["trivy"], returncode=returncode, stdout=stdout, stderr=stderr)


def _vulnerability(**overrides):
    """A real CVE record as Trivy 0.74.0 emits it, trimmed to the parsed fields."""
    vulnerability = {
        "VulnerabilityID": "CVE-2018-1000656",
        "PkgName": "flask",
        "InstalledVersion": "0.12.2",
        "FixedVersion": "0.12.3",
        "Status": "fixed",
        "Severity": "HIGH",
        "CweIDs": ["CWE-20"],
        "Title": "python-flask: Denial of Service via crafted JSON file",
        "Description": "The Pallets Project flask version Before 0.12.3 contains a CWE-20 vulnerability.",
        "PrimaryURL": "https://avd.aquasec.com/nvd/cve-2018-1000656",
        "References": [
            "https://access.redhat.com/security/cve/CVE-2018-1000656",
            "https://github.com/advisories/GHSA-562c-5r94-xh97",
        ],
        "Fingerprint": "a" * 64,
    }
    vulnerability.update(overrides)
    return vulnerability


def _payload(*vulnerabilities, target="requirements.txt"):
    return json.dumps(
        {
            "SchemaVersion": 2,
            "Results": [
                {
                    "Target": target,
                    "Class": "lang-pkgs",
                    "Type": "pip",
                    "Vulnerabilities": list(vulnerabilities),
                }
            ],
        }
    )


def _scan(tmp_path, stdout, returncode=0, stderr=""):
    with patch(
        "sentinelai.scanners.trivy.subprocess.run",
        return_value=_completed(stdout=stdout, returncode=returncode, stderr=stderr),
    ):
        return TrivyScanner().scan(_context(tmp_path))


# --- command construction -------------------------------------------------------------------------------------------


def test_command_uses_vulnerability_only_filesystem_mode(tmp_path):
    """`--scanners vuln` is not optional: Trivy's default is vuln,secret, which would duplicate GitLeaks."""
    with patch(
        "sentinelai.scanners.trivy.subprocess.run", return_value=_completed(stdout=_payload())
    ) as run:
        TrivyScanner().scan(_context(tmp_path))

    command = run.call_args[0][0]
    assert command[:6] == ["trivy", "fs", "--scanners", "vuln", "--format", "json"]
    assert command[6] == str(tmp_path.resolve())
    assert "secret" not in command
    assert "misconfig" not in command
    assert "--exit-code" not in command  # would make findings look like a failure


def test_custom_executable_is_honoured(tmp_path):
    with patch(
        "sentinelai.scanners.trivy.subprocess.run", return_value=_completed(stdout=_payload())
    ) as run:
        TrivyScanner(executable="/opt/bin/trivy").scan(_context(tmp_path))

    assert run.call_args[0][0][0] == "/opt/bin/trivy"


# --- successful parsing ---------------------------------------------------------------------------------------------


def test_exit_zero_with_findings_is_success(tmp_path):
    """Trivy returns 0 whether or not it found anything - verified against the real tool."""
    findings = _scan(tmp_path, _payload(_vulnerability()), returncode=0)

    assert len(findings) == 1
    finding = findings[0]
    assert isinstance(finding, ScannerFinding)
    assert finding.scanner == "trivy"
    assert finding.category == "vulnerable-dependency"
    assert finding.rule_id == "CVE-2018-1000656"
    assert finding.severity is Severity.HIGH
    assert finding.file == "requirements.txt"
    assert finding.cwe == "CWE-20"


def test_message_names_package_version_advisory_and_fix(tmp_path):
    finding = _scan(tmp_path, _payload(_vulnerability()))[0]

    assert "flask" in finding.message
    assert "0.12.2" in finding.message
    assert "CVE-2018-1000656" in finding.message
    assert "Denial of Service" in finding.message
    assert "Fixed in 0.12.3" in finding.message


def test_message_says_so_when_no_fix_exists(tmp_path):
    finding = _scan(tmp_path, _payload(_vulnerability(FixedVersion="")))[0]

    assert "No fixed version is available." in finding.message


def test_evidence_is_a_concise_summary_not_the_raw_json(tmp_path):
    finding = _scan(tmp_path, _payload(_vulnerability()))[0]

    assert finding.raw_evidence.splitlines()[:6] == [
        "manifest: requirements.txt",
        "package: flask",
        "installed: 0.12.2",
        "fixed: 0.12.3",
        "advisory: CVE-2018-1000656",
        "severity: HIGH",
    ]
    # The whole record is not dumped in: no CVSS block, no PkgIdentifier, no fingerprint.
    assert "CVSS" not in finding.raw_evidence
    assert "PkgIdentifier" not in finding.raw_evidence
    assert "Fingerprint" not in finding.raw_evidence
    assert len(finding.raw_evidence) < len(_payload(_vulnerability()))


def test_evidence_carries_every_cwe_even_though_the_field_holds_one(tmp_path):
    finding = _scan(tmp_path, _payload(_vulnerability(CweIDs=["CWE-20", "CWE-400"])))[0]

    assert finding.cwe == "CWE-20"  # contract holds a single value
    assert "cwe: CWE-20, CWE-400" in finding.raw_evidence  # nothing is lost


def test_references_are_capped_with_primary_url_first(tmp_path):
    finding = _scan(
        tmp_path, _payload(_vulnerability(References=[f"https://example.test/{i}" for i in range(20)]))
    )[0]
    references = [line for line in finding.raw_evidence.splitlines() if line.startswith("reference: ")]

    assert len(references) == 3
    assert references[0] == "reference: https://avd.aquasec.com/nvd/cve-2018-1000656"


# --- severity -------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "trivy_severity,expected",
    [
        ("CRITICAL", Severity.CRITICAL),
        ("HIGH", Severity.HIGH),
        ("MEDIUM", Severity.MEDIUM),
        ("LOW", Severity.LOW),
        ("UNKNOWN", Severity.LOW),
        ("", Severity.LOW),
        (None, Severity.LOW),
    ],
)
def test_severity_mapping(tmp_path, trivy_severity, expected):
    finding = _scan(tmp_path, _payload(_vulnerability(Severity=trivy_severity)))[0]

    assert finding.severity is expected


# --- CWE ------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cwe_ids,expected",
    [
        (["CWE-20"], "CWE-20"),
        (["CWE-20", "CWE-400"], "CWE-20"),  # first only, per the contract's single field
        ([], None),
        (None, None),
        (["", "CWE-79"], "CWE-79"),  # blanks skipped rather than emitted
    ],
)
def test_cwe_extraction(tmp_path, cwe_ids, expected):
    """CweIDs is a list, and 1 of the benchmark's 12 real vulnerabilities has none."""
    finding = _scan(tmp_path, _payload(_vulnerability(CweIDs=cwe_ids)))[0]

    assert finding.cwe == expected


# --- no line numbers ------------------------------------------------------------------------------------------------


def test_no_line_numbers_are_ever_synthesized(tmp_path):
    """A dependency vulnerability belongs to the dependency, not to a manifest line."""
    findings = _scan(tmp_path, _payload(_vulnerability(), _vulnerability(VulnerabilityID="CVE-2019-1010083")))

    assert all(f.line_start is None and f.line_end is None for f in findings)


def test_package_locations_are_ignored_even_when_present(tmp_path):
    """Trivy exposes package-level Locations; using them would merge distinct CVEs.

    Measured on the real benchmark: CVE-2020-14343 and CVE-2020-1747 both sit
    at pyyaml's line 2 under CWE-20, so line-based correlation would collapse
    two real advisories into one issue.
    """
    payload = json.dumps(
        {
            "Results": [
                {
                    "Target": "requirements.txt",
                    "Packages": [
                        {"Name": "pyyaml", "Version": "5.1.0", "Locations": [{"StartLine": 2, "EndLine": 2}]}
                    ],
                    "Vulnerabilities": [
                        _vulnerability(VulnerabilityID="CVE-2020-14343", PkgName="pyyaml", CweIDs=["CWE-20"]),
                        _vulnerability(VulnerabilityID="CVE-2020-1747", PkgName="pyyaml", CweIDs=["CWE-20"]),
                    ],
                }
            ]
        }
    )

    findings = _scan(tmp_path, payload)

    assert len(findings) == 2
    assert all(f.line_start is None for f in findings)
    assert {f.rule_id for f in findings} == {"CVE-2020-14343", "CVE-2020-1747"}


# --- deterministic ids ----------------------------------------------------------------------------------------------


def test_finding_id_is_derived_not_positional(tmp_path):
    finding = _scan(tmp_path, _payload(_vulnerability()))[0]

    assert finding.finding_id == "trivy-requirements.txt:flask@0.12.2:CVE-2018-1000656"


def test_finding_id_is_stable_when_an_unrelated_advisory_appears_first(tmp_path):
    """A positional id would renumber every finding as the advisory database grows."""
    alone = _scan(tmp_path, _payload(_vulnerability()))
    with_newcomer = _scan(
        tmp_path, _payload(_vulnerability(VulnerabilityID="CVE-2000-0001"), _vulnerability())
    )

    assert alone[0].finding_id == with_newcomer[1].finding_id


def test_finding_ids_are_unique_even_if_trivy_repeats_a_record(tmp_path):
    """finding_id is the join key every renderer indexes on - duplicates silently collapse entries."""
    findings = _scan(tmp_path, _payload(_vulnerability(), _vulnerability()))

    assert len({f.finding_id for f in findings}) == len(findings) == 2


def test_same_package_in_two_manifests_gets_distinct_ids(tmp_path):
    payload = json.dumps(
        {
            "Results": [
                {"Target": "requirements.txt", "Vulnerabilities": [_vulnerability()]},
                {"Target": "app/requirements.txt", "Vulnerabilities": [_vulnerability()]},
            ]
        }
    )

    findings = _scan(tmp_path, payload)

    assert len({f.finding_id for f in findings}) == 2
    assert {f.file for f in findings} == {"requirements.txt", "app/requirements.txt"}


def test_output_is_deterministic_across_calls(tmp_path):
    payload = _payload(_vulnerability(), _vulnerability(VulnerabilityID="CVE-2019-1010083", CweIDs=[]))

    assert _scan(tmp_path, payload) == _scan(tmp_path, payload)


# --- manifest path normalization ------------------------------------------------------------------------------------


def test_absolute_target_under_the_repository_is_made_relative(tmp_path):
    absolute = str(tmp_path.resolve() / "requirements.txt")
    finding = _scan(tmp_path, _payload(_vulnerability(), target=absolute))[0]

    assert finding.file == "requirements.txt"


def test_target_outside_the_repository_is_left_alone(tmp_path):
    finding = _scan(tmp_path, _payload(_vulnerability(), target="/elsewhere/requirements.txt"))[0]

    assert finding.file == "/elsewhere/requirements.txt"


# --- empty results --------------------------------------------------------------------------------------------------


def test_null_results_means_no_manifest_not_an_error(tmp_path):
    """Trivy emits `Results: null`, not `[]`, for a tree with no manifest - verified against the real tool."""
    assert _scan(tmp_path, json.dumps({"SchemaVersion": 2, "Results": None})) == []


def test_clean_manifest_has_packages_but_no_vulnerabilities_key(tmp_path):
    """Verified against the real tool: a manifest with no known CVEs omits the key entirely."""
    payload = json.dumps(
        {"Results": [{"Target": "requirements.txt", "Packages": [{"Name": "six", "Version": "1.17.0"}]}]}
    )

    assert _scan(tmp_path, payload) == []


def test_empty_vulnerability_list(tmp_path):
    assert _scan(tmp_path, _payload()) == []


# --- failures (fail-closed) -----------------------------------------------------------------------------------------


def test_missing_executable_raises_scanner_execution_error(tmp_path):
    with patch("sentinelai.scanners.trivy.subprocess.run", side_effect=FileNotFoundError("no trivy")):
        with pytest.raises(ScannerExecutionError, match="Failed to run trivy"):
            TrivyScanner().scan(_context(tmp_path))


def test_non_zero_exit_status_is_a_failure_not_findings(tmp_path):
    """Unlike Bandit/GitLeaks/OSV, findings never make Trivy's exit code non-zero."""
    with pytest.raises(ScannerExecutionError, match="trivy exited with status 1"):
        _scan(tmp_path, "", returncode=1, stderr="FATAL\tfs scan error: stat /nope: no such file or directory")


def test_failure_preserves_actionable_stderr(tmp_path):
    with pytest.raises(ScannerExecutionError, match="no such file or directory"):
        _scan(tmp_path, "", returncode=1, stderr="FATAL\tfs scan error: stat /nope: no such file or directory")


def test_invalid_json_raises_scanner_execution_error(tmp_path):
    with pytest.raises(ScannerExecutionError, match="Failed to parse trivy JSON output"):
        _scan(tmp_path, "not json at all")


def test_unsupported_output_shape_raises_scanner_execution_error(tmp_path):
    with pytest.raises(ScannerExecutionError, match="Failed to parse trivy JSON output"):
        _scan(tmp_path, json.dumps({"Results": [{"Target": "requirements.txt", "Vulnerabilities": "nope"}]}))


def test_unexpected_subprocess_failure_propagates_with_cause(tmp_path):
    with patch("sentinelai.scanners.trivy.subprocess.run", side_effect=OSError("boom")):
        with pytest.raises(ScannerExecutionError) as excinfo:
            TrivyScanner().scan(_context(tmp_path))

    assert isinstance(excinfo.value.__cause__, OSError)


def test_missing_optional_fields_use_sensible_defaults(tmp_path):
    payload = json.dumps({"Results": [{"Target": "requirements.txt", "Vulnerabilities": [{}]}]})

    finding = _scan(tmp_path, payload)[0]

    assert finding.rule_id == "unknown"
    assert finding.severity is Severity.LOW
    assert finding.cwe is None
    assert finding.line_start is None


def test_scanner_returns_only_scanner_finding_objects(tmp_path):
    findings = _scan(tmp_path, _payload(_vulnerability()))

    assert all(isinstance(f, ScannerFinding) for f in findings)
