"""
Conformance of emitted SARIF against the official OASIS 2.1.0 JSON Schema.

Separate from tests/test_sarif_report.py on purpose. That file asserts the
fields this project depends on - the emitter's contract with its own
consumers - and answers "does our output contain what we promise". This file
answers a different question: "is our output a valid SARIF document at all",
judged by the standard rather than by us. Keeping them apart means a change to
one concern cannot silently weaken the other.

The schema is vendored under tests/data/ rather than fetched, so the suite
stays offline and deterministic, and `jsonschema` is a dev-only extra, so the
installed package carries neither. The vendored file is byte-identical
(sha256 c3b4bb2d…) to the copy served from the OASIS canonical URL and from
the oasis-tcs mirror.

What this does NOT establish: schema conformance is a statement about document
shape, not about whether GitHub code scanning, Azure DevOps or any other
platform will ingest the file. That remains untested.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

jsonschema = pytest.importorskip("jsonschema", reason="dev-only dependency")

from sentinelai.contracts import (
    AIEnrichedFinding,
    ConfidenceLabel,
    CorrelatedFinding,
    RepositoryInfo,
    ScanMetadata,
    ScanMode,
    ScanResult,
    ScannerFinding,
    Severity,
    VerificationStatus,
)
from sentinelai.patching import PatchAttempt, PatchOutcome, PatchRunResult
from sentinelai.reporting import build_patch_application_report, to_sarif
from sentinelai.statistics import calculate_statistics

_SCHEMA_PATH = Path(__file__).parent / "data" / "sarif-schema-2.1.0.json"


@pytest.fixture(scope="module")
def validator():
    """Validator chosen from the schema's own declared draft, not assumed.

    The OASIS document declares draft-04; hard-coding a newer draft would
    silently reinterpret its keywords.
    """
    schema = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
    return jsonschema.validators.validator_for(schema)(schema)


def _finding(finding_id="SENT-001", **overrides):
    defaults = dict(
        finding_id=finding_id,
        scanner="semgrep",
        category="sql-injection",
        severity=Severity.CRITICAL,
        file="app/db.py",
        line_start=7,
        line_end=9,
        rule_id="semgrep.python.sql-injection.string-concat",
        message="User input concatenated into a SQL query.",
        raw_evidence='query = f"SELECT * FROM users WHERE name = \'{name}\'"',
        cwe="CWE-89",
    )
    defaults.update(overrides)
    return ScannerFinding(**defaults)


def _ai(finding_id="SENT-001", **overrides):
    defaults = dict(
        finding_id=finding_id,
        title="SQL injection",
        severity=Severity.CRITICAL,
        explanation="A query string is built by concatenating a parameter value.",
        exploit_path="A crafted parameter alters the statement.",
        impact="Unauthorised data access.",
        remediation="Use a parameterised query API.",
        patch_suggestion="cursor.execute(sql, (name,))",
        confidence_score=0.84,
        confidence_label=ConfidenceLabel.HIGH,
        verification_status=VerificationStatus.VERIFIED,
        references=["https://cwe.mitre.org/data/definitions/89.html"],
    )
    defaults.update(overrides)
    return AIEnrichedFinding(**defaults)


def _result(scanner_findings=None, ai_findings=None, correlated=None, mode=ScanMode.STANDARD):
    return ScanResult(
        repository=RepositoryInfo(name="demo", path="/tmp/demo", commit_hash="abc123", branch="main"),
        metadata=ScanMetadata(
            timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc), mode=mode, duration_seconds=1.5
        ),
        scanner_findings=scanner_findings if scanner_findings is not None else [_finding()],
        ai_findings=ai_findings or [],
        correlated_findings=correlated or [],
    )


def _emit(result, patch_application=None) -> dict:
    return json.loads(
        to_sarif(result, calculate_statistics(result), patch_application=patch_application)
    )


def _assert_valid(validator, document):
    errors = sorted(validator.iter_errors(document), key=lambda e: list(e.path))
    assert not errors, "\n".join(
        f"{'/'.join(str(p) for p in e.path) or '<root>'}: {e.message}" for e in errors[:10]
    )


# --- the claim the paper makes -------------------------------------------------------------------------------------


def test_the_vendored_schema_is_the_official_oasis_document(validator):
    """Guards provenance: a swapped or truncated schema would make every test below vacuous."""
    schema = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))

    assert schema["$schema"] == "http://json-schema.org/draft-04/schema#"
    assert schema["title"] == "Static Analysis Results Format (SARIF) Version 2.1.0 JSON Schema"
    # draft-04 spells it `id`; the value is the OASIS canonical location.
    assert schema["id"] == (
        "https://docs.oasis-open.org/sarif/sarif/v2.1.0/errata01/os/schemas/"
        "sarif-schema-2.1.0.json"
    )


def test_emitted_sarif_validates_against_the_official_schema(validator):
    _assert_valid(validator, _emit(_result()))


def test_the_emitted_document_declares_sarif_2_1_0(validator):
    document = _emit(_result())

    assert document["version"] == "2.1.0"
    assert document["$schema"].endswith("sarif-schema-2.1.0.json")


# --- every shape the emitter can produce ------------------------------------------------------------------------


def test_empty_scan_validates(validator):
    _assert_valid(validator, _emit(_result(scanner_findings=[])))


def test_ai_enriched_findings_validate(validator):
    _assert_valid(validator, _emit(_result(ai_findings=[_ai()])))


def test_correlated_findings_validate(validator):
    correlated = CorrelatedFinding(
        correlation_id="CORR-001",
        canonical_finding_id="SENT-001",
        source_finding_ids=["SENT-001", "SENT-002"],
        scanners=["semgrep", "bandit"],
        category="sql-injection",
        rule="same_cwe_same_line",
        correlation_reason="Both scanners reported CWE-89 at app/db.py:7.",
        severity=Severity.CRITICAL,
        file="app/db.py",
        line_start=7,
        cwe="CWE-89",
    )
    result = _result(
        scanner_findings=[_finding(), _finding("SENT-002", scanner="bandit", rule_id="B608")],
        correlated=[correlated],
    )
    _assert_valid(validator, _emit(result))


@pytest.mark.parametrize("severity", list(Severity))
def test_every_severity_maps_to_a_valid_sarif_level(validator, severity):
    """The four-tier internal scale is mapped down to SARIF's three levels."""
    _assert_valid(validator, _emit(_result(scanner_findings=[_finding(severity=severity)])))


def test_a_finding_without_a_location_validates(validator):
    """Dependency findings are manifest-scoped and carry no line."""
    result = _result(
        scanner_findings=[_finding(scanner="osv-scanner", file="requirements.txt",
                                   line_start=None, line_end=None, cwe=None)]
    )
    _assert_valid(validator, _emit(result))


def test_the_patch_application_property_bag_validates(validator):
    """Phase 3.2 added a run-level property; the standard must still accept the document."""
    run = PatchRunResult(
        attempts=(
            PatchAttempt("SENT-001", "app/db.py", PatchOutcome.APPLIED),
            PatchAttempt("SENT-002", None, PatchOutcome.SKIPPED_NO_PATCH),
        )
    )
    _assert_valid(validator, _emit(_result(), build_patch_application_report(run)))


# --- negative control --------------------------------------------------------------------------------------------


def test_the_validator_actually_rejects_invalid_sarif(validator):
    """Without this, a validator that accepted anything would make the suite meaningless."""
    document = _emit(_result())
    del document["runs"][0]["tool"]

    assert list(validator.iter_errors(document)), "a run with no tool must not validate"
