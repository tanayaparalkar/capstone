"""Unit tests for the shared Pydantic contracts."""
import json

import pytest
from pydantic import ValidationError

from sentinelai.contracts import (
    AIEnrichedFinding,
    ConfidenceLabel,
    RepositoryInfo,
    ScanMetadata,
    ScanMode,
    ScanResult,
    ScannerFinding,
    Severity,
    VerificationStatus,
)


def _valid_scanner_finding_kwargs():
    return dict(
        finding_id="SENT-001",
        scanner="semgrep",
        category="sql-injection",
        severity=Severity.CRITICAL,
        file="app/db/queries.py",
        line_start=47,
        line_end=47,
        rule_id="semgrep.python.sql-injection.string-concat",
        message="User input concatenated directly into a SQL query.",
        raw_evidence="query = f\"SELECT * FROM users WHERE name = '{username}'\"",
        cwe="CWE-89",
    )


class TestScannerFinding:
    def test_valid_finding_constructs(self):
        finding = ScannerFinding(**_valid_scanner_finding_kwargs())
        assert finding.severity == Severity.CRITICAL
        assert finding.line_start == 47

    def test_optional_line_fields_default_to_none(self):
        kwargs = _valid_scanner_finding_kwargs()
        kwargs.pop("line_start")
        kwargs.pop("line_end")
        kwargs["file"] = "requirements.txt"
        finding = ScannerFinding(**kwargs)
        assert finding.line_start is None
        assert finding.line_end is None

    def test_optional_file_and_cwe_default_to_none(self):
        kwargs = _valid_scanner_finding_kwargs()
        kwargs.pop("file")
        kwargs.pop("cwe")
        kwargs.pop("line_start")
        kwargs.pop("line_end")
        finding = ScannerFinding(**kwargs)
        assert finding.file is None
        assert finding.cwe is None

    def test_invalid_severity_raises(self):
        kwargs = _valid_scanner_finding_kwargs()
        kwargs["severity"] = "catastrophic"
        with pytest.raises(ValidationError):
            ScannerFinding(**kwargs)

    def test_missing_required_field_raises(self):
        kwargs = _valid_scanner_finding_kwargs()
        kwargs.pop("rule_id")
        with pytest.raises(ValidationError):
            ScannerFinding(**kwargs)

    def test_line_end_without_line_start_raises(self):
        kwargs = _valid_scanner_finding_kwargs()
        kwargs.pop("line_start")
        with pytest.raises(ValidationError):
            ScannerFinding(**kwargs)

    def test_line_end_before_line_start_raises(self):
        kwargs = _valid_scanner_finding_kwargs()
        kwargs["line_start"] = 50
        kwargs["line_end"] = 10
        with pytest.raises(ValidationError):
            ScannerFinding(**kwargs)

    def test_round_trip_serialization(self):
        finding = ScannerFinding(**_valid_scanner_finding_kwargs())
        restored = ScannerFinding.model_validate_json(finding.model_dump_json())
        assert restored == finding


def _valid_ai_finding_kwargs():
    return dict(
        finding_id="SENT-001",
        title="Hardcoded AWS credentials",
        severity=Severity.CRITICAL,
        scanner_sources=["gitleaks"],
        evidence="AWS_ACCESS_KEY_ID assigned a literal value in config/settings.py",
        repository_context="config/settings.py is imported at application startup",
        explanation=(
            "A hardcoded AWS access key grants full programmatic account access "
            "to anyone with repo read access."
        ),
        exploit_path="An attacker with repo read access extracts the key and uses it directly with the AWS CLI.",
        impact="Full compromise of the associated AWS account.",
        remediation="Revoke the key and move credentials to a secrets manager.",
        patch_suggestion=None,
        confidence_score=0.92,
        confidence_label=ConfidenceLabel.HIGH,
        verification_status=VerificationStatus.VERIFIED,
        related_findings=[],
        references=["CWE-798"],
    )


class TestAIEnrichedFinding:
    def test_valid_finding_constructs(self):
        finding = AIEnrichedFinding(**_valid_ai_finding_kwargs())
        assert finding.confidence_label == ConfidenceLabel.HIGH

    def test_confidence_score_out_of_range_raises(self):
        kwargs = _valid_ai_finding_kwargs()
        kwargs["confidence_score"] = 1.5
        with pytest.raises(ValidationError):
            AIEnrichedFinding(**kwargs)

    def test_invalid_verification_status_raises(self):
        kwargs = _valid_ai_finding_kwargs()
        kwargs["verification_status"] = "maybe"
        with pytest.raises(ValidationError):
            AIEnrichedFinding(**kwargs)

    def test_missing_required_field_raises(self):
        kwargs = _valid_ai_finding_kwargs()
        kwargs.pop("explanation")
        with pytest.raises(ValidationError):
            AIEnrichedFinding(**kwargs)

    def test_round_trip_serialization(self):
        finding = AIEnrichedFinding(**_valid_ai_finding_kwargs())
        restored = AIEnrichedFinding.model_validate_json(finding.model_dump_json())
        assert restored == finding


class TestScanResult:
    def test_valid_scan_result_constructs(self):
        result = ScanResult(
            repository=RepositoryInfo(name="demo-app", path="/tmp/demo-app"),
            metadata=ScanMetadata(timestamp="2026-08-08T12:00:00Z", mode=ScanMode.STANDARD),
            scanner_findings=[ScannerFinding(**_valid_scanner_finding_kwargs())],
            ai_findings=[AIEnrichedFinding(**_valid_ai_finding_kwargs())],
        )
        assert len(result.scanner_findings) == 1
        assert len(result.ai_findings) == 1

    def test_findings_default_to_empty_lists(self):
        result = ScanResult(
            repository=RepositoryInfo(name="demo-app", path="/tmp/demo-app"),
            metadata=ScanMetadata(timestamp="2026-08-08T12:00:00Z", mode=ScanMode.QUICK),
        )
        assert result.scanner_findings == []
        assert result.ai_findings == []

    def test_invalid_mode_raises(self):
        with pytest.raises(ValidationError):
            ScanMetadata(timestamp="2026-08-08T12:00:00Z", mode="turbo")

    def test_round_trip_serialization(self):
        result = ScanResult(
            repository=RepositoryInfo(name="demo-app", path="/tmp/demo-app"),
            metadata=ScanMetadata(timestamp="2026-08-08T12:00:00Z", mode=ScanMode.FULL, duration_seconds=12.4),
            scanner_findings=[ScannerFinding(**_valid_scanner_finding_kwargs())],
        )
        restored = ScanResult.model_validate(json.loads(result.model_dump_json()))
        assert restored == result
