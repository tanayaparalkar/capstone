"""
Tests demonstrating the scanner-finding <-> AI-enriched-finding correlation
contract.

AIEnrichedFinding.finding_id is the correlation key back to the primary
ScannerFinding it enriches - a consumer (e.g. the reporting layer) joins
scanner_findings and ai_findings on this field. related_findings carries
the finding_ids of any additional findings participating in the same
multi-step exploit chain, without requiring a separate relationship model.
"""
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


def test_ai_finding_correlates_to_scanner_finding_by_id():
    scanner_finding = ScannerFinding(
        finding_id="SENT-002",
        scanner="semgrep",
        category="sql-injection",
        severity=Severity.CRITICAL,
        file="app/db/queries.py",
        line_start=47,
        line_end=47,
        rule_id="semgrep.python.sql-injection.string-concat",
        message="User-supplied input concatenated directly into a SQL query string.",
        raw_evidence="query = f\"SELECT * FROM users WHERE username = '{username}'\"",
        cwe="CWE-89",
    )

    ai_finding = AIEnrichedFinding(
        finding_id="SENT-002",
        title="SQL injection via unparameterized login query",
        severity=Severity.CRITICAL,
        scanner_sources=["semgrep"],
        explanation="The login query is built via string concatenation, allowing arbitrary SQL to be injected.",
        exploit_path="Submitting \"' OR '1'='1' --\" as the username bypasses authentication entirely.",
        impact="Full authentication bypass and potential database compromise.",
        remediation="Use parameterized queries instead of string concatenation.",
        confidence_score=0.95,
        confidence_label=ConfidenceLabel.HIGH,
        verification_status=VerificationStatus.VERIFIED,
    )

    # The correlation key: an AIEnrichedFinding enriches the ScannerFinding
    # that shares its finding_id.
    assert ai_finding.finding_id == scanner_finding.finding_id


def test_related_findings_represent_a_multi_step_exploit_chain():
    # SENT-001 leaks an AWS key; SENT-008 lets that same access path deserialize
    # attacker-controlled pickle data. An exploit-chain enrichment for SENT-008
    # references SENT-001 via related_findings rather than a separate model.
    chained_finding = AIEnrichedFinding(
        finding_id="SENT-008",
        title="Insecure deserialization reachable via leaked cache credentials",
        severity=Severity.CRITICAL,
        scanner_sources=["bandit", "gitleaks"],
        explanation=(
            "Cached objects are restored with pickle.loads() on attacker-reachable "
            "data. Combined with the leaked AWS credentials in SENT-001, an attacker "
            "with cache write access can plant a malicious payload."
        ),
        exploit_path=(
            "1) Use the leaked key from SENT-001 to gain cache write access. "
            "2) Write a malicious pickle payload to the cache. "
            "3) Trigger the vulnerable pickle.loads() call to execute code."
        ),
        impact="Remote code execution on the application server.",
        remediation="Rotate the leaked credentials and switch to a safe serialization format.",
        confidence_score=0.81,
        confidence_label=ConfidenceLabel.HIGH,
        verification_status=VerificationStatus.VERIFIED,
        related_findings=["SENT-001"],
    )

    assert chained_finding.finding_id == "SENT-008"
    assert "SENT-001" in chained_finding.related_findings


def test_correlation_join_across_a_scan_result():
    # Simulates the join a report generator performs: pairing each
    # AIEnrichedFinding with its primary ScannerFinding via finding_id.
    scanner_findings = [
        ScannerFinding(
            finding_id="SENT-002",
            scanner="semgrep",
            category="sql-injection",
            severity=Severity.CRITICAL,
            file="app/db/queries.py",
            line_start=47,
            rule_id="semgrep.python.sql-injection.string-concat",
            message="User-supplied input concatenated directly into a SQL query string.",
        ),
    ]
    ai_findings = [
        AIEnrichedFinding(
            finding_id="SENT-002",
            title="SQL injection via unparameterized login query",
            severity=Severity.CRITICAL,
            explanation="The login query is built via string concatenation.",
            remediation="Use parameterized queries instead of string concatenation.",
            confidence_score=0.9,
            confidence_label=ConfidenceLabel.HIGH,
            verification_status=VerificationStatus.VERIFIED,
        ),
    ]
    result = ScanResult(
        repository=RepositoryInfo(name="demo-app", path="/tmp/demo-app"),
        metadata=ScanMetadata(timestamp="2026-08-08T12:00:00Z", mode=ScanMode.STANDARD),
        scanner_findings=scanner_findings,
        ai_findings=ai_findings,
    )

    by_id = {f.finding_id: f for f in result.scanner_findings}
    for enriched in result.ai_findings:
        assert enriched.finding_id in by_id
