from datetime import datetime, timezone
from pathlib import Path
from typer.testing import CliRunner

from sentinelai.contracts import (
    AIEnrichedFinding,
    ConfidenceLabel,
    RepositoryInfo,
    ScannerFinding,
    ScannerTier,
    ScanMetadata,
    ScanMode,
    ScanResult,
    Severity,
    VerificationStatus,
)
from sentinelai.main import app
from sentinelai.patcher import (
    Patch,
    PatchStatus,
    apply_patch,
    extract_patch,
    rollback_patch,
    run_remediation_session,
    validate_code_syntax,
    verify_patch,
)
from rich.console import Console
import io


def _make_finding(
    file_path: str = "app/db.py",
    line_start: int = 6,
    line_end: int = 7,
    category: str = "sql-injection",
    cwe: str = "CWE-89",
    rule_id: str = "semgrep.sql-injection",
) -> ScannerFinding:
    return ScannerFinding(
        finding_id="SENT-001",
        scanner="semgrep",
        category=category,
        severity=Severity.HIGH,
        file=file_path,
        line_start=line_start,
        line_end=line_end,
        rule_id=rule_id,
        message="SQL query constructed via string concatenation",
        raw_evidence='query = "SELECT * FROM users WHERE name = \'" + username + "\'"',
        cwe=cwe,
    )


def test_validate_code_syntax():
    py_path = Path("sample.py")
    valid_code = "def foo():\n    return 42\n"
    invalid_code = "def foo():\n    return 42 +"

    assert validate_code_syntax(py_path, valid_code) is None
    err = validate_code_syntax(py_path, invalid_code)
    assert err is not None
    assert "syntax error" in err.lower()


def test_extract_patch_from_ai_suggestion(tmp_path):
    target_file = tmp_path / "app.py"
    target_file.write_text(
        "import hashlib\n\ndef hash_pass(p):\n    return hashlib.md5(p.encode()).hexdigest()\n",
        encoding="utf-8",
    )

    finding = ScannerFinding(
        finding_id="SENT-HASH-1",
        scanner="bandit",
        category="weak-cryptography",
        severity=Severity.MEDIUM,
        file="app.py",
        line_start=4,
        line_end=4,
        rule_id="bandit.B303",
        message="Use of weak MD5 hash",
        cwe="CWE-327",
    )

    ai = AIEnrichedFinding(
        finding_id="SENT-HASH-1",
        title="Insecure MD5 hash function",
        severity=Severity.MEDIUM,
        explanation="MD5 is cryptographically broken and vulnerable to collision attacks.",
        remediation="Upgrade hashing algorithm to SHA-256.",
        patch_suggestion="```python\n    return hashlib.sha256(p.encode()).hexdigest()\n```",
        confidence_score=0.95,
        confidence_label=ConfidenceLabel.HIGH,
        verification_status=VerificationStatus.VERIFIED,
    )

    patch = extract_patch(tmp_path, finding, ai)
    assert patch is not None
    assert patch.finding_id == "SENT-HASH-1"
    assert "sha256" in patch.replacement_snippet
    assert "md5" in patch.original_snippet
    assert "+    return hashlib.sha256(p.encode()).hexdigest()" in patch.diff


def test_extract_patch_fallback_template(tmp_path):
    db_file = tmp_path / "db.py"
    db_file.write_text(
        "import sqlite3\n\n"
        "def get_user(conn, username):\n"
        "    query = \"SELECT * FROM users WHERE name = '\" + username + \"'\"\n"
        "    return conn.execute(query).fetchone()\n",
        encoding="utf-8",
    )

    finding = _make_finding(file_path="db.py", line_start=4, line_end=5)
    patch = extract_patch(tmp_path, finding, ai=None)

    assert patch is not None
    assert "WHERE name = ?\"" in patch.replacement_snippet
    assert "execute(query, (username,))" in patch.replacement_snippet


def test_apply_patch_and_rollback(tmp_path):
    target = tmp_path / "calc.py"
    target.write_text("def run(a, b):\n    return a + b\n", encoding="utf-8")

    patch = Patch(
        finding_id="TEST-1",
        file_path=target,
        line_start=2,
        line_end=2,
        original_snippet="    return a + b",
        replacement_snippet="    return a * b",
        diff="--- a/calc.py\n+++ b/calc.py\n@@ -2,1 +2,1 @@\n-    return a + b\n+    return a * b",
        explanation="Multiply instead of add",
    )

    result = apply_patch(patch)
    assert result.status == PatchStatus.APPLIED
    assert target.read_text(encoding="utf-8") == "def run(a, b):\n    return a * b\n"

    # Rollback
    rolled_back = rollback_patch(result)
    assert rolled_back is True
    assert target.read_text(encoding="utf-8") == "def run(a, b):\n    return a + b\n"


def test_apply_patch_rejects_syntax_error(tmp_path):
    target = tmp_path / "broken.py"
    initial_content = "def test():\n    pass\n"
    target.write_text(initial_content, encoding="utf-8")

    patch = Patch(
        finding_id="TEST-ERR",
        file_path=target,
        line_start=2,
        line_end=2,
        original_snippet="    pass",
        replacement_snippet="    pass +++ invalid syntax !!!",
        diff="",
        explanation="Syntax error test",
    )

    result = apply_patch(patch, validate_syntax=True)
    assert result.status == PatchStatus.FAILED
    assert "syntax error" in result.message.lower()
    # Content must remain uncorrupted
    assert target.read_text(encoding="utf-8") == initial_content


def test_remediation_session_auto_yes(tmp_path):
    sample = tmp_path / "sample.py"
    sample.write_text(
        "import yaml\n\ndef load_data(text):\n    return yaml.load(text)\n",
        encoding="utf-8",
    )

    finding = ScannerFinding(
        finding_id="SENT-YAML-1",
        scanner="bandit",
        category="insecure-deserialization",
        severity=Severity.HIGH,
        file="sample.py",
        line_start=4,
        line_end=4,
        rule_id="bandit.B506",
        message="Use of unsafe yaml.load",
        cwe="CWE-502",
    )

    repo_info = RepositoryInfo(
        name="test-repo",
        path=str(tmp_path),
        commit_hash="abc",
        branch="main",
        languages=["python"],
    )
    scan_meta = ScanMetadata(
        timestamp=datetime.now(timezone.utc),
        mode=ScanMode.STANDARD,
        duration_seconds=0.5,
        scanner_tier=ScannerTier.CORE,
    )
    result = ScanResult(
        repository=repo_info,
        metadata=scan_meta,
        scanner_findings=[finding],
    )

    buf = io.StringIO()
    console = Console(file=buf, force_terminal=True)

    summary = run_remediation_session(
        repo_path=tmp_path,
        result=result,
        console=console,
        auto_yes=True,
    )

    assert summary.total_findings == 1
    assert summary.patches_applied == 1
    assert summary.vulnerabilities_avoided == 1
    assert "safe_load" in sample.read_text(encoding="utf-8")


def test_cli_fix_command_help():
    runner = CliRunner()
    res = runner.invoke(app, ["fix", "--help"])
    assert res.exit_code == 0
    assert "narrate detected vulnerabilities" in res.output.lower()
    assert "--yes" in res.output
    assert "--dry-run" in res.output


def test_cli_scan_with_fix_flag_help():
    runner = CliRunner()
    res = runner.invoke(app, ["scan", "--help"])
    assert res.exit_code == 0
    assert "--fix" in res.output


def test_structured_patch_schema():
    from sentinelai.ai.agents.schemas import PatchType, RemediationPlan, StructuredPatch

    sp = StructuredPatch(
        patch_type=PatchType.UNIFIED_DIFF,
        target_file="app/db.py",
        line_start=6,
        line_end=7,
        patch_content="--- a/app/db.py\n+++ b/app/db.py\n",
    )
    plan = RemediationPlan(
        remediation="Use parameterized queries.",
        structured_patch=sp,
    )
    assert plan.structured_patch is not None
    assert plan.structured_patch.patch_type == PatchType.UNIFIED_DIFF
    assert plan.structured_patch.target_file == "app/db.py"


def test_atomic_snapshot_and_undo(tmp_path):
    from sentinelai.patcher import (
        apply_patch_atomically,
        check_git_clean,
        create_snapshot,
        restore_latest_backup_session,
    )

    clean, _ = check_git_clean(tmp_path)
    assert isinstance(clean, bool)

    target = tmp_path / "service.py"
    target.write_text("API_KEY = 'secret'\n", encoding="utf-8")

    # Apply atomic patch
    patch = Patch(
        finding_id="SECRET-1",
        file_path=target,
        line_start=1,
        line_end=1,
        original_snippet="API_KEY = 'secret'",
        replacement_snippet="API_KEY = os.environ.get('API_KEY')",
        diff="",
        explanation="Load secret from environment",
    )
    res = apply_patch_atomically(patch, repo_root=tmp_path)
    assert res.status == PatchStatus.APPLIED
    assert "os.environ" in target.read_text(encoding="utf-8")

    # Verify backup snapshot was created in .sentinelai/backups/
    backup_dir = tmp_path / ".sentinelai" / "backups"
    assert backup_dir.exists()
    assert (backup_dir / "backup_manifest.json").exists()

    # Manual Undo / Rollback session
    restored = restore_latest_backup_session(tmp_path)
    assert len(restored) == 1
    assert restored[0][1] is True
    assert target.read_text(encoding="utf-8") == "API_KEY = 'secret'\n"


def test_cli_undo_command(tmp_path):
    runner = CliRunner()
    res = runner.invoke(app, ["undo", "--help"])
    assert res.exit_code == 0
    assert "revert the most recent" in res.output.lower()


# ─── Phase 4: Closed-Loop Verification & Vulnerabilities Avoided Tests ────────


def test_verification_diffing_confirmed_fixed():
    """Diffing returns CONFIRMED_FIXED when scanner reports 0 findings."""
    from sentinelai.patcher import diff_findings, VerificationStatus

    patch = Patch(
        finding_id="SENT-SQL-1",
        file_path=Path("db.py"),
        line_start=10,
        line_end=12,
        original_snippet="SELECT * FROM users",
        replacement_snippet="SELECT * FROM users WHERE id = ?",
        diff="",
        explanation="Use parameterization",
        cwe="CWE-89",
        rule_id="semgrep.python.sqli",
    )

    # Empty findings list from targeted re-scan
    rescan_findings = []
    ok, msg, status = diff_findings(patch, rescan_findings, "db.py")
    assert ok is True
    assert status == VerificationStatus.CONFIRMED_FIXED
    assert "Confirmed Fixed" in msg
    assert "CWE-89" in msg or "semgrep.python.sqli" in msg
    assert "0 findings" in msg


def test_verification_diffing_unresolved():
    """Diffing returns UNRESOLVED when the scanner still triggers the original rule."""
    from sentinelai.patcher import diff_findings, VerificationStatus

    patch = Patch(
        finding_id="SENT-SQL-1",
        file_path=Path("db.py"),
        line_start=10,
        line_end=12,
        original_snippet="SELECT * FROM users",
        replacement_snippet="SELECT * FROM users WHERE id = ?",
        diff="",
        explanation="Use parameterization",
        cwe="CWE-89",
        rule_id="semgrep.python.sqli",
    )

    # Scanner re-scan still reports the exact same rule on line 11
    rescan_findings = [
        {
            "rule_id": "semgrep.python.sqli",
            "line_start": 11,
            "cwe": "CWE-89",
            "message": "Possible SQL injection detected",
        }
    ]
    ok, msg, status = diff_findings(patch, rescan_findings, "db.py")
    assert ok is False
    assert status == VerificationStatus.UNRESOLVED
    assert "Unresolved" in msg
    assert "semgrep.python.sqli" in msg
    assert "line 11" in msg


def test_verification_diffing_regressed():
    """Diffing returns REGRESSED when original rule is gone but new warning appeared."""
    from sentinelai.patcher import diff_findings, VerificationStatus

    patch = Patch(
        finding_id="SENT-SQL-1",
        file_path=Path("db.py"),
        line_start=10,
        line_end=12,
        original_snippet="SELECT * FROM users",
        replacement_snippet="SELECT * FROM users WHERE id = ?",
        diff="",
        explanation="Use parameterization",
        cwe="CWE-89",
        rule_id="semgrep.python.sqli",
    )

    # Re-scan reports a DIFFERENT, newly introduced warning
    rescan_findings = [
        {
            "rule_id": "bandit.B301.pickle",
            "line_start": 14,
            "cwe": "CWE-502",
            "message": "Deserialization of untrusted data",
        }
    ]
    ok, msg, status = diff_findings(patch, rescan_findings, "db.py")
    assert ok is False
    assert status == VerificationStatus.REGRESSED
    assert "Regressed" in msg
    assert "bandit.B301.pickle" in msg


def test_verify_patch_syntax_error_regression(tmp_path):
    """verify_patch detects syntax error regressions in Python files."""
    from sentinelai.patcher import verify_patch, VerificationStatus

    target = tmp_path / "broken.py"
    target.write_text("def broken():\n    return 1 +\n", encoding="utf-8")

    patch = Patch(
        finding_id="SENT-001",
        file_path=target,
        line_start=1,
        line_end=2,
        original_snippet="def broken():\n    return 1",
        replacement_snippet="def broken():\n    return 1 +",
        diff="",
        explanation="Syntax broken",
    )

    res = verify_patch(patch, tmp_path)
    assert res.verified is False
    assert res.status == VerificationStatus.SYNTAX_ERROR
    assert "Syntax regression" in res.message


def test_verify_patch_with_custom_rescan_fn(tmp_path):
    """verify_patch correctly delegates to targeted re-scan engine."""
    from sentinelai.patcher import verify_patch, VerificationStatus

    target = tmp_path / "clean.py"
    target.write_text("def safe_query(db, user_id):\n    return db.execute('SELECT * FROM users WHERE id = ?', (user_id,))\n", encoding="utf-8")

    patch = Patch(
        finding_id="SENT-002",
        file_path=target,
        line_start=2,
        line_end=2,
        original_snippet="return db.execute('SELECT * FROM users WHERE id = ' + user_id)",
        replacement_snippet="return db.execute('SELECT * FROM users WHERE id = ?', (user_id,))",
        diff="",
        explanation="Use parameters",
        cwe="CWE-89",
        rule_id="semgrep.sql-injection",
        scanner="semgrep",
    )

    called = []

    def mock_rescan(fpath: Path):
        called.append(fpath)
        return []  # 0 findings

    res = verify_patch(patch, tmp_path, rescan_fn=mock_rescan)
    assert len(called) == 1
    assert called[0] == target
    assert res.verified is True
    assert res.status == VerificationStatus.CONFIRMED_FIXED
    assert "Confirmed Fixed: CWE-89" in res.message


def test_closed_loop_remediation_session_success(tmp_path):
    """End-to-end remediation session verifies patch, prints notifications, and updates summary."""
    from sentinelai.patcher import run_remediation_session, VerificationStatus

    app_py = tmp_path / "app.py"
    app_py.write_text(
        "def query_user(conn, username):\n"
        "    cursor = conn.cursor()\n"
        "    cursor.execute(\"SELECT * FROM users WHERE username = '\" + username + \"'\")\n"
        "    return cursor.fetchall()\n",
        encoding="utf-8",
    )

    finding = ScannerFinding(
        finding_id="SENT-SQL-1",
        scanner="semgrep",
        category="sql-injection",
        severity=Severity.HIGH,
        file="app.py",
        line_start=3,
        line_end=3,
        rule_id="semgrep.python.sqli",
        message="SQL injection via string concatenation",
        raw_evidence='cursor.execute("SELECT * FROM users WHERE username = \'" + username + "\'")',
        cwe="CWE-89",
    )

    scan_res = ScanResult(
        scanner_findings=[finding],
        ai_findings=[],
        correlated_findings=[],
        repository=RepositoryInfo(name="test-repo", path=str(tmp_path), languages=["python"]),
        metadata=ScanMetadata(
            timestamp=datetime.now(timezone.utc),
            mode=ScanMode.STANDARD,
            duration_seconds=0.5,
            scanner_tier=ScannerTier.CORE,
        ),
    )

    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False, no_color=True, width=120)

    # Run session with mock rescan that confirms 0 findings
    summary = run_remediation_session(
        repo_path=tmp_path,
        result=scan_res,
        console=console,
        auto_yes=True,
        rescan_fn=lambda path: [],
    )

    out = buf.getvalue()

    # 1. Check all notification messages match requirements
    assert "Running closed-loop verification re-scan on app.py" in out
    assert "[APPLIED] Code updated successfully in app.py" in out
    assert "[AVOIDED / RESOLVED] Confirmed Fixed: CWE-89" in out

    # 2. Check summary metrics
    assert summary.total_findings == 1
    assert summary.patches_applied == 1
    assert summary.vulnerabilities_avoided == 1
    assert summary.static_analysis_clean is True
    assert len(summary.mitigated_findings) == 1
    mf = summary.mitigated_findings[0]
    assert mf.finding_id == "SENT-SQL-1"
    assert mf.cwe == "CWE-89"
    assert mf.severity == Severity.HIGH
    assert mf.file == "app.py"
    assert mf.status == VerificationStatus.CONFIRMED_FIXED

    # 3. Check final dashboard contains expected elements
    assert "SentinelAI Remediation Summary" in out
    assert "Vulnerabilities Successfully Mitigated" in out
    assert "Clean (Passed)" in out
    assert "Confirmation: Code passes static analysis clean." in out


def test_closed_loop_remediation_session_regression_detected(tmp_path):
    """Session handles regression notifications and marks static analysis not clean."""
    from sentinelai.patcher import run_remediation_session, VerificationStatus

    app_py = tmp_path / "app.py"
    app_py.write_text(
        "def query_user(conn, username):\n"
        "    cursor = conn.cursor()\n"
        "    cursor.execute(\"SELECT * FROM users WHERE username = '\" + username + \"'\")\n"
        "    return cursor.fetchall()\n",
        encoding="utf-8",
    )

    finding = ScannerFinding(
        finding_id="SENT-SQL-2",
        scanner="semgrep",
        category="sql-injection",
        severity=Severity.HIGH,
        file="app.py",
        line_start=3,
        line_end=3,
        rule_id="semgrep.python.sqli",
        message="SQL injection via string concatenation",
        cwe="CWE-89",
    )

    scan_res = ScanResult(
        scanner_findings=[finding],
        ai_findings=[],
        correlated_findings=[],
        repository=RepositoryInfo(name="test-repo", path=str(tmp_path), languages=["python"]),
        metadata=ScanMetadata(
            timestamp=datetime.now(timezone.utc),
            mode=ScanMode.STANDARD,
            duration_seconds=0.5,
            scanner_tier=ScannerTier.CORE,
        ),
    )

    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False, no_color=True, width=120)

    # Re-scan introduces a new warning
    def regressed_rescan(path: Path):
        return [{"rule_id": "bandit.B608", "line_start": 3, "message": "Hardcoded SQL query"}]

    summary = run_remediation_session(
        repo_path=tmp_path,
        result=scan_res,
        console=console,
        auto_yes=True,
        rescan_fn=regressed_rescan,
    )

    out = buf.getvalue()
    assert "[REGRESSION]" in out
    assert "bandit.B608" in out
    assert summary.regressions_detected == 1
    assert summary.static_analysis_clean is False



def test_render_remediation_summary_dashboard_phase4():
    """render_remediation_summary displays table of mitigated vulnerabilities by CWE, severity, and file."""
    from sentinelai.presentation.narrator import render_remediation_summary
    from sentinelai.patcher import RemediationSessionSummary, MitigatedVulnerability, VerificationStatus

    summary = RemediationSessionSummary(
        total_findings=2,
        patches_applied=2,
        patches_rejected=0,
        vulnerabilities_avoided=2,
        modified_files=["app/db.py", "app/auth.py"],
        avoided_cwes=["CWE-89 (app/db.py)", "CWE-798 (app/auth.py)"],
        mitigated_findings=[
            MitigatedVulnerability(
                finding_id="SENT-001",
                rule_id="semgrep.sqli",
                cwe="CWE-89",
                severity=Severity.CRITICAL,
                file="app/db.py",
                scanner="semgrep",
                status=VerificationStatus.CONFIRMED_FIXED,
                message="Confirmed fixed: 0 findings",
            ),
            MitigatedVulnerability(
                finding_id="SENT-002",
                rule_id="bandit.hardcoded-secret",
                cwe="CWE-798",
                severity=Severity.HIGH,
                file="app/auth.py",
                scanner="bandit",
                status=VerificationStatus.CONFIRMED_FIXED,
                message="Confirmed fixed: 0 findings",
            ),
        ],
        static_analysis_clean=True,
    )

    buf = io.StringIO()
    console = Console(file=buf, force_terminal=True, width=120)

    render_remediation_summary(console, summary, elapsed_seconds=4.5)
    out = buf.getvalue()

    assert "SentinelAI Remediation Summary" in out
    assert "Session Results" in out
    assert "Total Findings Reviewed" in out
    assert "Patches Applied" in out
    assert "Vulnerabilities Avoided" in out
    assert "Clean (Passed)" in out
    assert "Vulnerabilities Successfully Mitigated" in out
    assert "CWE-89" in out
    assert "CWE-798" in out
    assert "CRITICAL" in out
    assert "HIGH" in out
    assert "app/db.py" in out
    assert "app/auth.py" in out
    assert "Modified Files" in out
    assert "Confirmation: Code passes static analysis clean." in out


