"""Phase 5: Comprehensive unit, integration, and end-to-end fixture tests.

Covers:
  5.1 — Unit & Integration Tests for patch components
  5.2 — End-to-End Fixture Validation against all vulnerability classes
"""
from datetime import datetime, timezone
from pathlib import Path

import io
import pytest

from rich.console import Console

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
from sentinelai.patcher import (
    MitigatedVulnerability,
    Patch,
    PatchStatus,
    RemediationSessionSummary,
    apply_patch,
    apply_patch_atomically,
    create_snapshot,
    extract_patch,
    restore_latest_backup_session,
    rollback_patch,
    run_remediation_session,
    validate_code_syntax,
    verify_patch,
)
from sentinelai.patcher.extractor import (
    _align_indentation,
    _parse_diff_lines,
    _strip_markdown_code_fences,
)
from sentinelai.patcher.rate_limit import (
    detect_repository_framework,
    get_rate_limiting_advisory,
    is_rate_limiting_relevant,
)
from sentinelai.patcher.verifier import VerificationResult, diff_findings


# ═══════════════════════════════════════════════════════════════════════════════
# 5.1 — Unit & Integration Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestDiffParsing:
    """Unit tests for extractor diff-parsing helpers."""

    def test_strip_markdown_fences_python(self):
        text = "```python\n    return hashlib.sha256(p.encode()).hexdigest()\n```"
        result = _strip_markdown_code_fences(text)
        assert "```" not in result
        assert "sha256" in result

    def test_strip_markdown_fences_no_fences(self):
        text = "    return hashlib.sha256(p.encode()).hexdigest()"
        result = _strip_markdown_code_fences(text)
        assert result == text

    def test_strip_markdown_fences_empty_lines(self):
        text = "\n\n```\ncode\n```\n\n"
        result = _strip_markdown_code_fences(text)
        assert result == "code"

    def test_parse_diff_lines_unified(self):
        diff = "--- a/file.py\n+++ b/file.py\n-    old_line\n+    new_line"
        orig, repl = _parse_diff_lines(diff)
        assert orig is not None
        assert repl is not None
        assert "old_line" in orig
        assert "new_line" in repl

    def test_parse_diff_lines_no_prefix(self):
        text = "just regular code\nnothing special"
        orig, repl = _parse_diff_lines(text)
        assert orig is None
        assert repl is None

    def test_parse_diff_lines_context_lines(self):
        diff = "-removed\n context_line\n+added"
        orig, repl = _parse_diff_lines(diff)
        assert "removed" in orig
        assert "context_line" in orig
        assert "added" in repl
        assert "context_line" in repl

    def test_align_indentation_adds_indent(self):
        orig = "    return foo"
        repl = "return bar"
        result = _align_indentation(orig, repl)
        assert result.startswith("    ")

    def test_align_indentation_preserves_existing(self):
        orig = "    return foo"
        repl = "    return bar"
        result = _align_indentation(orig, repl)
        assert result == "    return bar"


class TestRollbackBackup:
    """Integration tests for snapshot/backup/restore lifecycle."""

    def test_snapshot_creates_manifest(self, tmp_path):
        target = tmp_path / "app.py"
        target.write_text("original\n", encoding="utf-8")

        patch = Patch(
            finding_id="BAK-1",
            file_path=target,
            line_start=1,
            line_end=1,
            original_snippet="original",
            replacement_snippet="modified",
            diff="",
            explanation="test backup",
        )
        result = apply_patch_atomically(patch, repo_root=tmp_path)
        assert result.status == PatchStatus.APPLIED

        backup_dir = tmp_path / ".sentinelai" / "backups"
        assert backup_dir.exists()
        manifest = backup_dir / "backup_manifest.json"
        assert manifest.exists()

        import json
        data = json.loads(manifest.read_text(encoding="utf-8"))
        assert len(data) >= 1

    def test_restore_reverts_to_original(self, tmp_path):
        target = tmp_path / "service.py"
        original = "SECRET = 'hunter2'\n"
        target.write_text(original, encoding="utf-8")

        patch = Patch(
            finding_id="BAK-2",
            file_path=target,
            line_start=1,
            line_end=1,
            original_snippet="SECRET = 'hunter2'",
            replacement_snippet='SECRET = os.environ.get("SECRET", "")',
            diff="",
            explanation="env var",
        )
        apply_patch_atomically(patch, repo_root=tmp_path)
        assert "os.environ" in target.read_text(encoding="utf-8")

        restored = restore_latest_backup_session(tmp_path)
        assert len(restored) >= 1
        assert target.read_text(encoding="utf-8") == original

    def test_restore_empty_repo_returns_empty(self, tmp_path):
        restored = restore_latest_backup_session(tmp_path)
        assert restored == []

    def test_multiple_patches_single_restore(self, tmp_path):
        f1 = tmp_path / "a.py"
        f2 = tmp_path / "b.py"
        f1.write_text("x = 1\n", encoding="utf-8")
        f2.write_text("y = 2\n", encoding="utf-8")

        for f, old, new, fid in [(f1, "x = 1", "x = 10", "M-1"), (f2, "y = 2", "y = 20", "M-2")]:
            patch = Patch(
                finding_id=fid, file_path=f, line_start=1, line_end=1,
                original_snippet=old, replacement_snippet=new, diff="", explanation="test",
            )
            apply_patch_atomically(patch, repo_root=tmp_path)

        assert "10" in f1.read_text(encoding="utf-8")
        assert "20" in f2.read_text(encoding="utf-8")

        restored = restore_latest_backup_session(tmp_path)
        assert len(restored) >= 2


class TestPatchEdgeCases:
    """Edge-case tests for patch application mechanics."""

    def test_multiline_replacement(self, tmp_path):
        target = tmp_path / "multi.py"
        target.write_text(
            "def connect():\n"
            "    host = 'localhost'\n"
            "    port = 3306\n"
            "    return host, port\n",
            encoding="utf-8",
        )
        patch = Patch(
            finding_id="MULTI-1",
            file_path=target,
            line_start=2,
            line_end=3,
            original_snippet="    host = 'localhost'\n    port = 3306",
            replacement_snippet="    host = os.environ.get('DB_HOST', 'localhost')\n    port = int(os.environ.get('DB_PORT', '3306'))",
            diff="",
            explanation="externalize config",
        )
        result = apply_patch(patch)
        assert result.status == PatchStatus.APPLIED
        content = target.read_text(encoding="utf-8")
        assert "os.environ" in content
        assert "def connect():" in content

    def test_patch_file_not_found(self, tmp_path):
        patch = Patch(
            finding_id="NF-1",
            file_path=tmp_path / "nonexistent.py",
            line_start=1, line_end=1,
            original_snippet="x", replacement_snippet="y",
            diff="", explanation="test",
        )
        result = apply_patch(patch)
        assert result.status == PatchStatus.FAILED

    def test_patch_snippet_not_found(self, tmp_path):
        target = tmp_path / "mismatch.py"
        target.write_text("actual_content = True\n", encoding="utf-8")
        patch = Patch(
            finding_id="MIS-1",
            file_path=target,
            line_start=None, line_end=None,
            original_snippet="completely_different_content",
            replacement_snippet="replaced",
            diff="", explanation="test",
        )
        result = apply_patch(patch)
        assert result.status == PatchStatus.FAILED


# ═══════════════════════════════════════════════════════════════════════════════
# 5.1 — Template Patch Coverage for All 13 Vulnerability Classes
# ═══════════════════════════════════════════════════════════════════════════════


class TestExtractorTemplates:
    """Ensure deterministic template patches exist for every vulnerability class."""

    def _finding(self, **kw) -> ScannerFinding:
        defaults = dict(
            finding_id="TPL-1", scanner="bandit", severity=Severity.HIGH,
            rule_id="test.rule", message="test", category="test",
        )
        defaults.update(kw)
        return ScannerFinding(**defaults)

    def test_sql_injection_template(self, tmp_path):
        f = tmp_path / "db.py"
        f.write_text(
            "import sqlite3\n\n"
            "def get_user(conn, username):\n"
            "    query = \"SELECT * FROM users WHERE name = '\" + username + \"'\"\n"
            "    return conn.execute(query).fetchone()\n",
            encoding="utf-8",
        )
        finding = self._finding(file="db.py", category="sql-injection", cwe="CWE-89",
                                rule_id="semgrep.sql-injection", line_start=4, line_end=5)
        patch = extract_patch(tmp_path, finding)
        assert patch is not None
        assert "?" in patch.replacement_snippet

    def test_weak_crypto_template(self, tmp_path):
        f = tmp_path / "crypto.py"
        f.write_text(
            "import hashlib\n\ndef hash_pw(p):\n    return hashlib.md5(p.encode()).hexdigest()\n",
            encoding="utf-8",
        )
        finding = self._finding(file="crypto.py", category="weak-cryptography",
                                cwe="CWE-327", rule_id="bandit.B303", line_start=4, line_end=4)
        patch = extract_patch(tmp_path, finding)
        assert patch is not None
        assert "sha256" in patch.replacement_snippet

    def test_pickle_deserialization_template(self, tmp_path):
        f = tmp_path / "deser.py"
        f.write_text("import pickle\n\ndef load(blob):\n    return pickle.loads(blob)\n", encoding="utf-8")
        finding = self._finding(file="deser.py", category="insecure-deserialization",
                                cwe="CWE-502", rule_id="bandit.B301", line_start=4, line_end=4)
        patch = extract_patch(tmp_path, finding)
        assert patch is not None
        assert "json.loads" in patch.replacement_snippet

    def test_yaml_unsafe_load_template(self, tmp_path):
        f = tmp_path / "yconf.py"
        f.write_text("import yaml\n\ndef load_cfg(t):\n    return yaml.load(t)\n", encoding="utf-8")
        finding = self._finding(file="yconf.py", category="insecure-deserialization",
                                cwe="CWE-502", rule_id="bandit.B506", line_start=4, line_end=4)
        patch = extract_patch(tmp_path, finding)
        assert patch is not None
        assert "safe_load" in patch.replacement_snippet

    def test_subprocess_shell_true_template(self, tmp_path):
        f = tmp_path / "cmd.py"
        f.write_text("import subprocess\n\ndef backup(cmd):\n    subprocess.call(cmd, shell=True)\n", encoding="utf-8")
        finding = self._finding(file="cmd.py", category="command-injection",
                                cwe="CWE-78", rule_id="bandit.B602", line_start=4, line_end=4)
        patch = extract_patch(tmp_path, finding)
        assert patch is not None
        assert "shell=False" in patch.replacement_snippet or "shlex" in patch.replacement_snippet

    def test_os_system_concat_template(self, tmp_path):
        f = tmp_path / "ping.py"
        f.write_text(
            "import os\n\ndef ping(host):\n    os.system(\"ping -c 1 \" + host)\n",
            encoding="utf-8",
        )
        finding = self._finding(file="ping.py", category="command-injection",
                                cwe="CWE-78", rule_id="bandit.B605", line_start=4, line_end=4)
        patch = extract_patch(tmp_path, finding)
        assert patch is not None
        assert "subprocess.run" in patch.replacement_snippet

    def test_eval_template(self, tmp_path):
        f = tmp_path / "calc.py"
        f.write_text("def compute(expr):\n    return eval(expr)\n", encoding="utf-8")
        finding = self._finding(file="calc.py", category="arbitrary-code-execution",
                                cwe="CWE-94", rule_id="bandit.B307", line_start=2, line_end=2)
        patch = extract_patch(tmp_path, finding)
        assert patch is not None
        assert "ast.literal_eval" in patch.replacement_snippet

    def test_hardcoded_secret_template(self, tmp_path):
        f = tmp_path / "cfg.py"
        f.write_text("API_KEY = 'supersecret123'\n", encoding="utf-8")
        finding = self._finding(file="cfg.py", category="hardcoded-secret",
                                cwe="CWE-798", rule_id="gitleaks.generic-api-key",
                                line_start=1, line_end=1)
        patch = extract_patch(tmp_path, finding)
        assert patch is not None
        assert "os.environ" in patch.replacement_snippet

    def test_debug_mode_template(self, tmp_path):
        f = tmp_path / "settings.py"
        f.write_text("DEBUG = True\n", encoding="utf-8")
        finding = self._finding(file="settings.py", category="security-misconfiguration",
                                cwe="CWE-16", rule_id="bandit.debug-mode", line_start=1, line_end=1)
        patch = extract_patch(tmp_path, finding)
        assert patch is not None
        assert "os.environ" in patch.replacement_snippet

    def test_xss_template(self, tmp_path):
        f = tmp_path / "web.py"
        f.write_text('def greet(name):\n    return f"<h1>{name}</h1>"\n', encoding="utf-8")
        finding = self._finding(file="web.py", category="xss",
                                cwe="CWE-79", rule_id="semgrep.xss", line_start=2, line_end=2)
        patch = extract_patch(tmp_path, finding)
        assert patch is not None
        assert "html.escape" in patch.replacement_snippet

    def test_idor_template(self, tmp_path):
        f = tmp_path / "api.py"
        f.write_text(
            "def get_profile(user_id):\n"
            "    return db.session.query(Profile).filter_by(id=user_id).first()\n",
            encoding="utf-8",
        )
        finding = self._finding(file="api.py", category="broken-access-control",
                                cwe="CWE-639", rule_id="semgrep.idor", line_start=2, line_end=2)
        patch = extract_patch(tmp_path, finding)
        assert patch is not None
        assert "owner_id" in patch.replacement_snippet or "current_user" in patch.replacement_snippet

    def test_ssrf_template(self, tmp_path):
        f = tmp_path / "fetch.py"
        f.write_text(
            "import requests\n\ndef fetch(url):\n    return requests.get(url)\n",
            encoding="utf-8",
        )
        finding = self._finding(file="fetch.py", category="ssrf",
                                cwe="CWE-918", rule_id="semgrep.ssrf", line_start=4, line_end=4)
        patch = extract_patch(tmp_path, finding)
        assert patch is not None
        assert "validate" in patch.replacement_snippet.lower()

    def test_csrf_template(self, tmp_path):
        f = tmp_path / "routes.py"
        f.write_text(
            "@app.route('/transfer', methods=['POST'])\n"
            "def transfer():\n"
            "    return do_transfer()\n",
            encoding="utf-8",
        )
        finding = self._finding(file="routes.py", category="csrf",
                                cwe="CWE-352", rule_id="semgrep.csrf", line_start=1, line_end=3)
        patch = extract_patch(tmp_path, finding)
        assert patch is not None
        assert "csrf_protect" in patch.replacement_snippet


# ═══════════════════════════════════════════════════════════════════════════════
# 5.1 — Rate Limiting Advisory Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestRateLimiting:
    """Tests for rate limiting detection and advisory generation."""

    def test_detect_framework_flask(self, tmp_path):
        (tmp_path / "requirements.txt").write_text("flask==2.3.0\n", encoding="utf-8")
        assert detect_repository_framework(tmp_path) == "flask"

    def test_detect_framework_fastapi(self, tmp_path):
        (tmp_path / "requirements.txt").write_text("fastapi==0.100.0\nuvicorn\n", encoding="utf-8")
        assert detect_repository_framework(tmp_path) == "fastapi"

    def test_detect_framework_django(self, tmp_path):
        (tmp_path / "requirements.txt").write_text("django>=4.2\n", encoding="utf-8")
        assert detect_repository_framework(tmp_path) == "django"

    def test_detect_framework_express(self, tmp_path):
        (tmp_path / "package.json").write_text('{"dependencies":{"express":"^4.18"}}', encoding="utf-8")
        assert detect_repository_framework(tmp_path) == "express"

    def test_detect_framework_fallback_python(self, tmp_path):
        assert detect_repository_framework(tmp_path) == "python"

    def test_detect_framework_from_import(self, tmp_path):
        (tmp_path / "app.py").write_text("from flask import Flask\n", encoding="utf-8")
        assert detect_repository_framework(tmp_path) == "flask"

    def test_is_rate_limiting_relevant_cwe400(self):
        f = ScannerFinding(
            finding_id="RL-1", scanner="semgrep", category="dos",
            severity=Severity.HIGH, rule_id="test", message="denial", cwe="CWE-400",
        )
        assert is_rate_limiting_relevant(f) is True

    def test_is_rate_limiting_relevant_keyword(self):
        f = ScannerFinding(
            finding_id="RL-2", scanner="bandit", category="test",
            severity=Severity.MEDIUM, rule_id="test", message="brute-force attack possible",
        )
        assert is_rate_limiting_relevant(f) is True

    def test_is_rate_limiting_not_relevant(self):
        f = ScannerFinding(
            finding_id="RL-3", scanner="bandit", category="sql-injection",
            severity=Severity.HIGH, rule_id="test", message="sql injection", cwe="CWE-89",
        )
        assert is_rate_limiting_relevant(f) is False

    def test_advisory_flask(self, tmp_path):
        (tmp_path / "requirements.txt").write_text("flask\n", encoding="utf-8")
        adv = get_rate_limiting_advisory(tmp_path)
        assert adv["framework"] == "Flask"
        assert "Flask-Limiter" in adv["package"]
        assert "limiter" in adv["code_example"].lower()

    def test_advisory_fastapi(self, tmp_path):
        (tmp_path / "requirements.txt").write_text("fastapi\n", encoding="utf-8")
        adv = get_rate_limiting_advisory(tmp_path)
        assert adv["framework"] == "FastAPI"
        assert "slowapi" in adv["package"]

    def test_advisory_django(self, tmp_path):
        (tmp_path / "requirements.txt").write_text("django\n", encoding="utf-8")
        adv = get_rate_limiting_advisory(tmp_path)
        assert "Django" in adv["framework"]
        assert "ratelimit" in adv["code_example"]

    def test_advisory_express(self, tmp_path):
        (tmp_path / "package.json").write_text('{"dependencies":{"express":"*"}}', encoding="utf-8")
        adv = get_rate_limiting_advisory(tmp_path)
        assert adv["framework"] == "Express.js"
        assert "express-rate-limit" in adv["package"]

    def test_advisory_generic_python(self, tmp_path):
        adv = get_rate_limiting_advisory(tmp_path)
        assert adv["framework"] == "Python"
        assert "limits" in adv["package"]


# ═══════════════════════════════════════════════════════════════════════════════
# 5.2 — End-to-End Fixture Validation
# ═══════════════════════════════════════════════════════════════════════════════


def _run_e2e_fix(tmp_path, file_name, file_content, finding, expected_in_fixed):
    """Helper: write a vulnerable file, extract patch, apply, verify, check content."""
    target = tmp_path / file_name
    target.write_text(file_content, encoding="utf-8")

    repo_info = RepositoryInfo(
        name="fixture", path=str(tmp_path), languages=["python"],
    )
    scan_meta = ScanMetadata(
        timestamp=datetime.now(timezone.utc),
        mode=ScanMode.STANDARD,
        duration_seconds=0.1,
        scanner_tier=ScannerTier.CORE,
    )
    result = ScanResult(
        repository=repo_info,
        metadata=scan_meta,
        scanner_findings=[finding],
    )

    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False, no_color=True, width=120)

    summary = run_remediation_session(
        repo_path=tmp_path,
        result=result,
        console=console,
        auto_yes=True,
        rescan_fn=lambda path: [],
    )

    fixed_content = target.read_text(encoding="utf-8")
    out = buf.getvalue()

    assert summary.patches_applied >= 1, f"No patches applied for {file_name}"
    assert summary.vulnerabilities_avoided >= 1, f"No vulns avoided for {file_name}"
    for expected in expected_in_fixed:
        assert expected in fixed_content, (
            f"Expected '{expected}' in fixed {file_name}, got:\n{fixed_content}"
        )
    assert "[APPLIED]" in out
    assert "[AVOIDED / RESOLVED]" in out
    return summary, out


class TestE2ESQLInjection:
    """SQL injection: string-concat query → parameterized query."""

    def test_fix_db_py(self, tmp_path):
        _run_e2e_fix(
            tmp_path, "db.py",
            'import sqlite3\n\ndef get_user(conn, username):\n'
            '    query = "SELECT * FROM users WHERE name = \'" + username + "\'"\n'
            '    return conn.execute(query).fetchone()\n',
            ScannerFinding(
                finding_id="E2E-SQL-1", scanner="semgrep",
                category="sql-injection", severity=Severity.HIGH,
                file="db.py", line_start=4, line_end=5,
                rule_id="semgrep.sql-injection",
                message="SQL injection via string concatenation",
                cwe="CWE-89",
            ),
            ["?"],
        )


class TestE2ECommandInjection:
    """Command injection: os.system + subprocess shell=True."""

    def test_fix_os_system(self, tmp_path):
        _run_e2e_fix(
            tmp_path, "ping.py",
            'import os\n\ndef ping_host(host):\n    os.system("ping -c 1 " + host)\n',
            ScannerFinding(
                finding_id="E2E-CMD-1", scanner="bandit",
                category="command-injection", severity=Severity.HIGH,
                file="ping.py", line_start=4, line_end=4,
                rule_id="bandit.B605",
                message="Starting a process with a shell",
                cwe="CWE-78",
            ),
            ["subprocess.run"],
        )

    def test_fix_subprocess_shell(self, tmp_path):
        _run_e2e_fix(
            tmp_path, "backup.py",
            'import subprocess\n\ndef run_backup(cmd):\n    subprocess.call(cmd, shell=True)\n',
            ScannerFinding(
                finding_id="E2E-CMD-2", scanner="bandit",
                category="command-injection", severity=Severity.HIGH,
                file="backup.py", line_start=4, line_end=4,
                rule_id="bandit.B602",
                message="subprocess call with shell=True",
                cwe="CWE-78",
            ),
            ["shell=False"],
        )


class TestE2EDeserialization:
    """Insecure deserialization: pickle.loads → json.loads."""

    def test_fix_pickle(self, tmp_path):
        _run_e2e_fix(
            tmp_path, "deser.py",
            'import pickle\n\ndef load_cached_result(blob):\n    return pickle.loads(blob)\n',
            ScannerFinding(
                finding_id="E2E-DES-1", scanner="bandit",
                category="insecure-deserialization", severity=Severity.HIGH,
                file="deser.py", line_start=4, line_end=4,
                rule_id="bandit.B301",
                message="Deserializing untrusted data",
                cwe="CWE-502",
            ),
            ["json.loads"],
        )


class TestE2EUnsafeYAML:
    """Unsafe YAML: yaml.load → yaml.safe_load."""

    def test_fix_yaml_load(self, tmp_path):
        _run_e2e_fix(
            tmp_path, "config.py",
            'import yaml\n\ndef load_task_config(text):\n    return yaml.load(text)\n',
            ScannerFinding(
                finding_id="E2E-YAML-1", scanner="bandit",
                category="insecure-deserialization", severity=Severity.HIGH,
                file="config.py", line_start=4, line_end=4,
                rule_id="bandit.B506",
                message="Use of unsafe yaml.load",
                cwe="CWE-502",
            ),
            ["safe_load"],
        )


class TestE2EArbitraryCodeExecution:
    """Arbitrary code execution: eval → ast.literal_eval."""

    def test_fix_eval(self, tmp_path):
        _run_e2e_fix(
            tmp_path, "calc.py",
            'def evaluate_expression(expr):\n    return eval(expr)\n',
            ScannerFinding(
                finding_id="E2E-EVAL-1", scanner="bandit",
                category="arbitrary-code-execution", severity=Severity.HIGH,
                file="calc.py", line_start=2, line_end=2,
                rule_id="bandit.B307",
                message="Use of eval() detected",
                cwe="CWE-94",
            ),
            ["ast.literal_eval"],
        )


class TestE2EWeakCryptography:
    """Weak cryptography: md5 → sha256."""

    def test_fix_md5(self, tmp_path):
        _run_e2e_fix(
            tmp_path, "crypto_utils.py",
            'import hashlib\n\ndef hash_password(password):\n    return hashlib.md5(password.encode()).hexdigest()\n',
            ScannerFinding(
                finding_id="E2E-HASH-1", scanner="bandit",
                category="weak-cryptography", severity=Severity.MEDIUM,
                file="crypto_utils.py", line_start=4, line_end=4,
                rule_id="bandit.B303",
                message="Use of weak MD5 hash",
                cwe="CWE-327",
            ),
            ["sha256"],
        )


class TestE2EHardcodedSecrets:
    """Hardcoded secrets: literal API_KEY → os.environ."""

    def test_fix_hardcoded_key(self, tmp_path):
        _run_e2e_fix(
            tmp_path, "settings.py",
            'API_KEY = "0123456789abcdef0123456789abcdef"\n',
            ScannerFinding(
                finding_id="E2E-SEC-1", scanner="gitleaks",
                category="hardcoded-secret", severity=Severity.HIGH,
                file="settings.py", line_start=1, line_end=1,
                rule_id="gitleaks.generic-api-key",
                message="Hardcoded API key detected",
                cwe="CWE-798",
            ),
            ["os.environ"],
        )


class TestE2ESecurityMisconfiguration:
    """Security misconfiguration: DEBUG = True → env guard."""

    def test_fix_debug_mode(self, tmp_path):
        _run_e2e_fix(
            tmp_path, "settings.py",
            'DEBUG = True\nSERVER_NAME = "localhost"\n',
            ScannerFinding(
                finding_id="E2E-MISC-1", scanner="bandit",
                category="security-misconfiguration", severity=Severity.MEDIUM,
                file="settings.py", line_start=1, line_end=1,
                rule_id="bandit.debug-mode",
                message="DEBUG mode enabled",
                cwe="CWE-16",
            ),
            ["os.environ"],
        )


class TestE2EXSS:
    """Cross-Site Scripting: unescaped f-string → html.escape."""

    def test_fix_reflected_xss(self, tmp_path):
        _run_e2e_fix(
            tmp_path, "web.py",
            'def greet(name):\n    return f"<h1>{name}</h1>"\n',
            ScannerFinding(
                finding_id="E2E-XSS-1", scanner="semgrep",
                category="xss", severity=Severity.HIGH,
                file="web.py", line_start=2, line_end=2,
                rule_id="semgrep.xss-reflected",
                message="Reflected XSS: user input in HTML response",
                cwe="CWE-79",
            ),
            ["html.escape"],
        )


class TestE2EIDOR:
    """Broken Access Control / IDOR: unscoped query → owner_id guard."""

    def test_fix_idor(self, tmp_path):
        _run_e2e_fix(
            tmp_path, "api.py",
            'def get_profile(user_id):\n'
            '    return db.session.query(Profile).filter_by(id=user_id).first()\n',
            ScannerFinding(
                finding_id="E2E-IDOR-1", scanner="semgrep",
                category="broken-access-control", severity=Severity.HIGH,
                file="api.py", line_start=2, line_end=2,
                rule_id="semgrep.idor",
                message="Direct object reference without ownership check",
                cwe="CWE-639",
            ),
            ["owner_id"],
        )


class TestE2ESSRF:
    """Server-Side Request Forgery: raw requests.get → URL validation."""

    def test_fix_ssrf(self, tmp_path):
        _run_e2e_fix(
            tmp_path, "fetch.py",
            'import requests\n\ndef fetch_url(url):\n    return requests.get(url)\n',
            ScannerFinding(
                finding_id="E2E-SSRF-1", scanner="semgrep",
                category="ssrf", severity=Severity.HIGH,
                file="fetch.py", line_start=4, line_end=4,
                rule_id="semgrep.ssrf",
                message="Server-side request forgery",
                cwe="CWE-918",
            ),
            ["validate"],
        )


class TestE2ECSRF:
    """Cross-Site Request Forgery: POST route → @csrf_protect decorator."""

    def test_fix_csrf(self, tmp_path):
        _run_e2e_fix(
            tmp_path, "routes.py",
            "@app.route('/transfer', methods=['POST'])\n"
            "def transfer():\n"
            "    return do_transfer()\n",
            ScannerFinding(
                finding_id="E2E-CSRF-1", scanner="semgrep",
                category="csrf", severity=Severity.MEDIUM,
                file="routes.py", line_start=1, line_end=3,
                rule_id="semgrep.csrf-missing",
                message="POST endpoint without CSRF protection",
                cwe="CWE-352",
            ),
            ["csrf_protect"],
        )


class TestE2ERateLimiting:
    """Rate limiting: advisory is generated for DOS/brute-force findings."""

    def test_rate_limiting_advisory_generated(self, tmp_path):
        (tmp_path / "requirements.txt").write_text("flask==2.3.0\n", encoding="utf-8")
        (tmp_path / "app.py").write_text(
            "from flask import Flask\napp = Flask(__name__)\n"
            "@app.route('/api/data')\ndef get_data():\n    return 'ok'\n",
            encoding="utf-8",
        )
        finding = ScannerFinding(
            finding_id="E2E-RL-1", scanner="semgrep",
            category="dos", severity=Severity.HIGH,
            file="app.py", line_start=3, line_end=5,
            rule_id="semgrep.denial-of-service",
            message="API endpoint without rate limiting allows denial of service",
            cwe="CWE-400",
        )

        assert is_rate_limiting_relevant(finding)
        adv = get_rate_limiting_advisory(tmp_path, finding)
        assert adv["framework"] == "Flask"
        assert "Flask-Limiter" in adv["package"]
        assert "limiter" in adv["code_example"].lower()


# ═══════════════════════════════════════════════════════════════════════════════
# 5.2 — Verification & Session Integration
# ═══════════════════════════════════════════════════════════════════════════════


class TestVerificationIntegration:
    """Integration tests for the closed-loop verification system."""

    def test_fallback_verification_original_present(self, tmp_path):
        """Fallback: vulnerable snippet still present → UNRESOLVED."""
        from sentinelai.patcher.models import VerificationStatus as PatcherVS
        target = tmp_path / "still_vuln.py"
        target.write_text("unsafe_call(user_input)\n", encoding="utf-8")

        patch = Patch(
            finding_id="FV-1", file_path=target,
            line_start=1, line_end=1,
            original_snippet="unsafe_call(user_input)",
            replacement_snippet="safe_call(user_input)",
            diff="", explanation="safe call",
        )
        res = verify_patch(patch, tmp_path)
        assert res.verified is False
        assert res.status in (
            PatcherVS.UNRESOLVED,
            PatcherVS.SYNTAX_ERROR,
        )

    def test_fallback_verification_replacement_present(self, tmp_path):
        """Fallback: replacement present, original removed → FALLBACK_VERIFIED."""
        from sentinelai.patcher.models import VerificationStatus as PatcherVS
        target = tmp_path / "fixed.py"
        target.write_text("safe_call(user_input)\n", encoding="utf-8")

        patch = Patch(
            finding_id="FV-2", file_path=target,
            line_start=1, line_end=1,
            original_snippet="unsafe_call(user_input)",
            replacement_snippet="safe_call(user_input)",
            diff="", explanation="safe call",
        )
        res = verify_patch(patch, tmp_path)
        assert res.verified is True
        assert res.status == PatcherVS.FALLBACK_VERIFIED

    def test_session_mixed_patches(self, tmp_path):
        """Session with both fixable and unfixable findings still tracks metrics correctly."""
        fixable = tmp_path / "fix.py"
        fixable.write_text("import yaml\n\ndef load(t):\n    return yaml.load(t)\n", encoding="utf-8")

        unfixable = tmp_path / "nofix.py"
        unfixable.write_text("# No known vulnerability pattern\nx = 42\n", encoding="utf-8")

        findings = [
            ScannerFinding(
                finding_id="MIX-1", scanner="bandit",
                category="insecure-deserialization", severity=Severity.HIGH,
                file="fix.py", line_start=4, line_end=4,
                rule_id="bandit.B506", message="unsafe yaml.load", cwe="CWE-502",
            ),
            ScannerFinding(
                finding_id="MIX-2", scanner="custom",
                category="unknown-vuln", severity=Severity.LOW,
                file="nofix.py", line_start=2, line_end=2,
                rule_id="custom.unknown", message="some finding",
            ),
        ]

        result = ScanResult(
            scanner_findings=findings,
            repository=RepositoryInfo(name="mix", path=str(tmp_path), languages=["python"]),
            metadata=ScanMetadata(
                timestamp=datetime.now(timezone.utc),
                mode=ScanMode.STANDARD, duration_seconds=0.1, scanner_tier=ScannerTier.CORE,
            ),
        )

        buf = io.StringIO()
        console = Console(file=buf, force_terminal=False, no_color=True, width=120)

        summary = run_remediation_session(
            repo_path=tmp_path, result=result,
            console=console, auto_yes=True, rescan_fn=lambda p: [],
        )

        assert summary.total_findings == 2
        assert summary.patches_applied >= 1
        assert summary.patches_rejected >= 1
        assert "SentinelAI Remediation Summary" in buf.getvalue()
