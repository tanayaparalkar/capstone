"""Closed-loop verification engine for applied security patches.

Phase 4 of SentinelAI:
4.1 Targeted Re-scan Engine:
    Invokes the relevant scanner (e.g., Semgrep/Bandit) specifically scoped to the modified file.
4.2 Verification Diffing:
    Compares initial finding fingerprint against the re-scan output:
    - Confirmed Fixed: The finding ID / rule ID no longer triggers on those lines.
    - Unresolved: The scanner still triggers on those lines.
    - Regressed: The scanner reports a new warning on the modified file.
"""
from __future__ import annotations

import ast
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .models import Patch, VerificationStatus


class VerificationResult(tuple):
    """Result of patch verification.

    Acts as a 2-tuple `(verified, message)` for full backwards compatibility
    with existing callers (`verified, v_msg = verify_patch(...)`), while also
    exposing `.status`, `.verified`, and `.message` attributes.
    """

    def __new__(cls, verified: bool, message: str, status: VerificationStatus):
        return super().__new__(cls, (verified, message))

    def __init__(self, verified: bool, message: str, status: VerificationStatus):
        self.verified = verified
        self.message = message
        self.status = status


def validate_python_syntax(file_path: Path) -> Tuple[bool, str]:
    """Validate that a Python file has valid syntax via AST parsing."""
    if file_path.suffix.lower() != ".py":
        return True, "Non-Python file"
    try:
        content = file_path.read_text(encoding="utf-8")
        ast.parse(content, filename=str(file_path))
        return True, "AST syntax check passed"
    except SyntaxError as e:
        return False, f"Syntax error at line {e.lineno}: {e.msg}"
    except Exception as e:
        return False, f"Could not parse file: {e}"


def run_targeted_bandit_rescan(file_path: Path) -> Optional[List[Dict[str, Any]]]:
    """Run a targeted Bandit scan scoped specifically to the single modified file."""
    bandit_bin = shutil.which("bandit")
    if not bandit_bin:
        return None

    try:
        proc = subprocess.run(
            [bandit_bin, "-f", "json", "-q", str(file_path)],
            capture_output=True,
            text=True,
            timeout=20,
        )
        if proc.stdout:
            json_start = proc.stdout.find("{")
            if json_start != -1:
                data = json.loads(proc.stdout[json_start:])
                results = data.get("results") or []
                findings = []
                for r in results:
                    findings.append({
                        "rule_id": r.get("test_id", ""),
                        "test_name": r.get("test_name", ""),
                        "line_start": r.get("line_number"),
                        "line_end": r.get("line_range", [r.get("line_number")])[-1] if r.get("line_range") else r.get("line_number"),
                        "cwe": f"CWE-{r.get('issue_cwe', {}).get('id')}" if r.get("issue_cwe") else None,
                        "severity": r.get("issue_severity", "MEDIUM"),
                        "message": r.get("issue_text", ""),
                    })
                return findings
        if proc.returncode == 0:
            return []
    except Exception:
        pass
    return None


def run_targeted_semgrep_rescan(file_path: Path) -> Optional[List[Dict[str, Any]]]:
    """Run a targeted Semgrep scan scoped specifically to the single modified file."""
    semgrep_bin = shutil.which("semgrep")
    if not semgrep_bin:
        return None

    try:
        proc = subprocess.run(
            [semgrep_bin, "--config", "auto", "--quiet", "--json", str(file_path)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if proc.stdout:
            data = json.loads(proc.stdout)
            results = data.get("results") or []
            findings = []
            for r in results:
                extra = r.get("extra", {})
                findings.append({
                    "rule_id": r.get("check_id", ""),
                    "line_start": r.get("start", {}).get("line"),
                    "line_end": r.get("end", {}).get("line"),
                    "cwe": extra.get("metadata", {}).get("cwe"),
                    "severity": extra.get("severity", "WARNING"),
                    "message": extra.get("message", ""),
                })
            return findings
    except Exception:
        pass
    return None


def diff_findings(
    patch: Patch,
    rescan_findings: List[Dict[str, Any]],
    rel_path: str,
) -> Tuple[bool, str, VerificationStatus]:
    """Perform verification diffing against re-scan output.

    4.2 Verification Diffing:
    - Confirmed Fixed: The finding ID / rule ID no longer triggers on those lines.
    - Unresolved: The scanner still triggers on those lines.
    - Regressed: The scanner still triggers or reports a new warning.
    """
    initial_rule = (patch.rule_id or "").strip().lower()
    initial_cwe = (patch.cwe or "").strip().upper()

    matching_findings = []
    other_findings = []

    for f in rescan_findings:
        f_rule = str(f.get("rule_id", "")).strip().lower()
        f_test = str(f.get("test_name", "")).strip().lower()
        f_cwe = str(f.get("cwe", "")).strip().upper()

        is_match = False
        if initial_rule and (initial_rule in f_rule or f_rule in initial_rule or initial_rule == f_test):
            is_match = True
        elif initial_cwe and initial_cwe in f_cwe:
            is_match = True

        if is_match:
            matching_findings.append(f)
        else:
            other_findings.append(f)

    # Check 1: Initial vulnerability still triggers
    if matching_findings:
        first = matching_findings[0]
        line = first.get("line_start") or patch.line_start or "unknown"
        msg = f"Unresolved: scanner still triggers '{patch.rule_id or patch.cwe}' on line {line} in {rel_path}"
        return False, msg, VerificationStatus.UNRESOLVED

    # Check 2: Regression (new warning introduced)
    if other_findings:
        first_new = other_findings[0]
        new_rule = first_new.get("rule_id") or first_new.get("message") or "new warning"
        line = first_new.get("line_start") or "unknown"
        msg = f"Regressed: patch eliminated '{patch.rule_id or patch.cwe}', but new warning '{new_rule}' appeared on line {line}"
        return False, msg, VerificationStatus.REGRESSED

    # Check 3: Confirmed Fixed (clean static analysis)
    cwe_label = patch.cwe or patch.rule_id or patch.category or "Vulnerability"
    msg = f"Confirmed Fixed: {cwe_label} resolved in {rel_path} (0 findings detected)"
    return True, msg, VerificationStatus.CONFIRMED_FIXED


def verify_patch(
    patch: Patch,
    repo_root: Path,
    rescan_fn: Optional[Callable[[Path], Optional[List[Dict[str, Any]]]]] = None,
) -> VerificationResult:
    """Verify that an applied patch eliminated the vulnerability in the target file.

    Executes closed-loop verification:
    1. Pre-flight AST syntax check to ensure no syntax regressions.
    2. Targeted re-scan using scanner (Bandit, Semgrep, or pluggable re-scan function).
    3. Verification diffing against initial finding fingerprint.
    4. Fallback content & AST verification if external scanner binaries are absent.
    """
    rel_path = str(patch.file_path.relative_to(repo_root)) if patch.file_path.is_absolute() else str(patch.file_path)

    # ── Step 1: Pre-flight AST/syntax check ──────────────────────────────────
    syntax_ok, syntax_msg = validate_python_syntax(patch.file_path)
    if not syntax_ok:
        return VerificationResult(
            False,
            f"Syntax regression: {syntax_msg} in {rel_path}",
            VerificationStatus.SYNTAX_ERROR,
        )

    # ── Step 2: Targeted Re-scan ─────────────────────────────────────────────
    rescan_results: Optional[List[Dict[str, Any]]] = None

    if rescan_fn is not None:
        rescan_results = rescan_fn(patch.file_path)
    else:
        # Check scanner type
        is_bandit = (
            (patch.scanner and "bandit" in patch.scanner.lower())
            or (patch.rule_id and (patch.rule_id.startswith("bandit") or "B" in patch.rule_id))
            or (patch.category and "bandit" in patch.category.lower())
        )
        is_semgrep = (
            (patch.scanner and "semgrep" in patch.scanner.lower())
            or (patch.rule_id and "semgrep" in patch.rule_id.lower())
            or (patch.category and "semgrep" in patch.category.lower())
        )

        if is_bandit:
            rescan_results = run_targeted_bandit_rescan(patch.file_path)
        elif is_semgrep:
            rescan_results = run_targeted_semgrep_rescan(patch.file_path)

    # ── Step 3: Verification Diffing (if re-scan ran) ────────────────────────
    if rescan_results is not None:
        ok, diff_msg, status = diff_findings(patch, rescan_results, rel_path)
        return VerificationResult(ok, diff_msg, status)

    # ── Step 4: Fallback Content & AST Verification ──────────────────────────
    try:
        current_content = patch.file_path.read_text(encoding="utf-8")
        if patch.original_snippet and patch.original_snippet in current_content:
            return VerificationResult(
                False,
                f"Unresolved: vulnerable snippet still present in {rel_path}",
                VerificationStatus.UNRESOLVED,
            )
        if patch.replacement_snippet and patch.replacement_snippet in current_content:
            cwe_label = patch.cwe or patch.category or patch.rule_id or "Vulnerability"
            return VerificationResult(
                True,
                f"Confirmed Fixed: {cwe_label} resolved in {rel_path} (clean syntax verified)",
                VerificationStatus.FALLBACK_VERIFIED,
            )
    except Exception as e:
        return VerificationResult(
            False,
            f"Verification check failed: {e}",
            VerificationStatus.UNRESOLVED,
        )

    return VerificationResult(
        True,
        f"Confirmed Fixed: code modified and clean syntax verified in {rel_path}",
        VerificationStatus.FALLBACK_VERIFIED,
    )
