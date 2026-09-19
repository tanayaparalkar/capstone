"""Interactive remediation session orchestrator.

Phase 4 of SentinelAI:
Walks through each vulnerability, renders a narrated card, prompts the user
for an interactive decision, applies patches atomically, runs closed-loop
verification with targeted re-scans, and produces the "Vulnerabilities Avoided"
summary dashboard at the end.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from rich.console import Console

from ..contracts import ScanResult
from ..core import SEVERITY_RANK, build_ai_lookup
from ..presentation.narrator import (
    prompt_user_for_fix,
    render_remediation_summary,
    render_vulnerability_narration,
)
from .applicator import apply_patch_atomically
from .backup import check_git_clean
from .extractor import extract_patch
from .models import (
    MitigatedVulnerability,
    PatchStatus,
    RemediationSessionSummary,
    VerificationStatus,
)
from .verifier import verify_patch


def run_remediation_session(
    repo_path: Path,
    result: ScanResult,
    console: Console,
    auto_yes: bool = False,
    dry_run: bool = False,
    rescan_fn: Optional[Callable[[Path], Optional[List[Dict[str, Any]]]]] = None,
) -> RemediationSessionSummary:
    """Run an interactive remediation session through all scan findings with closed-loop verification."""
    session_start = time.monotonic()
    summary = RemediationSessionSummary()
    summary.total_findings = len(result.scanner_findings)

    if not result.scanner_findings:
        console.print("[green]✅ No vulnerabilities detected. Nothing to remediate.[/green]")
        return summary

    is_clean, git_msg = check_git_clean(repo_path)
    if not is_clean:
        console.print(f"[yellow]⚠ Git notice: {git_msg}[/yellow]\n")

    ai_by_id = build_ai_lookup(result)
    findings = sorted(result.scanner_findings, key=lambda f: -SEVERITY_RANK[f.severity])
    total = len(findings)

    # ── Session Header ────────────────────────────────────────────────────
    console.print()
    console.print(
        f"[bold green]{'═' * 60}[/bold green]\n"
        f"[bold green]  🛡️  SentinelAI Interactive Remediation Agent[/bold green]\n"
        f"[bold green]  {total} {'issue' if total == 1 else 'issues'} detected — reviewing by severity (highest first)[/bold green]\n"
        f"[bold green]{'═' * 60}[/bold green]\n"
    )

    if dry_run:
        console.print("[yellow]  📋 DRY RUN MODE — no files will be modified.[/yellow]\n")

    for idx, finding in enumerate(findings, start=1):
        ai = ai_by_id.get(finding.finding_id)
        patch = extract_patch(repo_path, finding, ai)

        # ── Progress indicator ────────────────────────────────────────────
        console.print(
            f"[dim]  ─── Finding {idx} of {total} ───[/dim]"
        )

        render_vulnerability_narration(
            console=console,
            finding=finding,
            ai=ai,
            patch=patch,
            index=idx,
            total=total,
            repo_root=repo_path,
        )

        if patch is None:
            console.print(
                "[yellow]  ⚠ No automated patch available for this issue.\n"
                "    Please follow the recommended remediation advice above.[/yellow]\n"
            )
            summary.patches_rejected += 1
            continue

        decision = prompt_user_for_fix(
            console=console,
            patch=patch,
            auto_yes=auto_yes,
            dry_run=dry_run,
            finding=finding,
            ai=ai,
            repo_root=repo_path,
        )

        if decision == "q":
            console.print("\n[yellow]  ⏹ Remediation session ended by user.[/yellow]")
            break
        elif decision == "a":
            auto_yes = True
            decision = "y"

        if decision == "y":
            res = apply_patch_atomically(patch, repo_root=repo_path)
            if res.status == PatchStatus.APPLIED:
                summary.patches_applied += 1
                rel_file = str(patch.file_path.relative_to(repo_path)) if patch.file_path.is_absolute() else str(patch.file_path)
                if rel_file not in summary.modified_files:
                    summary.modified_files.append(rel_file)

                console.print(f"  [bold green]✅ [APPLIED] Code updated successfully in {rel_file}[/bold green]")
                console.print(f"  [cyan]🔍 Running closed-loop verification re-scan on {rel_file}...[/cyan]")

                # Phase 4.1 & 4.2: Closed-loop verification and diffing
                v_res = verify_patch(patch, repo_path, rescan_fn=rescan_fn)
                verified = v_res.verified
                v_msg = v_res.message
                v_status = v_res.status

                res.verified = verified
                res.verification_message = v_msg
                res.verification_status = v_status

                cwe_label = patch.cwe or patch.category or finding.rule_id

                if verified:
                    summary.vulnerabilities_avoided += 1
                    summary.avoided_cwes.append(f"{cwe_label} ({rel_file})")
                    console.print(f"  [bold green]🛡️  [AVOIDED / RESOLVED] {v_msg}[/bold green]\n")
                else:
                    if v_status == VerificationStatus.REGRESSED:
                        summary.regressions_detected += 1
                        summary.static_analysis_clean = False
                        console.print(f"  [bold red]⚠️  [REGRESSION] {v_msg}[/bold red]\n")
                    elif v_status == VerificationStatus.SYNTAX_ERROR:
                        summary.regressions_detected += 1
                        summary.static_analysis_clean = False
                        console.print(f"  [bold red]✗  [SYNTAX ERROR] {v_msg}[/bold red]\n")
                    else:
                        summary.unresolved_count += 1
                        summary.static_analysis_clean = False
                        console.print(f"  [bold yellow]⚠️  [UNRESOLVED] {v_msg}[/bold yellow]\n")

                summary.mitigated_findings.append(
                    MitigatedVulnerability(
                        finding_id=finding.finding_id,
                        rule_id=finding.rule_id,
                        cwe=finding.cwe,
                        severity=finding.severity,
                        file=rel_file,
                        scanner=finding.scanner,
                        status=v_status,
                        message=v_msg,
                    )
                )
            else:
                summary.patches_rejected += 1
                console.print(f"  [bold red]✗ [FAILED] Could not apply patch: {res.message}[/bold red]\n")
        else:
            summary.patches_rejected += 1
            console.print("  [dim]⏭ Skipped by user.[/dim]\n")

    elapsed = time.monotonic() - session_start
    render_remediation_summary(console, summary, elapsed_seconds=elapsed)
    return summary
