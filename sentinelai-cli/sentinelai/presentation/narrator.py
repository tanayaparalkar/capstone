"""Interactive vulnerability narrator and remediation UI components.

Phase 3 of the SentinelAI polishing plan: transforms the remediation CLI
from a functional but plain output into a rich, interactive experience.

Each vulnerability is presented as a styled "card" with:
- Threat Narrative (root cause, affected lines, exploit risk)
- Remediation Plan (proposed fix rationale and test guidance)
- Visual Diff (syntax-highlighted unified diff: before vs. after)

The interactive prompt supports [y]es / [n]o / [a]ll / [s]kip / [d]etails / [q]uit.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

from rich.columns import Columns
from rich.console import Console, Group
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from ..contracts import AIEnrichedFinding, ScannerFinding, Severity
from ..core import format_location
from .console import SEVERITY_BORDER_STYLE, SEVERITY_STYLE

if TYPE_CHECKING:
    from ..patcher.models import Patch, RemediationSessionSummary


# ─── Severity Icons ──────────────────────────────────────────────────────────
SEVERITY_ICON = {
    Severity.CRITICAL: "🔴",
    Severity.HIGH: "🟠",
    Severity.MEDIUM: "🟡",
    Severity.LOW: "🔵",
}


def _confidence_badge(ai: Optional[AIEnrichedFinding]) -> str:
    """Return a Rich markup badge string for the AI confidence level."""
    if ai is None:
        return ""
    score = ai.confidence_score
    label = ai.confidence_label.value.upper()
    if score >= 0.8:
        color = "bold green"
    elif score >= 0.5:
        color = "bold yellow"
    else:
        color = "bold red"
    return f"[{color}]⬤ {label} ({score:.0%})[/{color}]"


def _source_context_panel(
    finding: ScannerFinding,
    repo_root: Optional[Path] = None,
) -> Optional[Syntax]:
    """Read the affected source file and return a syntax-highlighted snippet."""
    if not finding.file or not finding.line_start:
        return None

    target = Path(finding.file)
    if repo_root:
        target = repo_root / finding.file

    if not target.exists() or not target.is_file():
        return None

    try:
        content = target.read_text(encoding="utf-8")
    except Exception:
        return None

    # Show 3 lines of context above and below the affected region
    start = max(1, finding.line_start - 3)
    end = (finding.line_end or finding.line_start) + 3
    ext = target.suffix.lstrip(".")
    lang_map = {"py": "python", "js": "javascript", "ts": "typescript", "rb": "ruby", "go": "go", "java": "java", "rs": "rust", "c": "c", "cpp": "cpp", "cs": "csharp"}
    lang = lang_map.get(ext, ext or "text")

    return Syntax(
        content,
        lang,
        theme="monokai",
        line_numbers=True,
        line_range=(start, end),
        highlight_lines=set(range(finding.line_start, (finding.line_end or finding.line_start) + 1)),
    )


def render_vulnerability_narration(
    console: Console,
    finding: ScannerFinding,
    ai: Optional[AIEnrichedFinding],
    patch: Optional[Patch],
    index: int = 1,
    total: int = 1,
    repo_root: Optional[Path] = None,
) -> None:
    """Narrate a vulnerability with rich styling, threat explanation, and diff preview."""
    sev = finding.severity
    sev_style = SEVERITY_STYLE.get(sev, "white")
    border_style = SEVERITY_BORDER_STYLE.get(sev, "white")
    icon = SEVERITY_ICON.get(sev, "⚪")

    sections: list = []

    # ── Metadata Table ────────────────────────────────────────────────────
    meta = Table.grid(padding=(0, 2))
    meta.add_column(style="bold cyan", no_wrap=True, min_width=18)
    meta.add_column()
    meta.add_row("Severity", f"{icon} [{sev_style}]{sev.value.upper()}[/{sev_style}]")
    meta.add_row("Location", f"[bold]{format_location(finding)}[/bold]")
    meta.add_row("Scanner / Rule", f"{finding.scanner} → [dim]{finding.rule_id}[/dim]")
    if finding.cwe:
        meta.add_row("CWE", f"[bold yellow]{finding.cwe}[/bold yellow]")
    confidence = _confidence_badge(ai)
    if confidence:
        meta.add_row("AI Confidence", confidence)
    if ai and ai.verification_status:
        vs = ai.verification_status.value.upper()
        vs_color = "green" if vs == "VERIFIED" else "yellow" if vs == "UNVERIFIED" else "red"
        meta.add_row("Verification", f"[{vs_color}]{vs}[/{vs_color}]")

    sections.append(meta)
    sections.append("")

    # ── Source Code Context ───────────────────────────────────────────────
    source_panel = _source_context_panel(finding, repo_root=repo_root)
    if source_panel is not None:
        sections.append(Text("📄 Affected Source Code:", style="bold"))
        sections.append(source_panel)
        sections.append("")

    # ── Threat Narrative ──────────────────────────────────────────────────
    sections.append(Rule("Threat Narrative & Root Cause", style="yellow"))
    if ai and ai.explanation:
        sections.append(Text(ai.explanation))
    else:
        sections.append(Text(finding.message))

    # ── Exploit Path & Impact ─────────────────────────────────────────────
    if ai and (ai.exploit_path or ai.impact):
        sections.append("")
        if ai.exploit_path:
            sections.append(Text("⚡ Exploit Scenario:", style="bold red"))
            sections.append(Text(f"  {ai.exploit_path}"))
        if ai.impact:
            sections.append(Text("💥 Potential Impact:", style="bold magenta"))
            sections.append(Text(f"  {ai.impact}"))

    # ── Remediation Guidance ──────────────────────────────────────────────
    sections.append("")
    sections.append(Rule("Recommended Remediation", style="green"))
    if patch and patch.explanation:
        sections.append(Text(f"🔧 {patch.explanation}"))
    elif ai and ai.remediation:
        sections.append(Text(f"🔧 {ai.remediation}"))
    else:
        sections.append(Text("🔧 Review the affected code and validate user inputs."))

    # ── Visual Diff Preview ───────────────────────────────────────────────
    if patch and patch.diff:
        sections.append("")
        sections.append(Rule("Proposed Code Change (Diff)", style="cyan"))
        diff_syntax = Syntax(patch.diff, "diff", theme="monokai", line_numbers=False)
        sections.append(diff_syntax)

    # ── Rate Limiting Advisory (DoS / API endpoint resiliency) ────────────
    if repo_root is not None:
        try:
            from ..patcher.rate_limit import get_rate_limiting_advisory, is_rate_limiting_relevant
            if is_rate_limiting_relevant(finding):
                adv = get_rate_limiting_advisory(repo_root, finding)
                sections.append("")
                sections.append(Rule("🛡️ API Resiliency & Rate Limiting Advisory", style="bold yellow"))
                sections.append(Text(f"Detected Framework: {adv['framework']} | Recommended Package: {adv['package']}", style="bold cyan"))
                sections.append(Text("⚡ Risk: Unprotected API endpoints allow attackers to exhaust server resources and take the server down (DoS / CWE-400 / CWE-770).", style="yellow"))
                sections.append(Text(f"📦 Install: {adv['install']}", style="dim"))
                sections.append(Syntax(adv["code_example"], "python" if adv["framework"] != "Express.js" else "javascript", theme="monokai", line_numbers=False))
        except Exception:
            pass

    # ── Panel Title ───────────────────────────────────────────────────────
    title = f"{icon} [{index}/{total}] {finding.category.upper()} ({finding.finding_id})"
    subtitle = f"File: {finding.file or 'unknown'}"

    console.print()
    console.print(
        Panel(
            Group(*sections),
            title=title,
            title_align="left",
            subtitle=subtitle,
            subtitle_align="right",
            border_style=border_style,
            padding=(1, 2),
        )
    )


def _render_details_expansion(
    console: Console,
    finding: ScannerFinding,
    ai: Optional[AIEnrichedFinding],
) -> None:
    """Render expanded AI analysis details when user presses [d]."""
    sections: list = []

    sections.append(Rule("Extended Analysis Details", style="bright_magenta"))

    if ai:
        if ai.evidence:
            sections.append(Text("🔍 Evidence:", style="bold"))
            sections.append(Text(f"  {ai.evidence}"))
            sections.append("")

        if ai.repository_context:
            sections.append(Text("📂 Repository Context:", style="bold"))
            sections.append(Text(f"  {ai.repository_context}"))
            sections.append("")

        if ai.grounding_verdict:
            gv = ai.grounding_verdict.value.upper()
            gv_color = "green" if gv == "SUPPORTED" else "yellow"
            sections.append(Text(f"📋 Grounding Verdict: [{gv_color}]{gv}[/{gv_color}]"))

        if ai.supported_claims:
            sections.append(Text("✅ Supported Claims:", style="bold green"))
            for claim in ai.supported_claims:
                sections.append(Text(f"   • {claim}"))
        if ai.unsupported_claims:
            sections.append(Text("❌ Unsupported Claims:", style="bold red"))
            for claim in ai.unsupported_claims:
                sections.append(Text(f"   • {claim}"))

        if ai.references:
            sections.append("")
            sections.append(Text("📚 References:", style="bold"))
            for ref in ai.references:
                sections.append(Text(f"   → {ref}"))

        if ai.related_findings:
            sections.append("")
            sections.append(Text(f"🔗 Related Findings: {', '.join(ai.related_findings)}", style="bold cyan"))
    else:
        sections.append(Text("[dim]No AI-enriched analysis available for this finding.[/dim]"))

    if finding.raw_evidence:
        sections.append("")
        sections.append(Text("📝 Raw Scanner Evidence:", style="bold"))
        sections.append(Syntax(finding.raw_evidence, "python", theme="monokai", line_numbers=False))

    console.print(
        Panel(
            Group(*sections),
            border_style="bright_magenta",
            padding=(1, 2),
        )
    )


def _render_rate_limit_modal(console: Console, repo_root: Path, finding: Optional[ScannerFinding]) -> None:
    """Render framework-tailored rate limiting guidance when user presses [r]."""
    from ..patcher.rate_limit import get_rate_limiting_advisory
    adv = get_rate_limiting_advisory(repo_root, finding)

    sections = [
        Text(f"Framework Detected: {adv['framework']}", style="bold green"),
        Text(f"Summary: {adv['summary']}\n", style="italic"),
        Text(f"1. Install Dependency:", style="bold"),
        Text(f"   {adv['install']}\n", style="cyan"),
        Text(f"2. Recommended Configuration:", style="bold"),
        Syntax(adv["code_example"], "python" if adv["framework"] != "Express.js" else "javascript", theme="monokai", line_numbers=False),
    ]

    console.print(
        Panel(
            Group(*sections),
            title=f"🛡️ Rate Limiting Guidance for {adv['framework']}",
            border_style="yellow",
            padding=(1, 2),
        )
    )


def prompt_user_for_fix(
    console: Console,
    patch: Patch,
    auto_yes: bool = False,
    dry_run: bool = False,
    finding: Optional[ScannerFinding] = None,
    ai: Optional[AIEnrichedFinding] = None,
    repo_root: Optional[Path] = None,
) -> str:
    """Prompt user interactively in CLI for permission to apply a recommended fix.

    Returns one of: 'y', 'n', 'a', 'q'.
    The [d]etails option shows expanded analysis and then re-prompts.
    The [r]ate-limit option displays framework-tailored rate limiting configuration.
    The [s]kip option is a semantic alias for 'n'.
    """
    if dry_run:
        console.print("[yellow]  ⏭ [DRY RUN] Skipping code modification.[/yellow]")
        return "n"

    if auto_yes:
        console.print("[dim]  ✓ Auto-approving patch (--yes flag active)...[/dim]")
        return "y"

    filename = patch.file_path.name
    show_rate_limit = False
    if finding and repo_root:
        try:
            from ..patcher.rate_limit import is_rate_limiting_relevant
            show_rate_limit = is_rate_limiting_relevant(finding)
        except Exception:
            pass

    while True:
        prompt_parts = [
            f"\n  [bold]Apply fix to [cyan]{filename}[/cyan]?[/bold] ",
            "[green][y][/green]es  ",
            "[yellow][n][/yellow]o  ",
            "[cyan][a][/cyan]ll  ",
            "[dim][s][/dim]kip  ",
            "[magenta][d][/magenta]etails  ",
        ]
        if show_rate_limit:
            prompt_parts.append("[blue][r][/blue]ate-limit  ")
        prompt_parts.append("[red][q][/red]uit")

        console.print("".join(prompt_parts), end="  ")

        choices = ["y", "n", "a", "s", "d", "q", "yes", "no", "all", "skip", "details", "quit"]
        if show_rate_limit:
            choices.extend(["r", "rate-limit", "ratelimit"])

        choice = Prompt.ask(
            "",
            choices=choices,
            default="y",
            show_choices=False,
        ).strip().lower()

        if choice in ("d", "details"):
            _render_details_expansion(console, finding, ai) if finding else console.print("[dim]No additional details available.[/dim]")
            continue  # Re-prompt after showing details
        elif choice in ("r", "rate-limit", "ratelimit") and repo_root:
            _render_rate_limit_modal(console, repo_root, finding)
            continue  # Re-prompt after showing rate limiting
        elif choice in ("y", "yes"):
            return "y"
        elif choice in ("a", "all"):
            return "a"
        elif choice in ("q", "quit"):
            return "q"
        else:  # n, no, s, skip
            return "n"


def render_remediation_summary(
    console: Console,
    summary: RemediationSessionSummary,
    elapsed_seconds: float = 0.0,
) -> None:
    """Render the final summary dashboard of fixes applied and vulnerabilities avoided.

    Phase 4:
    - Total issues reviewed
    - Total patches applied
    - List of mitigated vulnerabilities by CWE, severity, and file
    - Confirmation that code passes static analysis clean
    """
    console.print()
    console.print(Rule("SentinelAI Remediation Summary", style="bold green"))
    console.print()

    # ── Stats Table ───────────────────────────────────────────────────────
    stats_table = Table(
        show_header=True,
        header_style="bold",
        show_lines=True,
        border_style="green",
        title="Session Results",
        title_style="bold green",
        min_width=50,
    )
    stats_table.add_column("Metric", style="bold", min_width=30)
    stats_table.add_column("Value", style="cyan", justify="right", min_width=15)

    stats_table.add_row("Total Findings Reviewed", str(summary.total_findings))
    stats_table.add_row(
        "Patches Applied",
        f"[bold green]{summary.patches_applied}[/bold green]",
    )
    stats_table.add_row(
        "Patches Skipped / Rejected",
        f"[yellow]{summary.patches_rejected}[/yellow]",
    )
    stats_table.add_row(
        "✅ Vulnerabilities Avoided",
        f"[bold green]{summary.vulnerabilities_avoided}[/bold green]",
    )

    # Static analysis clean check
    if summary.vulnerabilities_avoided > 0 and getattr(summary, "static_analysis_clean", True):
        stats_table.add_row("Static Analysis Verification", "[bold green]✅ Clean (Passed)[/bold green]")
    elif getattr(summary, "regressions_detected", 0) > 0:
        stats_table.add_row("Static Analysis Verification", "[bold red]⚠️ Regression Detected[/bold red]")
    elif getattr(summary, "unresolved_count", 0) > 0:
        stats_table.add_row("Static Analysis Verification", "[bold yellow]⚠️ Unresolved Issues[/bold yellow]")

    if elapsed_seconds > 0:
        mins, secs = divmod(int(elapsed_seconds), 60)
        time_str = f"{mins}m {secs}s" if mins else f"{secs}s"
        stats_table.add_row("Session Duration", f"[dim]{time_str}[/dim]")

    console.print(stats_table)

    # ── Modified Files ────────────────────────────────────────────────────
    if summary.modified_files:
        console.print()
        files_table = Table(
            title="Modified Files",
            title_style="bold cyan",
            show_lines=True,
            border_style="cyan",
            min_width=50,
        )
        files_table.add_column("#", style="dim", width=4)
        files_table.add_column("File Path", style="bold")
        for i, f in enumerate(summary.modified_files, 1):
            files_table.add_row(str(i), f)
        console.print(files_table)

    # ── Mitigated Vulnerabilities by CWE, Severity, and File ──────────────
    mitigated_list = getattr(summary, "mitigated_findings", [])
    if mitigated_list:
        console.print()
        mit_table = Table(
            title="Vulnerabilities Successfully Mitigated",
            title_style="bold green",
            show_lines=True,
            border_style="green",
            min_width=60,
        )
        mit_table.add_column("Status", style="bold green", width=10)
        mit_table.add_column("Vulnerability / CWE", style="bold")
        mit_table.add_column("Severity", width=14)
        mit_table.add_column("File", style="cyan")
        mit_table.add_column("Scanner / Rule", style="dim")

        for mf in mitigated_list:
            sev = mf.severity
            sev_style = SEVERITY_STYLE.get(sev, "white")
            sev_icon = SEVERITY_ICON.get(sev, "⚪")
            sev_cell = f"{sev_icon} [{sev_style}]{sev.value.upper()}[/{sev_style}]"

            status_str = str(getattr(mf.status, "value", mf.status)).lower()
            if "fixed" in status_str or "verified" in status_str:
                status_cell = "[bold green]✅ FIXED[/bold green]"
            elif "regressed" in status_str:
                status_cell = "[bold red]⚠️ REGRESSED[/bold red]"
            else:
                status_cell = "[bold yellow]⚠️ UNRESOLVED[/bold yellow]"

            cwe_display = mf.cwe or mf.rule_id
            mit_table.add_row(status_cell, cwe_display, sev_cell, mf.file, f"{mf.scanner} → {mf.rule_id}")

        console.print(mit_table)
    elif summary.avoided_cwes:
        console.print()
        cwe_table = Table(
            title="Vulnerabilities Successfully Mitigated",
            title_style="bold green",
            show_lines=True,
            border_style="green",
            min_width=50,
        )
        cwe_table.add_column("Status", style="bold green", width=8)
        cwe_table.add_column("Vulnerability / CWE", style="bold")
        for cwe in sorted(set(summary.avoided_cwes)):
            cwe_table.add_row("✅ FIXED", cwe)
        console.print(cwe_table)

    # ── Final verdict ─────────────────────────────────────────────────────
    console.print()
    if summary.patches_applied > 0 and summary.vulnerabilities_avoided > 0:
        clean_note = (
            "\n[bold green]✓ Confirmation: Code passes static analysis clean.[/bold green]"
            if getattr(summary, "static_analysis_clean", True)
            else ""
        )
        console.print(
            Panel(
                f"[bold green]🛡️  {summary.vulnerabilities_avoided} "
                f"{'vulnerability' if summary.vulnerabilities_avoided == 1 else 'vulnerabilities'} "
                f"successfully remediated and verified.[/bold green]{clean_note}\n"
                f"[dim]Run [bold]sentinelai undo[/bold] to revert all changes if needed.[/dim]",
                border_style="green",
                padding=(1, 2),
            )
        )
    elif summary.patches_applied > 0:
        console.print(
            Panel(
                f"[bold yellow]⚠️  {summary.patches_applied} "
                f"{'patch' if summary.patches_applied == 1 else 'patches'} applied. "
                f"Run [bold]sentinelai undo[/bold] to revert changes if needed.[/bold yellow]",
                border_style="yellow",
                padding=(1, 2),
            )
        )
    else:
        console.print(
            Panel(
                "[dim]No patches were applied during this session.[/dim]",
                border_style="dim",
                padding=(1, 2),
            )
        )

