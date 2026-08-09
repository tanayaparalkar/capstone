"""
Terminal presentation for scanner findings: a compact table for a full
scan, and a detailed single-finding view for `scan --details <id>`.
"""
from typing import Optional

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..contracts import AIEnrichedFinding, ScanResult, ScannerFinding
from ..core import SEVERITY_RANK, format_location
from .console import SEVERITY_BORDER_STYLE, SEVERITY_STYLE


def render_findings_table(console: Console, result: ScanResult) -> None:
    # Highest severity first: descending rank, so negate SEVERITY_RANK's
    # ascending (low->high) order rather than keeping a second, inverted
    # ranking dict here.
    findings = sorted(result.scanner_findings, key=lambda f: -SEVERITY_RANK[f.severity])
    ai_by_id = {f.finding_id: f for f in result.ai_findings}

    table = Table(title=f"SentinelAI Findings - {result.repository.name}", show_lines=False)
    table.add_column("ID", style="dim", min_width=8, no_wrap=True)
    table.add_column("Severity", justify="center", min_width=8, no_wrap=True)
    table.add_column("Location", no_wrap=True, overflow="ellipsis")
    table.add_column("Category", no_wrap=True, overflow="ellipsis")
    table.add_column("Scanner", no_wrap=True)
    table.add_column("Confidence", justify="center", min_width=10, no_wrap=True)

    for f in findings:
        sev_style = SEVERITY_STYLE.get(f.severity, "white")
        ai = ai_by_id.get(f.finding_id)
        confidence = "[bright_black]PENDING[/bright_black]" if ai is None else ai.confidence_label.value.upper()
        table.add_row(
            f.finding_id,
            f"[{sev_style}]{f.severity.value.upper()}[/{sev_style}]",
            format_location(f),
            f.category,
            f.scanner,
            confidence,
        )

    console.print(table)


def render_finding_detail(console: Console, finding: ScannerFinding, ai: Optional[AIEnrichedFinding]) -> None:
    """Full detail for one finding - only shown on explicit request (scan --details <id>)."""
    sev_style = SEVERITY_STYLE.get(finding.severity, "white")

    header = Table.grid(padding=(0, 2))
    header.add_column(style="bold cyan", no_wrap=True)
    header.add_column()
    header.add_row("Severity", f"[{sev_style}]{finding.severity.value.upper()}[/{sev_style}]")
    header.add_row("Scanner", finding.scanner)
    header.add_row("Rule", finding.rule_id)
    header.add_row("Location", format_location(finding))
    if finding.cwe:
        header.add_row("CWE", finding.cwe)

    sections = [header, "", Text("Description", style="bold"), Text(finding.message)]

    if finding.raw_evidence:
        sections += ["", Text("Evidence", style="bold"), Text(finding.raw_evidence)]

    if ai is not None:
        sections += ["", Text("AI Explanation", style="bold"), Text(ai.explanation)]
        if ai.exploit_path:
            sections += ["", Text("Exploit Path", style="bold"), Text(ai.exploit_path)]
        if ai.impact:
            sections += ["", Text("Impact", style="bold"), Text(ai.impact)]
        sections += ["", Text("Remediation", style="bold"), Text(ai.remediation)]
        sections += [
            "",
            Text(
                f"Confidence: {ai.confidence_label.value.upper()} ({ai.confidence_score:.2f})   "
                f"Verification: {ai.verification_status.value}",
                style="dim",
            ),
        ]
    else:
        sections += ["", Text("AI enrichment not yet available for this finding.", style="dim italic")]

    console.print(
        Panel(
            Group(*sections),
            title=f"{finding.finding_id} - {finding.category}",
            title_align="left",
            border_style=SEVERITY_BORDER_STYLE.get(finding.severity, "white"),
        )
    )
