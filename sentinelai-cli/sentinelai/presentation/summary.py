"""
Aggregate scan summary panel.

Renders a pre-computed ScanStatistics (see sentinelai/statistics/) rather
than recalculating anything itself - the terminal summary and, in a later
milestone, JSON/Markdown/HTML reports all consume the same statistics
engine, so the numbers can never drift between output formats.
"""
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ..contracts import Severity
from ..statistics import AIEnrichmentStatus, ScanStatistics
from .console import SEVERITY_STYLE


def render_summary(console: Console, stats: ScanStatistics) -> None:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold", no_wrap=True)
    table.add_column()

    table.add_row("Total findings", str(stats.total_findings))

    severity_counts = {
        Severity.CRITICAL: stats.critical_findings,
        Severity.HIGH: stats.high_findings,
        Severity.MEDIUM: stats.medium_findings,
        Severity.LOW: stats.low_findings,
    }
    for sev, count in severity_counts.items():
        if count:
            style = SEVERITY_STYLE[sev]
            table.add_row(f"  {sev.value.upper()}", f"[{style}]{count}[/{style}]")

    if stats.scanner_counts:
        by_scanner = ", ".join(f"{name} ({count})" for name, count in sorted(stats.scanner_counts.items()))
        table.add_row("By scanner", by_scanner)

    if stats.ai_enrichment_status == AIEnrichmentStatus.UNAVAILABLE:
        table.add_row("AI enrichment", "not yet available (0 findings enriched)")
    else:
        table.add_row(
            "AI enrichment",
            f"{stats.matched_ai_findings}/{stats.total_findings} findings enriched "
            f"({stats.ai_enrichment_status.value})",
        )
        confidence_parts = [
            f"{label} ({count})"
            for label, count in (
                ("high", stats.confidence.high_count),
                ("medium", stats.confidence.medium_count),
                ("low", stats.confidence.low_count),
            )
            if count
        ]
        if confidence_parts:
            table.add_row("Confidence", ", ".join(confidence_parts))
        if stats.verification_counts:
            verif_summary = ", ".join(f"{status} ({count})" for status, count in stats.verification_counts.items())
            table.add_row("Verification", verif_summary)

    console.print(Panel(table, title="Security Summary", title_align="left", border_style="cyan"))
