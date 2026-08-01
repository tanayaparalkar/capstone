"""
Rich terminal table rendering for findings.
"""
from rich.console import Console
from rich.table import Table

from .models import Finding

console = Console()

SEVERITY_STYLE = {
    "critical": "bold white on red",
    "high": "bold red",
    "medium": "bold yellow",
    "low": "bold blue",
}

CONFIDENCE_STYLE = {
    "high": "green",
    "medium": "yellow",
    "low": "dim white",
}

SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}


def render_table(findings: list[Finding]) -> None:
    ordered = sorted(findings, key=lambda f: SEVERITY_ORDER[f.severity.value])

    # Deliberately compact and single-line per row: a terminal table is for
    # scanning at a glance, not reading paragraphs. Fixed widths on the
    # short enum columns (ID/Severity/Conf.) keep them from being
    # ellipsis-truncated even in an 80-column terminal; Location and Type
    # flex with the remaining space and gracefully ellipsize if needed.
    # Full ai_description, exploit_path, and remediation belong in
    # --format markdown/html/json.
    table = Table(title="SentinelAI Findings", show_lines=False)
    table.add_column("ID", style="dim", min_width=8, no_wrap=True)
    table.add_column("Location", no_wrap=True, overflow="ellipsis")
    table.add_column("Severity", justify="center", min_width=8, no_wrap=True)
    table.add_column("Type", no_wrap=True, overflow="ellipsis")
    table.add_column("Conf.", justify="center", min_width=6, no_wrap=True)

    for f in ordered:
        sev_style = SEVERITY_STYLE.get(f.severity.value, "white")
        conf_style = CONFIDENCE_STYLE.get(f.confidence.value, "white")
        table.add_row(
            f.id,
            f"{f.file}:{f.line}",
            f"[{sev_style}]{f.severity.value.upper()}[/{sev_style}]",
            f.type,
            f"[{conf_style}]{f.confidence.value.upper()}[/{conf_style}]",
        )

    console.print(table)
    _print_summary(ordered)
    console.print(
        "[dim]Run with --format markdown/html/json (optionally -o report.md) "
        "for full details, exploit paths, and remediation.[/dim]"
    )


def _print_summary(findings: list[Finding]) -> None:
    counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    for f in findings:
        counts[f.severity.value] += 1

    parts = [
        f"[{SEVERITY_STYLE[sev]}]{sev.upper()}: {count}[/{SEVERITY_STYLE[sev]}]"
        for sev, count in counts.items()
        if count > 0
    ]
    console.print(f"\n[bold]{len(findings)} findings[/bold]   " + "   ".join(parts))
