"""
SentinelAI CLI entry point.

Phase 1: `scan` reads from the local mock_findings.json via data.load_findings().
Phase 4: swap that one function's internals for a real call to Nitaanth's
         /scan API endpoint - nothing else in this file needs to change.
"""
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from .data import load_findings
from .display import render_table
from .formatters import to_html, to_json, to_markdown
from .models import Finding

app = typer.Typer(
    name="sentinelai",
    help="SentinelAI - AI-Powered Vulnerability Detection for DevSecOps",
    add_completion=False,
)
console = Console()

SEVERITY_ORDER = {"low": 0, "medium": 1, "high": 2, "critical": 3}
VALID_FORMATS = {"table", "json", "markdown", "html"}


def _filter_by_severity(findings: list[Finding], min_severity: str) -> list[Finding]:
    threshold = SEVERITY_ORDER[min_severity]
    return [f for f in findings if SEVERITY_ORDER[f.severity.value] >= threshold]


@app.command()
def scan(
    path: str = typer.Argument(
        ".", help="Path to the repository to scan (defaults to current directory)"
    ),
    severity: Optional[str] = typer.Option(
        None,
        "--severity",
        "-s",
        help="Minimum severity to display: low, medium, high, critical",
    ),
    format: str = typer.Option(
        "table",
        "--format",
        "-f",
        help="Output format: table, json, markdown, html",
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Write report to this file instead of printing (ignored for table format)",
    ),
):
    """
    Scan a repository for vulnerabilities.

    NOTE: Phase 1 uses mock findings regardless of the path given - the
    real scanner pipeline gets wired in once the backend /scan endpoint
    is ready (Phase 4).
    """
    repo_path = Path(path)
    if not repo_path.exists():
        console.print(f"[bold red]Error:[/bold red] path '{path}' does not exist.")
        raise typer.Exit(code=1)

    format = format.lower()
    if format not in VALID_FORMATS:
        console.print(
            f"[bold red]Error:[/bold red] format must be one of "
            f"{sorted(VALID_FORMATS)}, got '{format}'."
        )
        raise typer.Exit(code=1)

    console.print(
        f"[bold cyan]Scanning[/bold cyan] {repo_path.resolve()} "
        f"[dim](using mock data \u2014 Phase 1)[/dim]"
    )

    findings = load_findings(str(repo_path))

    if severity:
        severity = severity.lower()
        if severity not in SEVERITY_ORDER:
            console.print(
                f"[bold red]Error:[/bold red] severity must be one of "
                f"{list(SEVERITY_ORDER)}, got '{severity}'."
            )
            raise typer.Exit(code=1)
        findings = _filter_by_severity(findings, severity)

    if not findings:
        console.print("[bold green]No findings matched your filters.[/bold green]")
        raise typer.Exit(code=0)

    if format == "table":
        if output:
            console.print(
                "[yellow]Note:[/yellow] --output is ignored for table format. "
                "Use --format json/markdown/html to save a report."
            )
        render_table(findings)
        return

    if format == "json":
        content = to_json(findings)
    elif format == "markdown":
        content = to_markdown(findings, str(repo_path))
    else:  # html
        content = to_html(findings, str(repo_path))

    if output:
        Path(output).write_text(content, encoding="utf-8")
        console.print(f"[bold green]Report written to[/bold green] {output}")
    else:
        console.print(content)


@app.command()
def version():
    """Show the SentinelAI CLI version."""
    from . import __version__

    console.print(f"SentinelAI CLI v{__version__} (Phase 1 \u2014 mock data)")


if __name__ == "__main__":
    app()
