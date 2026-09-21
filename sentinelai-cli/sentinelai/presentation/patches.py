"""
Terminal rendering of the deterministic patch stage.

Presentation only, and the same shape as the other renderers: the rows come from
reporting/patch_section.py, so the terminal cannot show different counts from the
Markdown, HTML or JSON views of the same scan.

Rendered as its own section rather than folded into the findings table. Patch
application is a separate deterministic stage that runs after enrichment, and a
row mixing "what the scanner found" with "what was written to disk" would make
the report ambiguous about which part of the system made which claim.
"""
from typing import Optional

from rich.console import Console
from rich.table import Table

from ..reporting.models import PatchApplicationReport
from ..reporting.patch_section import (
    ATTEMPT_COLUMNS,
    attempt_rows,
    mode_notice,
    summary_rows,
)


def render_patch_application(console: Console, report: Optional[PatchApplicationReport]) -> None:
    """Render the patch section, or nothing at all when patching was not requested."""
    if report is None:
        return

    console.print()
    heading = "Patch Application" + (" - DRY RUN" if report.dry_run else "")
    console.print(f"[bold]{heading}[/bold]")
    # Styled to stand out: a reader scanning the table below must not take an
    # "applied" row as evidence that a file changed.
    style = "yellow" if report.dry_run else "dim"
    console.print(f"[{style}]{mode_notice(report)}[/{style}]")

    summary = Table(show_header=True, header_style="bold")
    summary.add_column("Metric")
    summary.add_column("Count", justify="right")
    for label, count in summary_rows(report):
        summary.add_row(label, str(count))
    console.print(summary)

    rows = attempt_rows(report)
    if not rows:
        console.print("No findings were considered for patch application.")
        return

    attempts = Table(show_header=True, header_style="bold")
    for column in ATTEMPT_COLUMNS:
        attempts.add_column(column, overflow="fold")
    for row in rows:
        attempts.add_row(*row)
    console.print(attempts)
