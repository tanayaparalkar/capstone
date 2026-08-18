"""
Scan header: the first thing printed for a terminal-format scan, showing
what's about to run before any findings are retrieved.
"""
from datetime import datetime

from rich.console import Console
from rich.panel import Panel
from rich.table import Table


def render_scan_header(console: Console, *, repository: str, mode: str, version: str, started_at: datetime) -> None:
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bold cyan", no_wrap=True)
    grid.add_column()
    grid.add_row("Repository", repository)
    grid.add_row("Mode", mode)
    grid.add_row("Version", version)
    grid.add_row("Started", started_at.strftime("%Y-%m-%d %H:%M UTC"))

    console.print(Panel(grid, title="SentinelAI Security Scan", title_align="left", border_style="cyan"))
