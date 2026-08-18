"""
Scan progress reporting.

Wraps a small reusable "stage" concept around plain Rich console output.
Today MockFindingsProvider performs its work as a single opaque function
call - it does not report "repository analysis", "static analysis", "AI
reasoning", "verification", or "report generation" as discrete events, so
the CLI only drives this component through the one stage it can honestly
account for: retrieving the ScanResult from the provider. Once a real
provider exposes discrete stage callbacks, the CLI can call `.stage(...)`
once per real event it receives instead of wrapping one call in one
stage - nothing about this component needs to change for that to work.

Deliberately no animated spinner: Rich's Status/Progress live-rendering
is a no-op when stdout isn't a terminal (CI, redirected output), and an
animation is exactly the kind of noise this CLI avoids. Each stage prints
a start line and a completion (or failure) line - simple, deterministic,
and identical whether or not the terminal is interactive.

Deliberately plain ASCII text rather than Unicode glyphs (checkmarks,
arrows): on a legacy Windows console (no ANSI/VT support - still common
outside Windows Terminal), Rich falls back to the raw Win32 console API,
which encodes styled text using the console codepage (e.g. cp1252) and
raises rather than substituting on anything outside it. Plain text avoids
that failure mode entirely instead of depending on terminal/font support.
"""
from contextlib import contextmanager
from typing import Iterator

from rich.console import Console


class ScanProgress:
    def __init__(self, console: Console) -> None:
        self._console = console

    @contextmanager
    def stage(self, label: str) -> Iterator[None]:
        """Run one real unit of work, announced honestly as `label`."""
        self._console.print(f"[cyan]{label}...[/cyan]")
        try:
            yield
        except Exception:
            self._console.print(f"[bold red]{label} - failed[/bold red]")
            raise
        else:
            self._console.print(f"[green]{label} - done[/green]")
