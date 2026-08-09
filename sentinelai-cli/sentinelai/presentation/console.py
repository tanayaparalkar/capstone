"""
Centralized Rich presentation configuration.

Console construction and severity styling live here so every module in
sentinelai/presentation/ stays visually consistent, and so CI-safety
settings are defined in exactly one place instead of per-module. Rich
already auto-disables color/animation when stdout isn't a terminal (CI,
redirected output) - the one thing it doesn't do automatically is honor
the NO_COLOR convention when attached to a real terminal, which this
adds explicitly for accessibility.

Two consoles, two streams: get_console() (stdout) renders the terminal
UI and successful structured output; get_error_console() (stderr) is
where print_error() always writes. This means an error can never share
stdout with structured report content (--format json/markdown/html/
sarif) - a script piping stdout into a JSON parser is unaffected by an
error message, which shows up on stderr instead.
"""
import os
from typing import Optional

from rich.console import Console

from ..contracts import Severity

_console: Optional[Console] = None
_error_console: Optional[Console] = None


def get_console() -> Console:
    """Return the shared stdout Console, configured once."""
    global _console
    if _console is None:
        _console = Console(no_color=_no_color_requested(), highlight=False)
    return _console


def get_error_console() -> Console:
    """Return the shared stderr Console, configured once. Used for errors/diagnostics only."""
    global _error_console
    if _error_console is None:
        _error_console = Console(stderr=True, no_color=_no_color_requested(), highlight=False)
    return _error_console


def _no_color_requested() -> bool:
    # https://no-color.org/
    return os.environ.get("NO_COLOR") is not None


def print_error(message: str) -> None:
    get_error_console().print(f"[bold red]Error:[/bold red] {message}")


def print_traceback() -> None:
    """Print the currently-handled exception's traceback to stderr - call only from an except block."""
    get_error_console().print_exception(show_locals=False)


# Shared across findings.py / summary.py so severity colors never drift
# out of sync between the table, the summary panel, and the detail view.
SEVERITY_STYLE = {
    Severity.CRITICAL: "bold white on red",
    Severity.HIGH: "bold red",
    Severity.MEDIUM: "bold yellow",
    Severity.LOW: "bold blue",
}

SEVERITY_BORDER_STYLE = {
    Severity.CRITICAL: "red",
    Severity.HIGH: "red",
    Severity.MEDIUM: "yellow",
    Severity.LOW: "blue",
}
