"""
Scanner-framework exception hierarchy.

Not a subclass of core.errors.SentinelAIError, matching the same
independence backend/loader.py already established for its own
exceptions: this hierarchy belongs to the scanner framework, not the
CLI layer, and the two are free to evolve separately.

Two categories, split by when in a scanner's lifecycle something can go
wrong: ConfigurationError for the scanner framework itself being set up
incorrectly (e.g. two scanners registered under the same name), which
happens before any scan runs; ScannerExecutionError for a concrete
scanner failing while actually scanning a repository. No scanner exists
yet to raise the execution variant - it's defined now so future scanners
have a clear, already-agreed place to raise into.
"""


class ScannerError(Exception):
    """Base class for all scanner-framework errors."""


class ConfigurationError(ScannerError):
    """The scanner framework itself is set up incorrectly (e.g. a duplicate or missing scanner registration)."""


class ScannerExecutionError(ScannerError):
    """A scanner failed while running a scan."""
