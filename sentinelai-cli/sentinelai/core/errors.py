"""
Small error hierarchy.

This exists to give CLI code specific exceptions to catch, not to
multiply exit codes - several distinct exceptions can (and do) map to
the same ExitCode. A malformed saved report and a bad CLI flag are
different exceptions but the same *kind* of outcome (INVALID_INPUT):
both mean "the request as given can't be fulfilled," fixable by the
user without any code change.
"""


class SentinelAIError(Exception):
    """Base class for all SentinelAI errors the CLI knows how to handle cleanly."""


class InvalidInputError(SentinelAIError):
    """A bad CLI argument, an unusable input/output path, or a malformed/unsupported saved report."""


class ProviderError(SentinelAIError):
    """The FindingsProvider failed to produce a scan result."""
