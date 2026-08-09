"""Foundational CLI infrastructure: exit codes, error hierarchy, centralized severity/formatting logic."""
from .errors import InvalidInputError, ProviderError, SentinelAIError
from .exit_codes import ExitCode
from .formatting import format_location, normalize_file_uri
from .severity import SEVERITY_RANK, exceeds_fail_on_threshold, filter_by_severity

__all__ = [
    "ExitCode",
    "SentinelAIError",
    "InvalidInputError",
    "ProviderError",
    "SEVERITY_RANK",
    "filter_by_severity",
    "exceeds_fail_on_threshold",
    "format_location",
    "normalize_file_uri",
]
