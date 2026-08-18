"""Foundational CLI infrastructure: exit codes, error hierarchy, centralized severity/formatting logic."""
from .errors import AIEnrichmentError, InvalidInputError, ProviderError, SentinelAIError
from .exit_codes import ExitCode
from .formatting import build_ai_lookup, build_correlation_lookup, format_location, normalize_file_uri
from .severity import SEVERITY_RANK, exceeds_fail_on_threshold, filter_by_severity

__all__ = [
    "ExitCode",
    "SentinelAIError",
    "InvalidInputError",
    "ProviderError",
    "AIEnrichmentError",
    "SEVERITY_RANK",
    "filter_by_severity",
    "exceeds_fail_on_threshold",
    "format_location",
    "build_ai_lookup",
    "build_correlation_lookup",
    "normalize_file_uri",
]
