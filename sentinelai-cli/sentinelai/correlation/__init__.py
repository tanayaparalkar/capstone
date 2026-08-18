"""Deterministic cross-scanner correlation: grouping raw findings into logical issues."""
from .correlator import correlate_findings

__all__ = ["correlate_findings"]
