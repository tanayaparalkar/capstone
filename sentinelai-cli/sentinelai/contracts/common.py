"""
Shared enums used across the scanner-finding and AI-enriched-finding contracts.
"""
from enum import Enum


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
