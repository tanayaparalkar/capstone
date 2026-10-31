"""Shared Pydantic contracts for the SentinelAI pipeline."""
from .ai import AI_ROLES, AIConfig, AIFault, AIRole, AIRoleProvenance, RedactionReport
from .ai_finding import (
    AIEnrichedFinding,
    ConfidenceLabel,
    GroundingVerdict,
    StructuredPatch,
    VerificationStatus,
)
from .common import Severity
from .correlated_finding import CorrelatedFinding, CorrelationRule
from .scan_result import (
    RepositoryInfo,
    RetentionStatus,
    ScanMetadata,
    ScanMode,
    ScannerTier,
    ScanResult,
    aggregate_retention_status,
    retention_status_from_provider,
)
from .scanner_finding import ScannerFinding

__all__ = [
    "AIConfig",
    "AIFault",
    "AI_ROLES",
    "AIRole",
    "AIRoleProvenance",
    "RedactionReport",
    "AIEnrichedFinding",
    "ConfidenceLabel",
    "GroundingVerdict",
    "StructuredPatch",
    "VerificationStatus",
    "Severity",
    "CorrelatedFinding",
    "CorrelationRule",
    "RepositoryInfo",
    "RetentionStatus",
    "ScanMetadata",
    "ScanMode",
    "ScannerTier",
    "ScanResult",
    "ScannerFinding",
    "aggregate_retention_status",
    "retention_status_from_provider",
]
