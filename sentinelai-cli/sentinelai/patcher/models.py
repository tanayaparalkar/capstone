"""Data models for patch extraction, application, and verification."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional

from sentinelai.contracts import Severity


class PatchStatus(str, Enum):
    APPLIED = "applied"
    FAILED = "failed"
    SKIPPED = "skipped"
    ROLLED_BACK = "rolled_back"


class VerificationStatus(str, Enum):
    CONFIRMED_FIXED = "confirmed_fixed"
    UNRESOLVED = "unresolved"
    REGRESSED = "regressed"
    SYNTAX_ERROR = "syntax_error"
    FALLBACK_VERIFIED = "fallback_verified"


@dataclass
class MitigatedVulnerability:
    finding_id: str
    rule_id: str
    cwe: Optional[str]
    severity: Severity
    file: str
    scanner: str
    status: VerificationStatus
    message: str


@dataclass
class Patch:
    finding_id: str
    file_path: Path
    line_start: Optional[int]
    line_end: Optional[int]
    original_snippet: Optional[str]
    replacement_snippet: str
    diff: str
    explanation: str
    cwe: Optional[str] = None
    category: Optional[str] = None
    rule_id: Optional[str] = None
    scanner: Optional[str] = None
    severity: Optional[Severity] = None


@dataclass
class PatchResult:
    patch: Patch
    status: PatchStatus
    message: str
    verified: bool = False
    verification_message: str = ""
    verification_status: Optional[VerificationStatus] = None
    backup_path: Optional[Path] = None


@dataclass
class RemediationSessionSummary:
    total_findings: int = 0
    patches_applied: int = 0
    patches_rejected: int = 0
    vulnerabilities_avoided: int = 0
    regressions_detected: int = 0
    unresolved_count: int = 0
    modified_files: List[str] = field(default_factory=list)
    avoided_cwes: List[str] = field(default_factory=list)
    mitigated_findings: List[MitigatedVulnerability] = field(default_factory=list)
    static_analysis_clean: bool = True
