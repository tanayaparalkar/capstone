"""
Scanner-level finding contract.

This is what Nitaanth's repository-analysis/scanner layer produces: raw,
deterministic output from Semgrep/Bandit/GitLeaks/Trivy/OSV Scanner,
normalized into one common shape. It carries no AI-generated content -
that lives in AIEnrichedFinding (ai_finding.py), kept as a separate model
so scanner output and AI output can each evolve independently, and a
finding can exist (and be reported on) even before AI enrichment runs.
"""
from typing import Optional

from pydantic import BaseModel, Field, model_validator

from .common import Severity


class ScannerFinding(BaseModel):
    finding_id: str = Field(..., description="Stable identifier for this finding, e.g. 'SENT-001'")
    scanner: str = Field(
        ..., description="Tool that produced this finding, e.g. 'semgrep', 'bandit', 'gitleaks', 'trivy', 'osv-scanner'"
    )
    category: str = Field(..., description="Vulnerability category, e.g. 'sql-injection', 'secret-exposure'")
    severity: Severity
    file: Optional[str] = Field(None, description="Path to the affected file, relative to the repo root")
    line_start: Optional[int] = Field(
        None, ge=1, description="First affected line, if applicable (absent for e.g. whole-dependency findings)"
    )
    line_end: Optional[int] = Field(None, ge=1, description="Last affected line, if applicable")
    rule_id: str = Field(..., description="Scanner-specific rule identifier, e.g. 'semgrep.python.sql-injection.string-concat'")
    message: str = Field(..., description="Scanner-generated description of the finding")
    raw_evidence: Optional[str] = Field(None, description="Raw code snippet or scanner output supporting this finding")
    cwe: Optional[str] = Field(None, description="CWE identifier, e.g. 'CWE-89', where the scanner/rule maps to one")

    @model_validator(mode="after")
    def _validate_line_range(self) -> "ScannerFinding":
        if self.line_end is not None and self.line_start is None:
            raise ValueError("line_end cannot be set without line_start")
        if self.line_start is not None and self.line_end is not None and self.line_end < self.line_start:
            raise ValueError("line_end cannot be before line_start")
        return self
