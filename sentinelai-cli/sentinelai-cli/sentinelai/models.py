"""
Data models for SentinelAI findings.

This is the shared schema contract across the whole pipeline: Nitaanth's
scanners produce it (minus the AI fields), Tanaya's agents enrich it
(adding ai_description / exploit_path / remediation / confidence), and
this CLI (Viraj's side) consumes and renders it. Keep this file in sync
with the backend's Pydantic models — ideally it should eventually just
be imported from a shared package instead of duplicated.
"""
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Confidence(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Finding(BaseModel):
    id: str
    file: str
    line: int
    rule: str
    severity: Severity
    type: str
    ai_description: str = Field(
        ..., description="LLM-generated explanation of the vulnerability"
    )
    exploit_path: Optional[str] = Field(
        None, description="Narrative of how this finding could be exploited, if applicable"
    )
    remediation: str
    confidence: Confidence
    #feat(cli): scaffold SentinelAI CLI with mock data pipeline
