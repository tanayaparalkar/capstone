"""
AI provider configuration and redaction-reporting contracts.

Leaf module like the rest of contracts/: imports nothing from ai/.

AIConfig is privacy-safe by default: egress is "deny" and redaction is
on, so a config that sets nothing never sends a prompt to a remote
provider. extra="forbid" so a misspelled field fails loudly instead of
being silently dropped.

RedactionReport records how many secrets were masked and the *names*
of the patterns that matched. It must never carry a secret value, or
any fragment of one: it is written into ScanMetadata and so ends up in
every report format.
"""
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class RedactionReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    secrets_masked: int = Field(..., ge=0, description="Number of secret values masked before egress.")
    patterns_hit: list[str] = Field(
        default_factory=list,
        description="Names of the redaction patterns that matched, e.g. 'aws-access-key'. Pattern NAMES only - never secret values.",
    )
    slices_scanned: int = Field(..., ge=0, description="Number of context slices passed through redaction.")


class AIConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: Literal["ollama", "groq", "gemini", "mock"] = Field(..., description="Which LLM provider to use.")
    model: str = Field(..., description="Provider-specific model identifier.")
    egress: Literal["allow", "deny"] = Field(
        "deny", description="Whether prompts may leave the machine. Defaults to deny (fail closed)."
    )
    max_context_lines: int = Field(10, ge=0, description="Maximum source lines of context per slice.")
    max_context_chars: int = Field(2000, ge=0, description="Maximum characters of context per slice.")
    redaction_enabled: bool = Field(True, description="Whether secret redaction runs before a prompt is built.")


class AIFault(BaseModel):
    """An AI-layer operational fault recorded in scan metadata - never a security finding.

    message is only ever text that is payload-free by construction (a ProviderError or
    EgressViolation message: provider, kind, HTTP status, pattern names); for any other
    exception it is the exception type alone.
    """

    model_config = ConfigDict(extra="forbid")

    finding_id: Optional[str] = Field(None, description="Finding whose enrichment failed; None for a scan-level fault.")
    kind: str = Field(..., description="Exception type, e.g. 'ProviderError' or 'EgressViolation'.")
    message: str = Field(..., description="Payload-free description of the fault.")
