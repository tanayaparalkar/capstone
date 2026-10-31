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

# The three AI pipeline roles. Kept here (a leaf module) so AIConfig, provenance and
# ai/llm/roles.py agree on the spelling without contracts/ importing from ai/.
AIRole = Literal["discovery", "remediation", "verification"]
AI_ROLES: tuple[str, ...] = ("discovery", "remediation", "verification")


class RedactionReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    secrets_masked: int = Field(..., ge=0, description="Number of secret values masked before egress.")
    patterns_hit: list[str] = Field(
        default_factory=list,
        description="Names of the redaction patterns that matched, e.g. 'aws-access-key'. Pattern NAMES only - never secret values.",
    )
    slices_scanned: int = Field(
        ...,
        ge=0,
        description=(
            "Number of inputs passed through redaction. Counts prompt context slices and, since "
            "the embedding path was secured, every text embedded as well - both retrieval queries "
            "and the bundled knowledge-base entries indexed at startup. It is a count of redaction "
            "passes, not a count of repository context slices, so it is not a measure of how much "
            "source code was read."
        ),
    )


class AIConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: Literal["ollama", "groq", "gemini", "mock", "grok", "nvidia"] = Field(
        ..., description="Which LLM provider to use (the discovery provider when role_chains is set)."
    )
    model: str = Field(..., description="Provider-specific model identifier.")
    egress: Literal["allow", "deny"] = Field(
        "deny", description="Whether prompts may leave the machine. Defaults to deny (fail closed)."
    )
    max_context_lines: int = Field(10, ge=0, description="Maximum source lines of context per slice.")
    max_context_chars: int = Field(2000, ge=0, description="Maximum characters of context per slice.")
    redaction_enabled: bool = Field(True, description="Whether secret redaction runs before a prompt is built.")
    fallback_providers: list[str] = Field(
        default_factory=list, description="Ordered list of fallback providers to attempt on operational failure."
    )
    role_chains: dict[AIRole, list[str]] = Field(
        default_factory=dict,
        description=(
            "Per-role ordered provider chains (discovery, remediation, verification). Empty means "
            "single-provider mode: every call goes to `provider` (plus fallback_providers)."
        ),
    )


class AIRoleProvenance(BaseModel):
    """Which provider and model actually completed requests for one pipeline role.

    One entry per (role, provider, model) that completed at least one request, so a
    scan in which discovery fell back from one provider to another for some findings
    records both rather than reporting a single misleading provider.
    """

    model_config = ConfigDict(extra="forbid")

    role: AIRole = Field(..., description="Pipeline role: discovery, remediation or verification.")
    provider: str = Field(..., description="Provider that completed the request(s).")
    model: str = Field(..., description="Model identifier as reported by that provider.")
    retention_status: Literal["retained", "not_retained", "unverified"] = Field(
        ..., description="From the provider's configured retains_data metadata; None is recorded as 'unverified'."
    )
    retention_basis: Optional[str] = Field(
        None,
        description=(
            "Basis for retention_status (e.g. 'unverified_vendor_policy', "
            "'local_loopback_unverified_host_persistence', 'configured_retained'). "
            "Distinguishes documented vendor policy or local loopback execution from verified account settings."
        ),
    )
    retention_policy_note: Optional[str] = Field(
        None,
        description="Documented vendor retention policy or local-processing caveat for this provider.",
    )
    is_local: bool = Field(..., description="Whether the provider runs on this machine.")
    completed_requests: int = Field(..., ge=0, description="Requests this provider completed for this role.")


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
