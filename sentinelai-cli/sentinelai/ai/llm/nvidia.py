"""
NVIDIA NIM via NVIDIA's hosted API catalog:
https://integrate.api.nvidia.com/v1/chat/completions, authenticated with NVIDIA_API_KEY.

OpenAI-compatible, so it shares ai/llm/_openai_compat.py with native Grok.

Scope: the hosted endpoint only. A self-hosted NIM container exposes the
same API at an operator-chosen URL, but accepting an arbitrary URL here
would add an unvalidated egress destination; that is left out until it
can go through the same endpoint validation the local provider has.

Default model "meta/llama-3.1-70b-instruct" is a long-standing entry in
the NVIDIA API catalog. Availability depends on the catalog and the
account; override with SENTINELAI_AI_NVIDIA_MODEL (or --ai-model when
nvidia is the selected provider).

retains_data is None, meaning "not verified", deliberately.
This integration targets hosted NVIDIA NIM (integrate.api.nvidia.com), which is
distinct from self-hosted local NIM containers. Because hosted account and
endpoint retention settings cannot be verified at runtime, retention status
remains "unverified" (never "not_retained").
"""
from typing import ClassVar, Optional

from ._openai_compat import OpenAICompatibleProvider

API_KEY_ENV = "NVIDIA_API_KEY"
ENDPOINT = "https://integrate.api.nvidia.com/v1/chat/completions"
DEFAULT_MODEL = "meta/llama-3.1-70b-instruct"


class NvidiaNimProvider(OpenAICompatibleProvider):
    name: ClassVar[str] = "nvidia"
    is_local: ClassVar[bool] = False
    retains_data: ClassVar[Optional[bool]] = None
    retention_basis: ClassVar[str] = "unverified_hosted_endpoint_policy"
    retention_policy_note: ClassVar[str] = (
        "Targets hosted NVIDIA NIM (integrate.api.nvidia.com), which has distinct data-handling "
        "terms from self-hosted local NIM deployments; hosted account retention settings cannot "
        "be verified at runtime."
    )
    ENDPOINT: ClassVar[str] = ENDPOINT
    API_KEY_ENV: ClassVar[str] = API_KEY_ENV

