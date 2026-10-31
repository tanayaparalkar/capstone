"""
Native xAI Grok: https://api.x.ai/v1/chat/completions, authenticated with XAI_API_KEY.

Distinct from ai/llm/groq.py. Groq (api.groq.com) is a different company
serving open-weight models such as Llama; it does not serve Grok. Before
this module existed the name "grok" was aliased to the Groq provider,
which reported Groq-hosted Llama output as if it came from Grok. That
alias is gone: "grok" now means xAI and nothing else, and GROK_API_KEY
is not read by either provider.

Default model "grok-4" is an xAI alias for the current Grok 4 snapshot.
xAI's catalogue changes; override with SENTINELAI_AI_GROK_MODEL (or
--ai-model when grok is the selected provider) if the alias is retired.

retains_data is None, meaning "not verified", deliberately.
xAI documents default API request/response retention for 30 days for abuse
auditing, with Zero Data Retention available to eligible teams. Because
SentinelAI cannot inspect or verify the active account's ZDR configuration at
runtime, retention status remains "unverified" (never "not_retained").
"""
from typing import ClassVar, Optional

from ._openai_compat import OpenAICompatibleProvider

API_KEY_ENV = "XAI_API_KEY"
ENDPOINT = "https://api.x.ai/v1/chat/completions"
DEFAULT_MODEL = "grok-4"


class GrokProvider(OpenAICompatibleProvider):
    name: ClassVar[str] = "grok"
    is_local: ClassVar[bool] = False
    retains_data: ClassVar[Optional[bool]] = None
    retention_basis: ClassVar[str] = "unverified_account_zdr_or_30d_default"
    retention_policy_note: ClassVar[str] = (
        "xAI documents default API request/response retention for 30 days for abuse auditing, "
        "with Zero Data Retention available to eligible teams; actual account configuration "
        "cannot be verified at runtime."
    )
    ENDPOINT: ClassVar[str] = ENDPOINT
    API_KEY_ENV: ClassVar[str] = API_KEY_ENV

