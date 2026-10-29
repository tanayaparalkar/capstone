"""
AI provenance for every output format: which provider and model ran, under
which egress policy, and how many secrets were masked.

One module so the terminal, Markdown, HTML and SARIF views cannot word or
count it differently - the same reason patch_section.py exists.

Absence is stated, never implied. A scanner-only run renders "AI: not
used" rather than leaving the line out, and an enriched run whose
redaction was not recorded says so instead of showing "0 secrets
masked", which would claim a redaction pass that never happened.

Only the redaction COUNT is ever rendered. RedactionReport.patterns_hit
holds pattern names for debugging; next to a file path, "aws-access-key"
tells a reader where a credential lives. Nothing in this module reads
patterns_hit, and build_json_report strips it before serialization.

The model identifier is recorded per run because the evaluation protocol
compares latency and detection numbers across runs; a provider or model
change mid-evaluation is only detectable afterwards if every report says
which one produced it.
"""
from typing import Optional

from ..contracts import ScanMetadata

NOT_USED = "not used"
_EGRESS_WORDS = {"allow": "allowed", "deny": "denied"}


def ai_used(metadata: ScanMetadata) -> bool:
    return metadata.ai_provider is not None


def ai_label(metadata: ScanMetadata) -> str:
    """"groq/llama-3.3-70b-versatile", or "not used" for a scanner-only run."""
    if metadata.ai_provider is None:
        return NOT_USED
    return f"{metadata.ai_provider}/{metadata.ai_model or 'unknown model'}"


def egress_label(metadata: ScanMetadata) -> str:
    if metadata.egress_policy is None:
        return "not recorded"
    return _EGRESS_WORDS.get(metadata.egress_policy, metadata.egress_policy)


def redaction_count(metadata: ScanMetadata) -> Optional[int]:
    return metadata.redaction.secrets_masked if metadata.redaction is not None else None


def redaction_label(metadata: ScanMetadata) -> str:
    count = redaction_count(metadata)
    if count is None:
        return "redaction not applied"
    return f"{count} secret{'' if count == 1 else 's'} masked"


def provenance_line(metadata: ScanMetadata) -> str:
    """The one-line form: "AI: groq/llama-3.3-70b · egress: allowed · 3 secrets masked", or "AI: not used"."""
    if not ai_used(metadata):
        return f"AI: {NOT_USED}"
    return f"AI: {ai_label(metadata)} · egress: {egress_label(metadata)} · {redaction_label(metadata)}"


def provenance_items(metadata: ScanMetadata) -> list[tuple[str, str]]:
    """(label, value) pairs for layouts that show facts side by side, such as the HTML header."""
    if not ai_used(metadata):
        return [("AI", NOT_USED)]
    return [("AI", ai_label(metadata)), ("Egress", egress_label(metadata)), ("Redaction", redaction_label(metadata))]


def sarif_properties(metadata: ScanMetadata) -> dict:
    """Run-level SARIF properties, as raw values: null where nothing was recorded."""
    return {
        "aiProvider": metadata.ai_provider,
        "aiModel": metadata.ai_model,
        "egressPolicy": metadata.egress_policy,
        "redactionCount": redaction_count(metadata),
    }
