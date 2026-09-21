"""
Deterministic, heuristic check of whether an LLM's response is grounded
in the finding's own evidence - not a re-verification of the finding
itself (that's the scanner's job, already done), and not a fact-checker
(verifying LLM output for truth deterministically is an unsolved
problem this file doesn't pretend to solve).

Consumes ScannerFinding and LLMResponse only, deliberately not
RetrievedChunk: retrieved KB chunks are general, type-level knowledge
that informed the LLM's reasoning, not ground truth to check a specific
claim against - only the finding's own raw_evidence is. Using chunks
here would also edge toward needing a text-similarity comparison,
pulling in exactly the kind of embedding/retrieval work "no retrieval"
rules out.

Returns only VerificationStatus, never modifies or constructs
AIEnrichedFinding: assembly is exclusively ai/pipeline.py's _assemble()
job, the same boundary ai/confidence_scorer.py and
ai/agents/explainer.py already keep.

Independent of ai/confidence_scorer.py: different questions from
different inputs with different failure semantics, and already
independently-lifecycled (enable_verification can disable this stage
with no equivalent flag for confidence scoring) - an import between the
two would be an unjustified coupling.

Never calls the LLM again: an LLM call is not reproducible even at
temperature 0 (violates "deterministic pure function"), and an LLM
grading another LLM's output is prone to the same hallucination failure
mode being checked for. A second-pass LLM "critic" is this file's named
advanced evolution, not its MVP form.

The heuristic: if finding.raw_evidence is absent, there is nothing
concrete to check claims against -> INSUFFICIENT_EVIDENCE. If the
finding's category is empty/whitespace after normalizing, there is
equally nothing meaningful to check against -> INSUFFICIENT_EVIDENCE
too, not VERIFIED (an earlier draft let this fall through to VERIFIED
by accident of control flow, not by design - fixed).

Otherwise, checked in two tiers, phrase before tokens: first, whether
the LLM's combined output text contains the whole normalized category
as a phrase (e.g. "sql injection") -> VERIFIED. Only if that fails,
fall back to checking individual category terms, with generic security
suffixes (_GENERIC_CATEGORY_TERMS: injection, exposure, disclosure,
overflow, validation, execution, traversal) filtered out first ->
VERIFIED if any distinguishing term remains and appears. Otherwise ->
REJECTED.

The stopword filter exists because the token fallback alone
reintroduces the exact false positive the phrase tier was added to
avoid: "command-injection" would match on the bare word "injection"
even when the LLM's text is actually about SQL injection - a different
CWE family entirely, and one of several ("SQL", "Command", "LDAP",
"XPath", "CRLF" injection) that all share that one generic word.
Filtering it (and the other generic suffixes) out of the fallback means
only "command" is checked for "command-injection", so a same-suffix,
different-family mismatch is correctly rejected, while a
differently-phrased but genuinely on-topic response ("the command
execution here is dangerous") still verifies on the distinguishing
term "command". If every token in a category is generic (all filtered
out), the fallback has nothing left to check and falls through to
REJECTED - Python's any() over an empty list is already False, so this
needs no special-case branch.

Never returns UNVERIFIED - that state means "verification wasn't
attempted," a decision ai/pipeline.py makes by not calling this
function at all, not a conclusion this function itself would reach.
"""
import logging

from sentinelai.contracts import ScannerFinding, VerificationStatus

from .llm_response import LLMResponse

logger = logging.getLogger("sentinelai")

_GENERIC_CATEGORY_TERMS = {
    "injection",
    "exposure",
    "disclosure",
    "overflow",
    "validation",
    "execution",
    "traversal",
}

# Applied by _normalize_category only. Same terms as _GENERIC_CATEGORY_TERMS plus
# "security" - see _normalize_category's docstring for why the two sets are kept
# separate rather than merged.
_NORMALIZATION_STOP_WORDS = _GENERIC_CATEGORY_TERMS | {"security"}


def _combined_llm_text(llm_response: LLMResponse) -> str:
    parts = [llm_response.explanation, llm_response.remediation]
    if llm_response.exploit_path:
        parts.append(llm_response.exploit_path)
    if llm_response.impact:
        parts.append(llm_response.impact)
    return " ".join(parts).lower()


def _normalize_category(category: str) -> str:
    """Lowercase, split on both separators, and drop generic security nouns.

    Underscores are handled alongside hyphens because Bandit's categories are
    underscore-joined test names (`subprocess_popen_with_shell_equals_true`,
    `hardcoded_password_string`). Replacing only hyphens left those as a single
    40-character token that no generated explanation could ever contain, so every
    Bandit finding failed both the phrase tier and the token fallback and was
    REJECTED regardless of how on-topic the explanation actually was.

    `or text` keeps the pre-filter string when filtering would empty it, so a
    category that is *entirely* generic (Semgrep's bare "security", the value its
    metadata carries for most rules) still yields something to match on rather than
    collapsing to "" and being reported as INSUFFICIENT_EVIDENCE by verify_finding.
    Note this deliberately does not make such a category verify more often - it
    still has to appear in the generated text - it only preserves the existing
    outcome for it instead of changing it.

    _GENERIC_CATEGORY_TERMS is intentionally left as-is and still applied separately
    by _distinguishing_terms: this function's stop list additionally contains
    "security", which must not leak into the token fallback, where removing it would
    silently turn every already-REJECTED bare-"security" finding into a different
    verdict rather than leaving that behaviour untouched.
    """
    text = category.lower().replace("-", " ").replace("_", " ").strip()
    tokens = [token for token in text.split() if token not in _NORMALIZATION_STOP_WORDS]
    return " ".join(tokens) or text


def _distinguishing_terms(normalized_category: str) -> list[str]:
    return [term for term in normalized_category.split() if term not in _GENERIC_CATEGORY_TERMS]


def verify_finding(finding: ScannerFinding, llm_response: LLMResponse) -> VerificationStatus:
    """Public entry point. The deterministic check itself is unchanged, in _verify_finding below.

    Split only so the verdict can be recorded once rather than at each of the
    five exits; the logic, its ordering and its return values are untouched.
    """
    status = _verify_finding(finding, llm_response)
    logger.debug(
        "verifier: finding_id=%s category=%s verdict=%s",
        finding.finding_id,
        finding.category,
        status.value,
    )
    return status


def _verify_finding(finding: ScannerFinding, llm_response: LLMResponse) -> VerificationStatus:
    if not finding.raw_evidence:
        return VerificationStatus.INSUFFICIENT_EVIDENCE

    normalized_category = _normalize_category(finding.category)
    if not normalized_category:
        return VerificationStatus.INSUFFICIENT_EVIDENCE

    combined_text = _combined_llm_text(llm_response)

    if normalized_category in combined_text:
        return VerificationStatus.VERIFIED

    if any(term in combined_text for term in _distinguishing_terms(normalized_category)):
        return VerificationStatus.VERIFIED

    return VerificationStatus.REJECTED
