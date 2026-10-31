"""
Builds a retrieval query from a finding and repository context, and
calls a Retriever with it.

The translation boundary between the domain layer (ScannerFinding,
RepositoryContext) and the generic retrieval layer (Retriever): this
file is allowed to know about ScannerFinding specifically so Retriever
implementations never have to. Query construction lives here rather
than in prompt_builder.py because it is a different job - a short
string optimized for retrieval matching, not the full text sent to the
LLM - and because retrieval happens before prompting, so the reverse
dependency (this file importing prompt_builder.py) would run backwards.

repository_context is a required RepositoryContext, not
Optional[RepositoryContext]: ai/repository_context.py already assigns
"what does no context look like" to the caller (RepositoryContext()),
so this function doesn't need a None-handling branch of its own.
top_k takes no default for the same reason Retriever.retrieve itself
doesn't: ai/config.py's AISettings.retrieval_top_k is the one place
that default lives - reading config here would be exactly the hidden
configuration lookup the frozen constraints rule out.
"""
from sentinelai.contracts import ScannerFinding

from ..repository_context import RepositoryContext
import logging

from .base import Retriever
from .models import RetrievedChunk


logger = logging.getLogger("sentinelai")


def _redacted_evidence(finding: ScannerFinding) -> str:
    """finding.raw_evidence with the scan's known secrets masked, anchored to its own location.

    Why the anchoring matters. SecretIndex holds two kinds of entry: literal
    secret strings, and positional ranges (file plus line numbers) recorded from
    GitLeaks findings. scanners/gitleaks.py passes --redact unconditionally, so a
    GitLeaks finding's own evidence arrives with the secret already replaced and
    the literal store stays empty - leaving the positional ranges as the only
    record of where a known secret actually lives.

    Those ranges only match when the text is identified by path and first line:
    redactor._same_file returns False for an empty path, so redacting this
    evidence without a location falls back to the regex patterns alone. Measured
    across six realistic evidence shapes, regex alone masks only the
    `token = "..."` assignment form; a secret passed as a function argument, held
    in a dict or list, written bare, or interpolated into an f-string is not
    matched. The exposure is a Bandit or Semgrep finding whose evidence window
    happens to span a line GitLeaks flagged - that text is not --redact'ed by
    anyone, because a different scanner produced it.

    raw_evidence specifically, and not the whole query: redactor._range_spans
    maps text lines onto file lines as `line_start + len(lines) - 1`, so it is
    only correct for text that really is the slice beginning at that line. The
    query also carries category, message and the repository summary, whose lines
    do not correspond to the file at all - anchoring the joined string would mask
    the wrong region. This mirrors prompt_builder._format_finding_section, which
    redacts the same field with the same two arguments for the prompt path.

    Returns the evidence unchanged when nothing matches, so a finding with no
    secret near it produces byte-identical query text.
    """
    # Imported inside the function: prompt_builder imports .retrieval, and
    # .retrieval/__init__ imports this module, so a module-level import would
    # close that cycle. Same approach as embeddings_ollama._prepare_texts.
    from ..prompt_builder import current_scan_session

    evidence = finding.raw_evidence or ""
    if not evidence:
        return ""
    if not finding.file or finding.line_start is None:
        # No location to anchor to - the regex patterns still apply.
        return current_scan_session().redact(evidence, None, None)
    return current_scan_session().redact(evidence, finding.file, finding.line_start)


def _build_query(finding: ScannerFinding, repository_context: RepositoryContext) -> str:
    """Combine the finding's descriptive fields (not its identifiers/location/severity) and any repo context summary."""
    parts = [finding.category.strip(), finding.message.strip()]
    if finding.raw_evidence:
        parts.append(_redacted_evidence(finding).strip())
    if repository_context.summary:
        parts.append(repository_context.summary.strip())
    return " ".join(parts)


def retrieve_context(
    finding: ScannerFinding,
    repository_context: RepositoryContext,
    retriever: Retriever,
    top_k: int,
) -> list[RetrievedChunk]:
    query = _build_query(finding, repository_context)
    chunks = retriever.retrieve(query, top_k)
    # Counts and the best score only: the query is built from the finding's
    # message and raw evidence, and logging it would put repository source code
    # into the log.
    logger.debug(
        "kb retrieval: finding_id=%s top_k=%d chunks=%d best_score=%s",
        finding.finding_id,
        top_k,
        len(chunks),
        f"{max((c.score for c in chunks), default=0.0):.3f}",
    )
    return chunks
