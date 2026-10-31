"""
Assembles the reasoning prompt from a finding, repository context, and
already-retrieved knowledge-base chunks.

This is the boundary between the retrieval layer's output and the LLM
call: it consumes RetrievedChunk, imported from ai.retrieval's public
surface (`from .retrieval import RetrievedChunk`), not from
ai/retrieval/models.py directly - matching the same package-boundary
fix applied to ai/embeddings_ollama.py's EmbeddingFn import. Note this
does mean Python transitively loads Retriever/InMemoryRetriever/
context_retrieval.py as an import-mechanics side effect of
ai/retrieval/__init__.py eagerly importing its own __all__ - the same
accepted tradeoff already present wherever anything imports from
ai.retrieval. This file's own code never references any of them: no
Retriever, no context_retrieval.py, no retriever implementation is
used in this file's logic. Changing embedding providers or swapping
in-memory/Qdrant retrieval must never require touching this file's
code, and rewording this file must never require touching retrieval's
code - the transitive import graph is not the same thing as a code-level
dependency.

Includes ai/prompt_templates.py's content as a module constant here
rather than a separate file, per the frozen architecture's merge
decision - a plain string, not a Jinja2 template, since there is no
conditional templating problem here Jinja2 would actually be solving.

chunk similarity scores are deliberately excluded from the prompt: a
score is a fact about how well text matched the retrieval query, not
about the vulnerability, and showing it risks the LLM conflating
retrieval relevance with AIEnrichedFinding.confidence_score - a
different, separately-computed value produced later by
confidence_scorer.py, not by the LLM.

repository_context is a required RepositoryContext, not Optional, for
the same reason ai/retrieval/context_retrieval.py's parameter is: the
"no context" default lives with pipeline.py, not duplicated here.

_OUTPUT_FORMAT_INSTRUCTIONS was missing entirely until this file was
observed producing the actual bug it was meant to prevent:
_TASK_INSTRUCTIONS above asked the model to "explain... describe...
note... suggest," with no mention of JSON anywhere in the prompt, while
ai/agents/explainer.py has always called json.loads() on the result and
ai/llm_response.py's LLMResponse has always rejected any extra key.
Nothing in the prompt ever told the model about either constraint.
Verified directly against a real llama3.1:8b response before writing
this: with no format instruction, the model produced a well-formed,
useful, entirely prose explanation with Markdown bold headers - not
malformed output, just the ordinarily reasonable answer to a prompt
that never asked for JSON in the first place. Markdown code fences
(handled defensively by ai/llm_ollama.py's _strip_markdown_fences) are
a *secondary* symptom of the same root cause - a model given no output-
format instruction may reasonably choose to fence code-shaped content
- not a problem this section alone would have covered without the
fence-stripping already in place.

Placed as the *last* section build_prompt() appends, after the finding,
repository context, and retrieved knowledge - not folded into
_TASK_INSTRUCTIONS at the top. Format instructions are more reliably
followed the closer they are to where generation actually begins;
putting the six field names and the "JSON only" requirement immediately
before the model starts producing output, rather than several hundred
tokens earlier, is a real difference for a small local model, not a
stylistic preference.

Field list, names, and required/optional split are read directly from
LLMResponse's own field declarations (ai/llm_response.py) - title,
explanation, and remediation are required; exploit_path, impact, and
patch_suggestion are Optional[str] = None. Instructing the model to
include all six keys with explicit `null` for anything not applicable,
rather than allowing keys to be silently omitted, is the more reliable
instruction for a small model even though Pydantic itself would accept
either: an omitted-vs-present-with-null distinction is exactly the kind
of small inconsistency a smaller model is more likely to get wrong than
a larger one.

The provider path. Every prompt section built from repository text goes
through the same per-scan ScanSession, so the order is fixed by
construction:

    1. slice_around()            bounded code context (ai/context)
    2. redact_slice()            secrets masked, raw evidence included (ai/redaction)
    3. the prompt                built here, from redacted text only
    4. assert_safe()             the egress gate (ai/egress)
    5. provider.complete()       the call (ai/llm)
    6. redaction_report()        the accumulated RedactionReport, for ScanMetadata

Steps 4 and 5 live in one closure, guarded_generate(): it is the only
generate callable the pipeline is given, and it has no branch that calls
complete() without first returning from assert_safe(). The SecretIndex is
built once, in scan_session(), from the scan's full finding list before
any prompt exists. Each finding's sections are redacted once and cached,
so the three agents see identical evidence and the report counts each
slice once. Outside a scan_session() (direct callers, tests), a throwaway
session with no GitLeaks index still slices and pattern-redacts: no
prompt section built here is ever unredacted.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, Optional

from sentinelai.backend.code_context import CodeContext
from sentinelai.contracts import AIConfig, AIFault, RedactionReport, ScannerFinding

from .config import get_settings
from .context import CodeSliceError, slice_around
from .egress import EgressViolation, assert_safe
from .llm.base import LLMProvider
from .llm.errors import ProviderError
from .redaction import SecretIndex
from .repository_context import RepositoryContext
from .retrieval import RetrievedChunk

_TASK_INSTRUCTIONS = (
    "You are a security analyst reviewing a single static-analysis finding. Using the "
    "finding's evidence below and the related security knowledge (if any), explain the "
    "vulnerability, describe how it could be exploited if applicable, note its potential "
    "impact, and suggest a remediation."
)

_OUTPUT_FORMAT_INSTRUCTIONS = (
    "Respond with ONLY a single JSON object and no other text - no explanation before or "
    "after it, and no Markdown code fences (no ```). The JSON object must have exactly these "
    'six keys: "title", "explanation", "exploit_path", "impact", "remediation", '
    '"patch_suggestion".\n'
    '"title", "explanation", and "remediation" are required non-empty strings. '
    '"exploit_path", "impact", and "patch_suggestion" are optional: include the key and set '
    "it to JSON null if it does not apply to this finding, rather than omitting the key or "
    'writing a placeholder like "N/A".'
)


@dataclass
class ScanSession:
    """Per-scan redaction state: one SecretIndex, slice limits, and the running RedactionReport."""

    index: SecretIndex
    before: int
    after: int
    max_chars: int
    hits: list[str] = field(default_factory=list)
    slices_scanned: int = 0
    sections: dict[str, tuple[str, Optional[str]]] = field(default_factory=dict)

    def redact(self, text: str, path: Optional[str], line_start: Optional[int]) -> str:
        redacted, hits = self.index.redact_slice(text, path or "", line_start or 1)
        self.hits += hits
        self.slices_scanned += 1
        return redacted

    def redaction_report(self) -> RedactionReport:
        return RedactionReport(
            secrets_masked=len(self.hits), patterns_hit=sorted(set(self.hits)), slices_scanned=self.slices_scanned
        )


_session: ContextVar[Optional[ScanSession]] = ContextVar("sentinelai_scan_session", default=None)


@contextmanager
def scan_session(findings: list[ScannerFinding], cfg: AIConfig) -> Iterator[ScanSession]:
    """Build the scan's SecretIndex once, from every finding, before any prompt is constructed."""
    index = SecretIndex()
    index.add_from_findings(findings)
    before = cfg.max_context_lines // 2
    session = ScanSession(index, before, cfg.max_context_lines - before, cfg.max_context_chars)
    token = _session.set(session)
    try:
        yield session
    finally:
        _session.reset(token)


def _current_session() -> ScanSession:
    session = _session.get()
    if session is not None:
        return session
    lines = get_settings().code_context_lines
    return ScanSession(SecretIndex(), lines, lines, AIConfig.model_fields["max_context_chars"].default)


def current_scan_session() -> ScanSession:
    """The active ScanSession, or a throwaway one carrying the same defaults.

    Public because ai/embeddings_ollama.py needs the same per-scan SecretIndex
    and char limit this module uses for prompts: the retrieval query it embeds
    is built from the same finding evidence, so redacting it against a
    different index - or none - would mask a different set of secrets than the
    prompt path masks. Reaching into _current_session() from another module
    would couple to a private name; this is that one supported accessor.

    Outside a scan the throwaway session has an empty SecretIndex, so only the
    regex patterns in ai/redaction/patterns.py apply. That is the same
    best-effort fallback a direct prompt caller already gets, and the egress
    gate still runs afterwards either way.
    """
    return _current_session()


def guarded_generate(provider: LLMProvider, cfg: AIConfig, *, max_tokens: int, temperature: float) -> Callable[..., str]:
    """The only generate callable the pipeline receives: steps 4 and 5, in that order, with no other path."""
    if hasattr(provider, "complete_for_role"):
        def _call_role(
            role: str,
            prompt: str,
            response_schema=None,
            system: Optional[str] = None,
            role_max_tokens: Optional[int] = None,
            role_temperature: Optional[float] = None,
        ) -> str:
            return provider.complete_for_role(
                role,
                prompt,
                system=system,
                max_tokens=role_max_tokens if role_max_tokens is not None else max_tokens,
                temperature=role_temperature if role_temperature is not None else temperature,
                assert_safe_fn=assert_safe,
            ).text

        def generate(
            prompt: str,
            response_schema=None,
            system: Optional[str] = None,
            max_tokens: Optional[int] = None,
            temperature: Optional[float] = None,
        ) -> str:
            return _call_role(
                "discovery",
                prompt,
                response_schema=response_schema,
                system=system,
                role_max_tokens=max_tokens,
                role_temperature=temperature,
            )

        def _for_role(role: str) -> Callable[..., str]:
            def _role_generate(
                prompt: str,
                response_schema=None,
                system: Optional[str] = None,
                max_tokens: Optional[int] = None,
                temperature: Optional[float] = None,
            ) -> str:
                return _call_role(
                    role,
                    prompt,
                    response_schema=response_schema,
                    system=system,
                    role_max_tokens=max_tokens,
                    role_temperature=temperature,
                )
            return _role_generate

        generate.for_role = _for_role  # type: ignore[attr-defined]
        if hasattr(provider, "reset_for_finding"):
            generate.reset_for_finding = provider.reset_for_finding  # type: ignore[attr-defined]
        return generate

    def generate(prompt: str, response_schema=None, system: Optional[str] = None) -> str:
        effective_cfg = getattr(provider, "active_config", cfg)
        if system is not None:
            assert_safe(prompt, effective_cfg, provider, system=system)
            return provider.complete(prompt, system=system, max_tokens=max_tokens, temperature=temperature).text
        assert_safe(prompt, effective_cfg, provider)
        return provider.complete(prompt, max_tokens=max_tokens, temperature=temperature).text

    return generate


def ai_fault(finding_id: Optional[str], exc: BaseException) -> AIFault:
    """An operational fault for ScanMetadata. ProviderError and EgressViolation messages are payload-free by
    construction; any other exception is recorded by type alone, since its message may quote model output."""
    safe = isinstance(exc, (ProviderError, EgressViolation))
    return AIFault(finding_id=finding_id, kind=type(exc).__name__, message=str(exc) if safe else type(exc).__name__)


def _redacted_sections(finding: ScannerFinding) -> tuple[str, Optional[str]]:
    session = _current_session()
    if finding.finding_id not in session.sections:
        session.sections[finding.finding_id] = (
            _format_finding_section(finding, session),
            _format_code_context_section(finding, session),
        )
    return session.sections[finding.finding_id]


def _format_finding_section(finding: ScannerFinding, session: ScanSession) -> str:
    lines = [
        f"Finding: {finding.finding_id}",
        f"Category: {finding.category}",
        f"Severity: {finding.severity.value}",
        f"Rule: {finding.rule_id}",
    ]
    if finding.file:
        location = finding.file
        if finding.line_start:
            location += f":{finding.line_start}"
            if finding.line_end and finding.line_end != finding.line_start:
                location += f"-{finding.line_end}"
        lines.append(f"Location: {location}")
    if finding.cwe:
        lines.append(f"CWE: {finding.cwe}")
    lines.append(f"Description: {session.redact(finding.message, None, None)}")
    if finding.raw_evidence:
        # The evidence's first line is the finding's own line, so GitLeaks positions line up.
        lines.append(f"Evidence:\n{session.redact(finding.raw_evidence, finding.file, finding.line_start)}")
    return "\n".join(lines)


def _format_code_context_section(finding: ScannerFinding, session: ScanSession) -> Optional[str]:
    """Render the source lines around the finding, or None when there are none to show.

    Lines are numbered and the finding's own span is marked with `>`. Both are
    load-bearing rather than decorative: the numbers are what let a model refer
    to a specific line, and the marker is what distinguishes the reported line
    from the context around it - without it a twenty-line window buries the one
    line the finding is actually about.

    Returns None rather than a placeholder string when no context is available,
    so a finding with no readable source (a dependency advisory, a deleted file)
    produces a prompt byte-identical to the one it produced before this section
    existed.
    """
    if not finding.file or finding.line_start is None:
        return None
    try:
        code = slice_around(
            Path(finding.file), finding.line_start, before=session.before, after=session.after, max_chars=session.max_chars
        )
    except (CodeSliceError, ValueError):
        return None
    text = session.redact(code.text, finding.file, code.line_start)
    focus_end = min(max(finding.line_end or finding.line_start, finding.line_start), code.line_end)
    return _render_code_context(
        CodeContext(finding.file, code.line_start, code.line_end, finding.line_start, focus_end, tuple(text.split("\n")))
    )


def _render_code_context(context: CodeContext) -> str:
    """Format an already-read window. Split out so it can be tested without touching disk."""
    focus = (
        f"line {context.focus_start}"
        if context.focus_start == context.focus_end
        else f"lines {context.focus_start}-{context.focus_end}"
    )
    header = (
        f"Source context ({context.file_path}, showing lines "
        f"{context.start_line}-{context.end_line}; the finding is at {focus}, marked with >):"
    )
    width = len(str(context.end_line))
    body = [
        f"{'>' if context.focus_start <= number <= context.focus_end else ' '} "
        f"{str(number).rjust(width)} | {text}"
        for number, text in enumerate(context.lines, start=context.start_line)
    ]
    return "\n".join([header, *body])


def _format_repository_context_section(repository_context: RepositoryContext) -> Optional[str]:
    if not repository_context.summary:
        return None
    return f"Repository context: {repository_context.summary}"


def _format_retrieved_chunks_section(retrieved_chunks: list[RetrievedChunk]) -> Optional[str]:
    if not retrieved_chunks:
        return None
    lines = [
        "Related security knowledge:",
        "Use the following knowledge only as supporting context. Do not copy it verbatim. "
        "Prefer the finding's evidence if there is any conflict.",
        "",
    ]
    for chunk in retrieved_chunks:
        lines.append(f"- [{chunk.kb_entry_id}] {chunk.text}")
    return "\n".join(lines)


def build_evidence_block(
    finding: ScannerFinding,
    repository_context: RepositoryContext,
    retrieved_chunks: list[RetrievedChunk],
) -> str:
    """The shared evidence every agent reasons over: the finding, repository context, and KB chunks.

    Extracted so ai/agents/'s three prompts present *identical* evidence and
    differ only in their role instruction and output schema. Duplicating this
    formatting per agent would let the agents silently drift apart on what they
    were shown, which would undermine the one property the critic stage depends
    on - that every agent judged the same material.

    Returns only the evidence sections; the role instruction and the output
    contract are each agent's own concern.
    """
    finding_section, code_context_section = _redacted_sections(finding)
    sections = [finding_section]
    if code_context_section is not None:
        sections.append(code_context_section)

    repository_context_section = _format_repository_context_section(repository_context)
    if repository_context_section is not None:
        sections.append(repository_context_section)

    retrieved_chunks_section = _format_retrieved_chunks_section(retrieved_chunks)
    if retrieved_chunks_section is not None:
        sections.append(retrieved_chunks_section)

    return "\n\n".join(sections)


def build_prompt(
    finding: ScannerFinding,
    repository_context: RepositoryContext,
    retrieved_chunks: list[RetrievedChunk],
) -> str:
    sections = [_TASK_INSTRUCTIONS, _redacted_sections(finding)[0]]

    repository_context_section = _format_repository_context_section(repository_context)
    if repository_context_section is not None:
        sections.append(repository_context_section)

    retrieved_chunks_section = _format_retrieved_chunks_section(retrieved_chunks)
    if retrieved_chunks_section is not None:
        sections.append(retrieved_chunks_section)

    sections.append(_OUTPUT_FORMAT_INSTRUCTIONS)

    return "\n\n".join(sections)
