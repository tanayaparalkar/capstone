"""
SentinelAI CLI entry point.

Two distinct workflows live here:

- `scan`: retrieves a ScanResult from a FindingsProvider (LiveFindingsProvider
  by default - the real backend + scanner framework, per _get_provider();
  MockFindingsProvider remains available and is what the CLI test suite
  pins itself to, for deterministic, tool-free assertions). If
  Tanaya's AI layer is configured (SENTINELAI_AI_LLM_MODEL and
  SENTINELAI_AI_EMBEDDING_MODEL both set), rebuilds repository context via
  the backend (sentinelai.backend), adapts it with
  sentinelai.ai.repository_context.from_backend_context(), and runs
  sentinelai.ai.enrich_findings() over the scanner findings to populate
  ai_findings; otherwise ai_findings stays empty, exactly as it always
  has. Either way, renders the result live (--format terminal) or
  writes a report.
- `report`: reads a previously-saved JSON report back into a ScanResult
  (sentinelai/reporting/loader.py) and renders it. It NEVER calls a
  FindingsProvider and NEVER calls enrich_findings() - no scanner,
  repository analysis, RAG, LLM reasoning, or verification ever runs
  again for an existing result.

Both commands share the same report generators (sentinelai/reporting/)
and the same statistics engine (sentinelai/statistics/), so a report
generated via `report` is indistinguishable from one generated directly
by `scan` for the same underlying findings.

Rich terminal presentation (presentation/) is used only for `scan`'s
--format terminal (the default). Structured formats (json/markdown/html/
sarif) never touch it, keeping their output machine-readable and pipeable.

Error handling and exit codes: every `typer.Exit` here uses a named
ExitCode (sentinelai/core/exit_codes.py) rather than a raw integer, and
every user-facing error goes through print_error() (stderr), never
stdout - see that module's docstring, and core/exit_codes.py, for the
full convention. `--debug` (an app-level flag, available before any
subcommand) additionally prints the real traceback to stderr on an
unexpected failure; without it, only a one-line message is shown.
"""
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import typer
import typer.rich_utils
from pydantic import ValidationError
from rich.console import Console
from rich.markup import escape

from . import __version__
from .ai import enrich_findings
from .ai.config import PROVIDERS, AIConfigError, ai_models_configured, get_settings, resolve_config
from .ai.confidence_scorer import score_confidence
from .ai.factory import create_llm_provider, create_retriever_for_scan, create_role_generator
from .ai.prompt_builder import ai_fault, guarded_generate, scan_session
from .ai.repository_context import from_backend_context
from .ai.verifier import verify_finding
from .backend.context_builder import build_repository_context
from .patching import run_patches
from .backend.loader import load_repository
from .contracts import (
    AIConfig,
    ScanMode,
    ScannerTier,
    ScanResult,
    Severity,
    aggregate_retention_status,
    retention_status_from_provider,
)

_default_create_llm_provider = create_llm_provider
_default_create_role_generator = create_role_generator
from .core import ExitCode, exceeds_fail_on_threshold, filter_by_severity
from .core.errors import AIEnrichmentError
from .core.observability import log_duration
from .patcher import run_remediation_session
from .presentation import (
    ScanProgress,
    get_console,
    print_error,
    print_traceback,
    render_finding_detail,
    render_findings_table,
    render_scan_header,
    render_summary,
)
from .presentation.patches import render_patch_application
from .presentation.providers import render_providers
from .providers import FindingsProvider, LiveFindingsProvider
from .reporting.provenance import provenance_line
from .reporting import (
    ReportLoadError,
    build_patch_application_report,
    load_patch_application,
    load_scan_result,
    to_html,
    to_json,
    to_markdown,
    to_sarif,
)
from .statistics import ScanStatistics, calculate_statistics

app = typer.Typer(
    name="sentinelai",
    help="SentinelAI - AI-Powered Vulnerability Detection for DevSecOps",
    add_completion=False,
)

# Typer renders --help through Rich, which treats GitHub Actions as a
# colour-capable terminal even though stdout there is a pipe. Rich styles
# each hyphen of an option separately, so "--apply-patches" is emitted as
# ESC "-" ESC "-apply-patches" and the flag name never appears as a literal
# string: `sentinelai scan --help | grep -- --apply-patches` finds nothing.
# Keep the styled help when a real terminal is attached, and emit plain text
# whenever it is not, which is the usual convention for colourised CLIs.
# hasattr-guarded because FORCE_TERMINAL is a Typer internal: if a future
# Typer drops it, help simply keeps its colour rather than failing to import.
if not sys.stdout.isatty() and hasattr(typer.rich_utils, "FORCE_TERMINAL"):
    typer.rich_utils.FORCE_TERMINAL = False

# Operational logging, separate from the user-facing console output above:
# silent by default (no handler is configured here - that's left to
# whoever deploys/embeds SentinelAI, matching stdlib logging convention),
# so this adds no visible output and changes no existing behavior. It
# exists purely so an operator who *does* configure a handler gets a
# trail of scan starts/completions/failures for unattended runs.
logger = logging.getLogger("sentinelai")

VALID_FORMATS = {"terminal", "json", "markdown", "html", "sarif"}
# Matches ai/llm_ollama.py's deterministic temperature; the token cap bounds cost per agent call.
AI_TEMPERATURE = 0.0
AI_MAX_TOKENS = 2048
REPORT_FORMATS = {"json", "markdown", "html", "sarif"}  # no "terminal" - report renders persisted formats only
FAIL_ON_VALUES = {"low", "medium", "high", "critical", "none"}


@app.callback()
def main(
    ctx: typer.Context,
    debug: bool = typer.Option(
        False, "--debug", help="Show full tracebacks on unexpected errors instead of a one-line message."
    ),
) -> None:
    ctx.obj = {"debug": debug}


def _get_provider() -> FindingsProvider:
    """The single seam swapped from MockFindingsProvider to the real backend."""
    return LiveFindingsProvider()


def _fail(message: str, code: ExitCode) -> None:
    """Print a clean, actionable error to stderr and exit with `code`."""
    print_error(message)
    raise typer.Exit(code=code)


def _fail_from_exception(message: str, exc: Exception, code: ExitCode, debug: bool) -> None:
    print_error(f"{message}: {exc}")
    if debug:
        print_traceback()
    raise typer.Exit(code=code)


def _ai_settings():
    """get_settings(), with an unusable SENTINELAI_AI_* value reported as INVALID_INPUT.

    INVALID_INPUT (2) for the same reason _require_ai_configuration() below uses
    it: a rejected SENTINELAI_AI_LLM_HOST means the command as invoked is not
    satisfiable, nothing has failed and nothing external has been contacted.

    Without this the pydantic ValidationError propagates out of the command and
    Typer exits 1 - which is SECURITY_FINDINGS, so a CI gate could not tell a
    misconfigured host from findings crossing its threshold. Only the first
    error's message is printed; the full pydantic dump is noise to a user who
    set one environment variable wrongly.
    """
    try:
        return get_settings()
    except ValidationError as exc:
        _fail(_settings_error_message(exc), ExitCode.INVALID_INPUT)


def _settings_error_message(exc: ValidationError) -> str:
    """The first validation error's own message. The full pydantic dump is noise."""
    errors = exc.errors()
    detail = errors[0].get("msg", "") if errors else ""
    text = str(detail) or str(exc)
    if text.startswith("Value error, "):
        text = text[len("Value error, "):]
    return f"invalid AI configuration: {text}"


def _ai_decision():
    """Return (settings, error, models_configured) without refusing a scanner-only run.

    Goes through get_settings() - the same seam main.py uses everywhere else, so
    injecting settings in a test still governs the decision - but tolerates a
    failure here rather than exiting, because at this point it is not yet known
    whether AI will be used at all. A stale SENTINELAI_AI_LLM_HOST must not fail
    a scan that was never going to contact Ollama.

    When settings cannot be built, the environment is consulted for one fact
    only: did the user ask for AI? That keeps a genuine misconfiguration
    reportable - models set plus a bad host still fails - instead of silently
    degrading to a scanner-only run, while a scan with no AI configured is
    unaffected. The caller enforces the error before anything is transmitted.
    """
    try:
        settings = get_settings()
    except ValidationError as exc:
        return None, exc, ai_models_configured()
    configured = settings.llm_model is not None and settings.embedding_model is not None
    return settings, None, configured


def _require_ai_configuration() -> None:
    """Exit with INVALID_INPUT unless both AI model variables are set.

    INVALID_INPUT (2) rather than PROVIDER_ERROR (3): nothing has failed yet
    and nothing external has been contacted - the command as invoked is simply
    not satisfiable, which is the same category as `--quick --full` or an
    unknown --format value. PROVIDER_ERROR stays reserved for a configured
    dependency that actually failed, which is what the enrichment block below
    still reports.

    Names only the variables that are missing, so a user who has set one of the
    two is not told to set both.
    """
    settings = _ai_settings()
    missing = []
    if settings.llm_model is None:
        missing.append("SENTINELAI_AI_LLM_MODEL")
    if settings.embedding_model is None:
        missing.append("SENTINELAI_AI_EMBEDDING_MODEL")
    if not missing:
        return

    _fail(
        f"--ai requires AI enrichment to be configured, but {' and '.join(missing)} "
        f"{'is' if len(missing) == 1 else 'are'} not set. "
        "Start a local Ollama server (`ollama serve`), pull both models "
        "(`ollama pull llama3.1:8b` and `ollama pull nomic-embed-text`), then set "
        "SENTINELAI_AI_LLM_MODEL=llama3.1:8b and SENTINELAI_AI_EMBEDDING_MODEL=nomic-embed-text. "
        "No API key is required. Omit --ai to run a scanner-only scan.",
        ExitCode.INVALID_INPUT,
    )


def _resolve_ai_config(cli_overrides: dict) -> AIConfig:
    """resolve_config(), with a configuration problem reported as INVALID_INPUT.

    INVALID_INPUT (2), not PROVIDER_ERROR (3): an unknown provider, a bad egress value or a
    missing API key is CLI misconfiguration the user fixes, not a dependency that failed.
    """
    try:
        return resolve_config(cli_overrides)
    except AIConfigError as exc:
        _fail(f"invalid AI configuration: {exc}.", ExitCode.INVALID_INPUT)


def _render_structured_content(
    format: str,
    result: ScanResult,
    stats: ScanStatistics,
    patch_application=None,
) -> str:
    """Dispatch to the appropriate report generator - shared by `scan` and `report`."""
    with log_duration("report generation", format=format):
        return _render_structured_content_inner(format, result, stats, patch_application)


def _render_structured_content_inner(
    format: str,
    result: ScanResult,
    stats: ScanStatistics,
    patch_application=None,
) -> str:
    if format == "json":
        return to_json(result, stats, patch_application=patch_application)
    if format == "markdown":
        return to_markdown(result, stats, patch_application)
    if format == "html":
        return to_html(result, stats, patch_application)
    return to_sarif(result, stats, patch_application=patch_application)  # sarif


def _write_or_print(console: Console, content: str, output: Optional[str], debug: bool) -> None:
    """Write `content` (UTF-8) to `output` if given, else write raw UTF-8 bytes to stdout.

    Writing to an existing `output` path overwrites it without prompting,
    consistent across `scan` and `report` - the same convention `scan`
    has used for every structured format since it was introduced. A
    write failure is INVALID_INPUT (a bad/unwritable path is a request
    problem, not a SentinelAI bug).
    """
    if output:
        try:
            Path(output).write_text(content, encoding="utf-8")
        except OSError as exc:
            logger.error("failed to write report to '%s': %s", output, exc)
            _fail_from_exception(f"failed to write report to '{output}'", exc, ExitCode.INVALID_INPUT, debug)
        console.print(f"[bold green]Report written to[/bold green] {output}")
    else:
        # Write raw UTF-8 bytes, not console.print/print: Rich would word-wrap
        # long lines and interpret literal '[' / ']' as markup, and on
        # Windows plain print() encodes using the console codepage rather
        # than UTF-8 - both corrupt JSON/HTML/Markdown/SARIF content that's
        # meant to be captured or piped verbatim.
        sys.stdout.buffer.write((content + "\n").encode("utf-8"))


@app.command()
def scan(
    ctx: typer.Context,
    path: str = typer.Argument(
        ".", help="Path to the repository to scan (defaults to current directory)"
    ),
    quick: bool = typer.Option(False, "--quick", help="Run a quick scan (subset of checks)"),
    full: bool = typer.Option(False, "--full", help="Run a full scan (all checks)"),
    extended: bool = typer.Option(
        False,
        "--extended",
        help="Also run the dependency scanners (Trivy, OSV-Scanner) alongside Semgrep, Bandit, "
        "and GitLeaks. Off by default: the extended set needs Trivy's vulnerability database "
        "and network access to osv.dev, and reports dependency advisories that the default "
        "code-scanner set does not.",
    ),
    ai: bool = typer.Option(
        False,
        "--ai",
        help="Require AI enrichment. Fails fast with INVALID_INPUT if SENTINELAI_AI_LLM_MODEL or "
        "SENTINELAI_AI_EMBEDDING_MODEL is unset, instead of silently producing a scanner-only "
        "report. Needs a local Ollama server; no API key is involved.",
    ),
    no_ai: bool = typer.Option(
        False,
        "--no-ai",
        help="Skip AI enrichment even when both model variables are configured, for a fast, "
        "deterministic scanner-only run.",
    ),
    ai_provider: Optional[str] = typer.Option(
        None,
        "--ai-provider",
        help="LLM provider: ollama (default, local), groq, gemini or mock. Overrides "
        "SENTINELAI_AI_PROVIDER. A remote provider also needs --egress allow and its credential "
        "variable; run `sentinelai providers` to see what each one sends off the machine.",
    ),
    ai_model: Optional[str] = typer.Option(
        None,
        "--ai-model",
        help="Model identifier for the chosen provider. Overrides SENTINELAI_AI_MODEL; defaults "
        "to the provider's default model.",
    ),
    egress: Optional[str] = typer.Option(
        None,
        "--egress",
        help="allow or deny (default). Whether code slices may be sent to a remote provider. "
        "Overrides SENTINELAI_EGRESS.",
    ),
    severity: Optional[str] = typer.Option(
        None,
        "--severity",
        "-s",
        help="Only include findings at or above this severity: low, medium, high, critical. "
        "Controls what is shown/reported - does not affect the exit code.",
    ),
    fail_on: str = typer.Option(
        "none",
        "--fail-on",
        help="Exit with a security-findings exit code if findings at or above this severity are "
        "present: low, medium, high, critical, none. Default 'none' never fails the exit code "
        "based on findings - use this to opt a CI pipeline into gating.",
    ),
    format: str = typer.Option(
        "terminal",
        "--format",
        "-f",
        help="Output format: terminal, json, markdown, html, sarif",
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Write report to this file instead of printing (ignored for terminal format)",
    ),
    details: Optional[str] = typer.Option(
        None,
        "--details",
        "-d",
        help="Show the full detail view for one finding by ID, e.g. SENT-002 (terminal format only)",
    ),
    apply_patches: bool = typer.Option(
        False,
        "--apply-patches",
        help=(
            "Write AI-proposed fixes into the repository. Off by default: without this flag a "
            "scan never modifies the scanned files. Each file is backed up under "
            ".sentinelai/backups/ before it is written, and restored automatically if the write "
            "cannot be verified. Best effort per finding - a patch that fails never stops the scan."
        ),
    ),
    fix: bool = typer.Option(
        False,
        "--fix",
        help="Interactively review and apply recommended fixes for detected vulnerabilities.",
    ),
    interactive: bool = typer.Option(
        False,
        "--interactive",
        "-i",
        help="Interactively review and apply recommended fixes for detected vulnerabilities (alias for --fix).",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Automatically apply all recommended fixes without asking for confirmation.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help=(
            "With --apply-patches, compute and validate every patch and report what would "
            "change, without writing to any file. No backup is taken either, since writing one "
            "would itself modify the repository."
        ),
    ),
):
    """
    Scan a repository for vulnerabilities.

    Findings come from LiveFindingsProvider, which loads the given path
    through the backend and runs Semgrep/Bandit/GitLeaks against it via
    ScannerOrchestrator - see sentinelai/providers/live_provider.py.
    If both SENTINELAI_AI_LLM_MODEL and SENTINELAI_AI_EMBEDDING_MODEL are
    configured, every scanner finding is then run through Tanaya's frozen
    AI layer (sentinelai.ai.enrich_findings(), constructed via
    sentinelai.ai.factory's create_default_retriever()/
    create_llm_generate_fn(), given repository context rebuilt from the
    backend and adapted via sentinelai.ai.repository_context.
    from_backend_context()) to populate ai_findings before filtering,
    statistics, and rendering. If either is unconfigured, AI enrichment
    is skipped entirely and ai_findings stays empty - the same behavior
    `scan` has always had - rather than failing the command.

    The two variables must name two different Ollama models: a
    text-generation model for SENTINELAI_AI_LLM_MODEL (e.g. llama3.1:8b)
    and a dedicated embedding model for SENTINELAI_AI_EMBEDDING_MODEL
    (e.g. nomic-embed-text). A generation model cannot produce
    embeddings; setting both to the same value makes Ollama reject the
    embedding request with HTTP 501, reported here as PROVIDER_ERROR.

    Exit codes: 0 on success (including when findings exist but
    --fail-on wasn't crossed), 1 if --fail-on's threshold was crossed,
    2 for invalid input (bad path/flags), 3 if the provider or the AI
    layer failed, 4 for an unexpected internal error. See
    sentinelai/core/exit_codes.py for the full convention. AI enrichment
    failures - an unreachable Ollama server, or an embedding model that
    cannot produce embeddings - are reported as PROVIDER_ERROR (3) with
    a one-line message naming the problem and its fix, the same category
    as a scanner/backend failure, since in both cases something
    SentinelAI depends on failed rather than SentinelAI itself. Use
    --debug for the full traceback. Missing AI configuration is not
    treated as a failure at all - see above.
    """
    debug: bool = (ctx.obj or {}).get("debug", False)
    console = get_console()
    fix = fix or interactive

    repo_path = Path(path)
    if not repo_path.exists():
        _fail(f"path '{path}' does not exist.", ExitCode.INVALID_INPUT)

    if quick and full:
        _fail("--quick and --full cannot be used together.", ExitCode.INVALID_INPUT)
    mode = ScanMode.QUICK if quick else ScanMode.FULL if full else ScanMode.STANDARD

    if dry_run and not apply_patches and not fix:
        # Checked here, with the other flag-coherence rules, rather than at the
        # patch call site below: the combination can never do anything, so the
        # user should not pay for a full Semgrep/Bandit/GitLeaks run first.
        # Fails fast rather than silently doing nothing, matching how --ai
        # refuses an incoherent combination instead of guessing.
        _fail(
            "--dry-run only applies to patch application; pass --apply-patches as well, "
            "or drop --dry-run",
            ExitCode.INVALID_INPUT,
        )

    if ai and no_ai:
        _fail("--ai and --no-ai cannot be used together.", ExitCode.INVALID_INPUT)
    # Validated here, before any scanner runs, so `--ai` with an incomplete
    # configuration costs nothing. Deferring it to the enrichment block below
    # would make the user wait out a full scan before being told the run could
    # never have enriched anything.
    if ai:
        _require_ai_configuration()
    ai_flags = {"provider": ai_provider, "model": ai_model, "egress": egress}
    if no_ai and any(value is not None for value in ai_flags.values()):
        given = [flag for flag, value in zip(("--ai-provider", "--ai-model", "--egress"), ai_flags.values()) if value is not None]
        _fail(f"--no-ai cannot be used together with {', '.join(given)}.", ExitCode.INVALID_INPUT)
    # Resolved here only when AI was explicitly requested, for the same reason as
    # --ai above: fail fast rather than making the user wait out a full scan.
    #
    # Deliberately NOT resolved for every run that merely lacks --no-ai. Provider
    # settings are AI settings, so a scanner-only run must not be refused by one -
    # a stale SENTINELAI_AI_PROVIDER left in a shell would otherwise exit 2 on a
    # scan that was never going to contact any provider. The auto-detect path
    # resolves it below instead, inside the branch that can actually transmit.
    ai_config: Optional[AIConfig] = None
    if ai or any(value is not None for value in ai_flags.values()):
        ai_config = _resolve_ai_config(ai_flags)
        logger.info(
            "AI configuration: provider=%s model=%s egress=%s", ai_config.provider, ai_config.model, ai_config.egress
        )
    # Orthogonal to mode on purpose - --full does not imply --extended, so no
    # existing invocation silently acquires dependency scanning.
    tier = ScannerTier.EXTENDED if extended else ScannerTier.CORE

    format = format.lower()
    if format not in VALID_FORMATS:
        _fail(f"format must be one of {sorted(VALID_FORMATS)}, got '{format}'.", ExitCode.INVALID_INPUT)

    fail_on = fail_on.lower()
    if fail_on not in FAIL_ON_VALUES:
        _fail(f"--fail-on must be one of {sorted(FAIL_ON_VALUES)}, got '{fail_on}'.", ExitCode.INVALID_INPUT)

    min_severity: Optional[Severity] = None
    if severity:
        try:
            min_severity = Severity(severity.lower())
        except ValueError:
            _fail(
                f"severity must be one of {[s.value for s in Severity]}, got '{severity}'.",
                ExitCode.INVALID_INPUT,
            )

    if format != "terminal":
        # --details only makes sense against the terminal detail view;
        # structured formats already carry full detail for every finding.
        # Ignored silently rather than warned, so structured stdout stays
        # uncontaminated regardless of which flags were passed alongside it.
        details = None

    if format == "terminal":
        render_scan_header(
            console,
            repository=str(repo_path.resolve()),
            mode=f"{mode.value} ({tier.value} scanners)",
            version=__version__,
            started_at=datetime.now(timezone.utc),
        )

    provider = _get_provider()
    logger.info(
        "scan started: path=%s mode=%s tier=%s provider=%s",
        repo_path,
        mode.value,
        tier.value,
        provider.__class__.__name__,
    )
    started = time.monotonic()
    try:
        if format == "terminal":
            with ScanProgress(console).stage(f"Retrieving scan results ({provider.__class__.__name__})"):
                result = provider.get_scan_result(str(repo_path), mode=mode, tier=tier)
        else:
            result = provider.get_scan_result(str(repo_path), mode=mode, tier=tier)
    except Exception as exc:
        # The provider failed - a tool/integration problem, not a
        # security finding and not necessarily a SentinelAI bug.
        logger.error("scan failed: provider=%s path=%s error=%s", provider.__class__.__name__, repo_path, exc)
        _fail_from_exception("scan failed", exc, ExitCode.PROVIDER_ERROR, debug)
    elapsed = time.monotonic() - started
    result = result.model_copy(update={"metadata": result.metadata.model_copy(update={"duration_seconds": elapsed})})
    logger.info(
        "scan retrieved %d scanner findings in %.2fs", len(result.scanner_findings), elapsed
    )

    # Three states: --no-ai always skips; --ai has already been validated above
    # so it always enriches; with neither flag the pre-existing auto-detect
    # applies unchanged, which is what keeps every existing invocation - and
    # every benchmark - behaving exactly as before.
    #
    # _ai_decision() rather than get_settings(): a scanner-only run must not be
    # refused because of an AI setting it will never read, so a settings failure
    # is carried here rather than raised. It is enforced immediately below, on
    # the only branch that can transmit anything.
    #
    # --ai-provider counts as a request because passing a flag is a deliberate
    # act on this run. SENTINELAI_AI_PROVIDER deliberately does NOT: an
    # environment variable that happens to be set in a shell is configuration,
    # not consent, and letting it switch enrichment on would start sending
    # repository context to a provider the user never asked for here.
    _settings, settings_error, ai_configured = _ai_decision()
    enrich = False if no_ai else (ai or ai_configured or ai_provider is not None)
    if enrich:
        # Enforced before any AI request is built, so nothing is sent on a bad
        # host. The --ai path has already been through this at
        # _require_ai_configuration(); the auto-detect path reaches it here.
        if settings_error is not None:
            _fail(_settings_error_message(settings_error), ExitCode.INVALID_INPUT)
        # The auto-detect path has not resolved provider configuration yet, because
        # doing so before this point would refuse scanner-only runs. Resolved now,
        # still before any request is built, so a bad provider/egress combination
        # is reported as INVALID_INPUT rather than attempted.
        if ai_config is None:
            ai_config = _resolve_ai_config(ai_flags)
            logger.info(
                "AI configuration: provider=%s model=%s egress=%s",
                ai_config.provider, ai_config.model, ai_config.egress,
            )
        # Rebuilds repository context via the backend rather than threading it through
        # FindingsProvider: LiveFindingsProvider already builds one internally, but
        # ScanResult (the frozen FindingsProvider contract) has no field to carry it
        # out, and adding one would be exactly the new abstraction this integration is
        # meant to avoid. The cost is a second backend pass (git metadata, language
        # detection, dependency/snippet extraction) when AI is configured - accepted
        # rather than changing the provider contract for it.
        logger.info("AI enrichment started: %d findings", len(result.scanner_findings))
        faults = []
        llm_provider = None
        role_gen = None
        # The SecretIndex is built here, once, from every scanner finding, before any prompt.
        with scan_session(result.scanner_findings, ai_config) as session:
            try:
                if ai_config.role_chains and create_llm_provider is _default_create_llm_provider:
                    from .ai import factory as _ai_factory
                    _role_gen_fn = (
                        create_role_generator
                        if create_role_generator is not _default_create_role_generator
                        else _ai_factory.create_role_generator
                    )
                    role_gen = _role_gen_fn(
                        ai_config,
                        temperature=AI_TEMPERATURE,
                        max_tokens=AI_MAX_TOKENS,
                    )
                    generate = guarded_generate(
                        role_gen,
                        ai_config,
                        temperature=AI_TEMPERATURE,
                        max_tokens=AI_MAX_TOKENS,
                    )
                else:
                    llm_provider = create_llm_provider(ai_config)
                    generate = guarded_generate(
                        llm_provider,
                        ai_config,
                        temperature=AI_TEMPERATURE,
                        max_tokens=AI_MAX_TOKENS,
                    )
                backend_context = build_repository_context(load_repository(str(repo_path)))
                ai_findings = enrich_findings(
                    result.scanner_findings,
                    create_retriever_for_scan(),
                    generate,
                    score_confidence,
                    verify_finding,
                    repository_context=from_backend_context(backend_context),
                    # Enrich once per correlated issue rather than once per raw hit.
                    correlated_findings=result.correlated_findings,
                    on_failure=lambda finding_id, exc: faults.append(ai_fault(finding_id, exc)),
                )
            except Exception as exc:
                # Rule 3: a provider or enrichment failure is an operational fault recorded in
                # scan metadata, never a finding, and never a failed scan - the findings simply
                # come back unenriched. The pipeline raises only once every finding has failed,
                # each already recorded through on_failure, so that summary is not recorded twice.
                logger.error("AI enrichment failed: path=%s error=%s", repo_path, exc)
                print_error(f"AI enrichment failed; continuing without it: {exc}")
                if debug:
                    print_traceback()
                if not (faults and isinstance(exc, AIEnrichmentError)):
                    faults.append(ai_fault(None, exc))
                ai_findings = []
        role_prov_list = []
        override_retention_status = None
        if role_gen is not None:
            role_prov_list = role_gen.get_role_provenance()
            for rf in role_gen.get_faults():
                if rf not in faults:
                    faults.append(rf)
            disc_entry = next((rp for rp in role_prov_list if rp.role == "discovery"), None)
            primary_entry = disc_entry or (role_prov_list[0] if role_prov_list else None)
            active_name = (
                primary_entry.provider
                if primary_entry is not None
                else (role_gen.active_discovery_provider or ai_config.provider)
            )
            active_model = (
                primary_entry.model
                if primary_entry is not None
                else (role_gen.active_discovery_model or ai_config.model)
            )
            spec = PROVIDERS.get(active_name)
            active_retains = spec.retains_data if spec is not None else None
            active_is_local = spec.is_local if spec is not None else None
            if role_prov_list:
                override_retention_status = aggregate_retention_status(
                    [rp.retention_status for rp in role_prov_list]
                )
        elif llm_provider is not None:
            active_name = getattr(llm_provider, "active_provider_name", ai_config.provider)
            active_model = getattr(llm_provider, "active_model_name", ai_config.model)
            active_retains = getattr(llm_provider, "retains_data", None)
            active_is_local = getattr(llm_provider, "is_local", None)
            if hasattr(llm_provider, "fallback_faults"):
                for ff in llm_provider.fallback_faults:
                    if ff not in faults:
                        faults.append(ff)
        else:
            active_name = ai_config.provider
            active_model = ai_config.model
            spec = PROVIDERS.get(ai_config.provider)
            active_retains = spec.retains_data if spec is not None else None
            active_is_local = spec.is_local if spec is not None else None
        provenance = {
            "ai_provider": active_name,
            "ai_model": active_model,
            "egress_policy": ai_config.egress,
            "retention_status": override_retention_status
            or retention_status_from_provider(active_retains, is_local=active_is_local),
            "redaction": session.redaction_report(),
            "role_provenance": role_prov_list,
            "ai_faults": faults,
        }
        result = result.model_copy(
            update={"ai_findings": ai_findings, "metadata": result.metadata.model_copy(update=provenance)}
        )
        logger.info(
            "AI enrichment completed: %d findings enriched (provider=%s model=%s)",
            len(ai_findings), provenance["ai_provider"], provenance["ai_model"],
        )
    elif no_ai:
        logger.info("AI enrichment skipped: --no-ai")
    else:
        logger.info("AI enrichment skipped: not configured (SENTINELAI_AI_LLM_MODEL/SENTINELAI_AI_EMBEDDING_MODEL not set)")

    # Opt-in patch application. Deliberately after enrichment and before reporting:
    # the patches come from AI findings, and the report should describe the scan
    # that was run rather than the state after it was modified. `patch_run` is
    # retained rather than rendered - report generation is unchanged in this phase,
    # and a later one will consume this object.
    #
    # sentinelai.patching.run_patches() owns the per-finding loop; this call site
    # neither validates, backs up, applies nor restores anything. Failures are
    # recorded per finding and never reach the exit code, so --apply-patches
    # cannot turn a completed scan into a failed one.
    patch_run = None
    if apply_patches:
        patch_run = run_patches(result.ai_findings, root=repo_path, dry_run=dry_run)
        logger.info(
            "patch application%s: %d applied, %d skipped, %d failed, %d rolled back",
            " (dry run - nothing written)" if dry_run else "",
            len(patch_run.applied),
            len(patch_run.skipped),
            len(patch_run.failed),
            len(patch_run.rolled_back),
        )
    # One projection, shared by every format - see reporting/patch_section.py.
    patch_application = build_patch_application_report(patch_run)

    if min_severity is not None:
        result = filter_by_severity(result, min_severity)

    # Computed once, after filtering, so both the terminal summary and the
    # JSON report reflect the same (possibly filtered) result - one source
    # of truth, never recalculated per format.
    stats = calculate_statistics(result)

    if format == "terminal":
        if not result.scanner_findings and not result.ai_findings:
            console.print("[bold green]No findings matched your filters.[/bold green]")
            console.print(escape(provenance_line(result.metadata)))
            raise typer.Exit(code=ExitCode.SUCCESS)

        if details:
            match = next((f for f in result.scanner_findings if f.finding_id == details), None)
            if match is None:
                _fail(f"no finding with ID '{details}' in this scan result.", ExitCode.INVALID_INPUT)
            ai_match = next((a for a in result.ai_findings if a.finding_id == details), None)
            render_finding_detail(console, match, ai_match)
            return

        if output:
            console.print(
                "[yellow]Note:[/yellow] --output is ignored for terminal format. "
                "Use --format json/markdown/html/sarif to save a report."
            )

        if fix:
            run_remediation_session(
                repo_path=repo_path,
                result=result,
                console=console,
                auto_yes=yes,
                dry_run=dry_run,
            )
            return

        render_summary(console, stats, result.metadata)
        render_findings_table(console, result)
        render_patch_application(console, patch_application)

        if exceeds_fail_on_threshold(stats, fail_on):
            raise typer.Exit(code=ExitCode.SECURITY_FINDINGS)
        return

    try:
        content = _render_structured_content(format, result, stats, patch_application)
    except Exception as exc:
        logger.error("report generation failed: format=%s error=%s", format, exc)
        _fail_from_exception("report generation failed", exc, ExitCode.INTERNAL_ERROR, debug)

    _write_or_print(console, content, output, debug)

    if exceeds_fail_on_threshold(stats, fail_on):
        raise typer.Exit(code=ExitCode.SECURITY_FINDINGS)


@app.command()
def providers() -> None:
    """
    List every LLM provider: its default model, whether it runs locally or remotely,
    its data-retention status, and whether its credentials are set.

    Run this before a scan to see what would leave the machine. Makes no network
    call and never reads a credential's value.
    """
    render_providers(get_console())


@app.command()
def fix(
    ctx: typer.Context,
    path: str = typer.Argument(
        ".", help="Path to the repository to scan and fix (defaults to current directory)"
    ),
    quick: bool = typer.Option(False, "--quick", help="Run a quick scan (subset of checks)"),
    full: bool = typer.Option(False, "--full", help="Run a full scan (all checks)"),
    extended: bool = typer.Option(
        False,
        "--extended",
        help="Also run the dependency scanners (Trivy, OSV-Scanner) alongside Semgrep, Bandit, and GitLeaks.",
    ),
    ai: bool = typer.Option(
        False,
        "--ai",
        help="Require AI enrichment (local Ollama server).",
    ),
    no_ai: bool = typer.Option(
        False,
        "--no-ai",
        help="Skip AI enrichment for a deterministic rule-based fix run.",
    ),
    severity: Optional[str] = typer.Option(
        None,
        "--severity",
        "-s",
        help="Only review findings at or above this severity: low, medium, high, critical.",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Automatically apply all recommended fixes without asking for confirmation.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Simulate the remediation session without modifying files on disk.",
    ),
):
    """
    Scan a repository, narrate detected vulnerabilities, and interactively apply fixes.
    """
    scan(
        ctx=ctx,
        path=path,
        quick=quick,
        full=full,
        extended=extended,
        ai=ai,
        no_ai=no_ai,
        # Every scan() parameter is passed explicitly: called as a plain function, an
        # omitted one receives its typer.Option(...) object, which is truthy and not None.
        ai_provider=None,
        ai_model=None,
        egress=None,
        severity=severity,
        fail_on="none",
        format="terminal",
        output=None,
        details=None,
        apply_patches=False,
        fix=True,
        interactive=False,
        yes=yes,
        dry_run=dry_run,
    )


@app.command()
def undo(
    ctx: typer.Context,
    path: str = typer.Argument(
        ".", help="Path to the repository whose latest remediation changes should be undone"
    ),
):
    """
    Revert the most recent set of code changes applied by SentinelAI remediation.
    """
    console = get_console()
    repo_path = Path(path)
    if not repo_path.exists():
        _fail(f"path '{path}' does not exist.", ExitCode.INVALID_INPUT)

    from .patcher import restore_latest_backup_session

    reverted = restore_latest_backup_session(repo_path)
    if not reverted:
        console.print("[yellow]No recent SentinelAI backup found to revert in this repository.[/yellow]")
        return

    console.print(f"\n[bold green]=== SentinelAI Rollback ({len(reverted)} file(s) restored) ===[/bold green]\n")
    for file_name, success in reverted:
        if success:
            console.print(f"[green][RESTORED] {file_name}[/green]")
        else:
            console.print(f"[red][FAILED] Could not restore {file_name}[/red]")


@app.command()
def report(
    ctx: typer.Context,
    input_path: str = typer.Argument(
        ..., help="Path to a previously saved SentinelAI JSON report (see 'scan --format json')"
    ),
    format: str = typer.Option(
        "json",
        "--format",
        "-f",
        help="Output format: json, markdown, html, sarif",
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Write report to this file instead of printing",
    ),
):
    """
    Render an existing saved scan result as a report.

    Unlike `scan`, this command never runs a scan: it only reads a JSON
    report previously produced by `scan --format json`, reconstructs the
    underlying ScanResult, and renders it - no scanner, repository
    analysis, RAG, LLM reasoning, or verification runs again. Statistics
    are always recomputed from the reconstructed findings, not trusted
    from the input file, so a hand-edited report can't produce a report
    with inconsistent numbers.

    Exit codes: 0 on success, 2 for invalid input (bad path, malformed
    JSON, unsupported schema version, unwritable output path), 4 for an
    unexpected internal error. See sentinelai/core/exit_codes.py.
    """
    debug: bool = (ctx.obj or {}).get("debug", False)
    console = get_console()

    report_format = format.lower()
    if report_format not in REPORT_FORMATS:
        _fail(f"format must be one of {sorted(REPORT_FORMATS)}, got '{report_format}'.", ExitCode.INVALID_INPUT)

    try:
        result, stats = load_scan_result(Path(input_path))
    except ReportLoadError as exc:
        logger.error("report failed: path=%s error=%s", input_path, exc)
        _fail(str(exc), ExitCode.INVALID_INPUT)

    try:
        content = _render_structured_content(
            report_format, result, stats, load_patch_application(Path(input_path))
        )
    except Exception as exc:
        logger.error("report generation failed: format=%s error=%s", report_format, exc)
        _fail_from_exception("report generation failed", exc, ExitCode.INTERNAL_ERROR, debug)

    _write_or_print(console, content, output, debug)


@app.command()
def version():
    """Show the SentinelAI CLI version."""
    console = get_console()
    console.print(f"SentinelAI CLI v{__version__}")


if __name__ == "__main__":
    app()
