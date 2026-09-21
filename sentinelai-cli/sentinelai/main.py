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
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from . import __version__
from .ai import enrich_findings
from .ai.config import get_settings
from .ai.confidence_scorer import score_confidence
from .ai.factory import create_default_retriever, create_llm_generate_fn
from .ai.repository_context import from_backend_context
from .ai.verifier import verify_finding
from .backend.context_builder import build_repository_context
from .patching import run_patches
from .backend.loader import load_repository
from .contracts import ScanMode, ScannerTier, ScanResult, Severity
from .core import ExitCode, exceeds_fail_on_threshold, filter_by_severity
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
from .providers import FindingsProvider, LiveFindingsProvider
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

# Operational logging, separate from the user-facing console output above:
# silent by default (no handler is configured here - that's left to
# whoever deploys/embeds SentinelAI, matching stdlib logging convention),
# so this adds no visible output and changes no existing behavior. It
# exists purely so an operator who *does* configure a handler gets a
# trail of scan starts/completions/failures for unattended runs.
logger = logging.getLogger("sentinelai")

VALID_FORMATS = {"terminal", "json", "markdown", "html", "sarif"}
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
    settings = get_settings()
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


def _render_structured_content(
    format: str,
    result: ScanResult,
    stats: ScanStatistics,
    patch_application=None,
) -> str:
    """Dispatch to the appropriate report generator - shared by `scan` and `report`."""
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

    repo_path = Path(path)
    if not repo_path.exists():
        _fail(f"path '{path}' does not exist.", ExitCode.INVALID_INPUT)

    if quick and full:
        _fail("--quick and --full cannot be used together.", ExitCode.INVALID_INPUT)
    mode = ScanMode.QUICK if quick else ScanMode.FULL if full else ScanMode.STANDARD

    if ai and no_ai:
        _fail("--ai and --no-ai cannot be used together.", ExitCode.INVALID_INPUT)
    # Validated here, before any scanner runs, so `--ai` with an incomplete
    # configuration costs nothing. Deferring it to the enrichment block below
    # would make the user wait out a full scan before being told the run could
    # never have enriched anything.
    if ai:
        _require_ai_configuration()
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

    settings = get_settings()
    # Three states, one expression: --no-ai always skips; --ai has already been
    # validated above so it always enriches; with neither flag the pre-existing
    # auto-detect applies unchanged, which is what keeps every existing
    # invocation - and every benchmark - behaving exactly as before.
    ai_configured = settings.llm_model is not None and settings.embedding_model is not None
    enrich = False if no_ai else (True if ai else ai_configured)
    if enrich:
        # Rebuilds repository context via the backend rather than threading it through
        # FindingsProvider: LiveFindingsProvider already builds one internally, but
        # ScanResult (the frozen FindingsProvider contract) has no field to carry it
        # out, and adding one would be exactly the new abstraction this integration is
        # meant to avoid. The cost is a second backend pass (git metadata, language
        # detection, dependency/snippet extraction) when AI is configured - accepted
        # rather than changing the provider contract for it.
        logger.info("AI enrichment started: %d findings", len(result.scanner_findings))
        try:
            backend_context = build_repository_context(load_repository(str(repo_path)))
            ai_findings = enrich_findings(
                result.scanner_findings,
                create_default_retriever(),
                create_llm_generate_fn(),
                score_confidence,
                verify_finding,
                repository_context=from_backend_context(backend_context),
                # Enrich once per correlated issue rather than once per raw hit.
                correlated_findings=result.correlated_findings,
            )
        except Exception as exc:
            # Same category as a provider failure: something SentinelAI depends on -
            # the Ollama server, the configured embedding model - failed, not
            # SentinelAI itself. Without this, the single most likely AI
            # misconfiguration (a text-generation model set as the embedding model,
            # which Ollama answers with HTTP 501) reached the user as a raw traceback.
            # Caught broadly rather than on AIEnrichmentError alone so that failures
            # from outside the AI layer in this same block - an unreadable repository
            # path, a missing knowledge base - are reported the same clean way.
            # --debug still prints the full traceback.
            logger.error("AI enrichment failed: path=%s error=%s", repo_path, exc)
            _fail_from_exception("AI enrichment failed", exc, ExitCode.PROVIDER_ERROR, debug)
        result = result.model_copy(update={"ai_findings": ai_findings})
        logger.info("AI enrichment completed: %d findings enriched", len(ai_findings))
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
    if dry_run and not apply_patches:
        # Fails fast rather than silently doing nothing, matching how --ai
        # refuses an incoherent combination instead of guessing.
        _fail(
            "--dry-run only applies to patch application; pass --apply-patches as well, "
            "or drop --dry-run",
            ExitCode.INVALID_INPUT,
        )

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

        render_summary(console, stats)
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
