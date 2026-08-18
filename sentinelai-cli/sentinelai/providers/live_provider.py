"""
Live findings provider - the real FindingsProvider implementation,
composing the backend and scanner framework instead of reading a bundled
fixture (mock_provider.py).

Responsibilities are exactly the sequence FindingsProvider.get_scan_result()
implies and nothing more: load the repository, build its RepositoryContext,
select and run scanners through ScannerOrchestrator, and assemble the
result. No AI, reporting, statistics, CLI, or prompt-generation logic -
none of that is reachable from here, matching every other module in
backend/ and scanners/.

Scanner selection is by *tier*, not by mode. ScannerTier.CORE (the
default) runs the three code scanners - Semgrep, Bandit, GitLeaks - which
is exactly what every scan ran before the tier existed; EXTENDED adds the
two dependency scanners, Trivy and OSV-Scanner. Selection is a single
filtered lookup against the registry, with no per-tool heuristics.

Mode still selects nothing, for the reason it never did: QUICK/STANDARD/
FULL are a depth control, and there is still no benchmark or
product-specified criterion for calling one of the three code scanners
"quick" - inventing one would be exactly the kind of per-tool guess this
project has consistently avoided (see e.g. gitleaks.py's severity
mapping). Folding dependency scanning into FULL instead of giving it its
own axis was considered and rejected: it would silently change what an
existing `--full` scan means. _select_scanners() takes both parameters,
so mode remains available as the seam for a genuine depth decision later.

The scanner registry is constructed once per LiveFindingsProvider
instance by default (the three real wrappers, matching how
MockFindingsProvider defaults its own data source), but is injectable via
the constructor - the same optional-override-with-a-default shape
MockFindingsProvider already uses for `data_path`. This is what makes the
provider testable without invoking real scanner executables: tests inject
a ScannerRegistry populated with dummy Scanner subclasses instead of
patching subprocess.run three times over.

RepositoryInfo mapping reuses backend output directly rather than
recomputing anything: `name`/`commit_hash`/`branch` come straight from
RepositoryContext.repository/.git_metadata, and `languages` is
`[l.name for l in context.languages]` - detect_languages() already
returns them in its own deterministic order (descending file_count, then
name), so this mapping doesn't re-sort or otherwise transform that
ordering. `path` is the raw `repo_path` argument as given, matching
mock_provider.py's own `path=str(repo_path)` precedent exactly, not a
resolved/absolute path. `duration_seconds` is deliberately left unset,
also matching mock_provider.py: main.py already fills it in after
get_scan_result() returns (`elapsed = time.monotonic() - started`), so
setting it here would just be overwritten and duplicates a timer this
module has no business owning.

Error handling: the entire body is one try/except around Exception,
translated into ProviderError with the original chained
(`raise ProviderError(...) from exc`) - mirroring main.py's own existing
`except Exception as exc: _fail_from_exception(...)` around
provider.get_scan_result() calls, the direct precedent for "catch broadly
at this exact boundary, wrap and chain." This covers backend exceptions
(RepositoryError and anything build_repository_context() lets propagate,
e.g. tomllib.TOMLDecodeError, git.exc.GitCommandNotFound - see its own
docstring) and scanner exceptions (ScannerExecutionError) uniformly, so
callers depend only on ProviderError, never on backend/scanner exception
types. Nothing here catches per-scanner: ScannerOrchestrator already
fails closed on the first scanner failure (backend/context_builder.py's
sibling "no try/except, propagates unchanged" shape), so there is nothing
to add at this layer beyond translating whatever comes out.
"""
import logging
from datetime import datetime, timezone
from typing import List, Optional

from sentinelai.backend.context_builder import build_repository_context
from sentinelai.backend.loader import load_repository
from sentinelai.contracts import (
    RepositoryInfo,
    ScanMetadata,
    ScanMode,
    ScannerTier,
    ScanResult,
)
from sentinelai.core.errors import ProviderError
from sentinelai.correlation import correlate_findings
from sentinelai.scanners.bandit import BanditScanner
from sentinelai.scanners.base import Scanner
from sentinelai.scanners.gitleaks import GitleaksScanner
from sentinelai.scanners.orchestrator import ScannerOrchestrator
from sentinelai.scanners.osv import OSVScanner
from sentinelai.scanners.registry import ScannerRegistry
from sentinelai.scanners.semgrep import SemgrepScanner
from sentinelai.scanners.trivy import TrivyScanner

from .base import FindingsProvider

# Silent by default (no handler configured here, same convention as
# sentinelai.main's logger) - adds an operational trail for whoever
# configures logging, without changing any existing observable behavior.
logger = logging.getLogger("sentinelai")


def _default_registry() -> ScannerRegistry:
    """Every scanner this project ships, each tagged with the tier it belongs to.

    Registering all five here rather than building a different registry per
    tier keeps registration in one place and makes the tier the only thing
    that decides what runs - so "which scanners exist" and "which scanners run
    by default" stay separate questions with separate answers.

    Trivy and OSV-Scanner are EXTENDED, not CORE. They scan declared
    dependencies rather than source code, they need a vulnerability database
    (Trivy) or network access to osv.dev (OSV) that the core three do not, and
    the project's frozen performance baseline is measured against the core
    three alone. Making them opt-in is what keeps that baseline meaningful.
    """
    registry = ScannerRegistry()
    registry.register("semgrep", SemgrepScanner(), ScannerTier.CORE)
    registry.register("bandit", BanditScanner(), ScannerTier.CORE)
    registry.register("gitleaks", GitleaksScanner(), ScannerTier.CORE)
    registry.register("trivy", TrivyScanner(), ScannerTier.EXTENDED)
    registry.register("osv-scanner", OSVScanner(), ScannerTier.EXTENDED)
    return registry


class LiveFindingsProvider(FindingsProvider):
    """FindingsProvider backed by the real backend and scanner framework."""

    def __init__(self, registry: Optional[ScannerRegistry] = None) -> None:
        self._registry = registry or _default_registry()
        self._orchestrator = ScannerOrchestrator()

    def get_scan_result(
        self,
        repo_path: str,
        mode: ScanMode = ScanMode.STANDARD,
        tier: ScannerTier = ScannerTier.CORE,
    ) -> ScanResult:
        logger.info(
            "LiveFindingsProvider: scanning '%s' mode=%s tier=%s", repo_path, mode.value, tier.value
        )
        try:
            repository = load_repository(repo_path)
            context = build_repository_context(repository)
            scanners = self._select_scanners(mode, tier)
            scanner_findings = self._orchestrator.run(context, scanners)
            logger.info(
                "LiveFindingsProvider: '%s' produced %d scanner findings", repo_path, len(scanner_findings)
            )

            return ScanResult(
                repository=RepositoryInfo(
                    name=context.repository.name,
                    path=repo_path,
                    commit_hash=context.git_metadata.commit_hash,
                    branch=context.git_metadata.branch,
                    languages=[language.name for language in context.languages],
                ),
                metadata=ScanMetadata(
                    timestamp=datetime.now(timezone.utc), mode=mode, scanner_tier=tier
                ),
                scanner_findings=scanner_findings,
                ai_findings=[],
                # Deterministic and offline, so it runs in scanner-only mode too -
                # correlation needs no model and no configuration.
                correlated_findings=correlate_findings(scanner_findings),
            )
        except Exception as exc:
            logger.error("LiveFindingsProvider: scan of '%s' failed: %s", repo_path, exc)
            raise ProviderError(f"Live scan of '{repo_path}' failed: {exc}") from exc

    def _select_scanners(self, mode: ScanMode, tier: ScannerTier = ScannerTier.CORE) -> List[Scanner]:
        """Return the scanners to run for `mode` and `tier`.

        `tier` is what actually selects today; `mode` still selects nothing,
        for the reason the module docstring gives. The two are kept as separate
        parameters rather than collapsed because they answer different
        questions - see ScannerTier's own docstring.

        Ordering comes from ScannerRegistry.list_scanners(), which sorts by
        name, so the orchestrator receives a deterministic sequence exactly as
        it did before.
        """
        return [registration.scanner for registration in self._registry.list_scanners(tier)]
