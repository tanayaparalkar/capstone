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

Scanner selection by mode is a real decision, not a placeholder: today
every mode (QUICK/STANDARD/FULL) runs every registered scanner. There are
only three general-purpose scanners (Semgrep, Bandit, GitLeaks) and no
benchmark or product-specified criteria yet for excluding any of them
from a faster tier - inventing one (e.g. guessing GitLeaks is "quick")
would be exactly the kind of per-tool heuristic this project has
consistently avoided elsewhere (see e.g. gitleaks.py's severity mapping).
_select_scanners() is the seam for this once real criteria exist (heavier
tools like Trivy/OSV added as FULL-only, or timing data justifying a
QUICK subset) - it already takes `mode`, so extending it later doesn't
require restructuring get_scan_result().

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
from sentinelai.contracts import RepositoryInfo, ScanMetadata, ScanMode, ScanResult
from sentinelai.core.errors import ProviderError
from sentinelai.correlation import correlate_findings
from sentinelai.scanners.bandit import BanditScanner
from sentinelai.scanners.base import Scanner
from sentinelai.scanners.gitleaks import GitleaksScanner
from sentinelai.scanners.orchestrator import ScannerOrchestrator
from sentinelai.scanners.registry import ScannerRegistry
from sentinelai.scanners.semgrep import SemgrepScanner

from .base import FindingsProvider

# Silent by default (no handler configured here, same convention as
# sentinelai.main's logger) - adds an operational trail for whoever
# configures logging, without changing any existing observable behavior.
logger = logging.getLogger("sentinelai")


def _default_registry() -> ScannerRegistry:
    registry = ScannerRegistry()
    registry.register("semgrep", SemgrepScanner())
    registry.register("bandit", BanditScanner())
    registry.register("gitleaks", GitleaksScanner())
    return registry


class LiveFindingsProvider(FindingsProvider):
    """FindingsProvider backed by the real backend and scanner framework."""

    def __init__(self, registry: Optional[ScannerRegistry] = None) -> None:
        self._registry = registry or _default_registry()
        self._orchestrator = ScannerOrchestrator()

    def get_scan_result(self, repo_path: str, mode: ScanMode = ScanMode.STANDARD) -> ScanResult:
        logger.info("LiveFindingsProvider: scanning '%s' mode=%s", repo_path, mode.value)
        try:
            repository = load_repository(repo_path)
            context = build_repository_context(repository)
            scanners = self._select_scanners(mode)
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
                metadata=ScanMetadata(timestamp=datetime.now(timezone.utc), mode=mode),
                scanner_findings=scanner_findings,
                ai_findings=[],
                # Deterministic and offline, so it runs in scanner-only mode too -
                # correlation needs no model and no configuration.
                correlated_findings=correlate_findings(scanner_findings),
            )
        except Exception as exc:
            logger.error("LiveFindingsProvider: scan of '%s' failed: %s", repo_path, exc)
            raise ProviderError(f"Live scan of '{repo_path}' failed: {exc}") from exc

    def _select_scanners(self, mode: ScanMode) -> List[Scanner]:
        """Return the scanners to run for `mode` - see module docstring for why every mode is identical today."""
        return [registration.scanner for registration in self._registry.list_scanners()]
