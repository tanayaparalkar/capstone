"""
Scanner orchestrator - runs an explicit, caller-selected set of scanners
against one RepositoryContext and concatenates their findings.

Accepts Iterable[Scanner] - the scanner abstraction only (base.py), not
ScannerRegistration or ScannerRegistry. This module's job is to execute
scanners that have already been selected; it has no reason to know they
came from a registry, or that a registry exists at all. An earlier
version accepted ScannerRegistration for its name, sorting by name before
running. On reconsideration that coupling wasn't earning its keep: the
name was never used for anything but that sort key (errors below are
never wrapped, so no message ever referenced it either), and "which
order is canonical" is a selection-adjacent policy decision that belongs
with the caller, not here.

Ordering is therefore not derived here at all - the given scanners are
run in the order the caller provides, unmodified. That still satisfies
"deterministic order": this module introduces no nondeterminism of its
own (no set, no dict, a plain iteration), so a deterministic input
produces a deterministic output. Guaranteeing that input is deterministic
in the first place (e.g. by sorting a ScannerRegistry.list_scanners()
result, or a mode-filtered subset of it) is the caller's job - typically
LiveFindingsProvider.

No try/except around scan(): mirrors backend/context_builder.py, the
direct precedent for this exact shape (call a fixed set of things once
each, concatenate/bundle results). Every wrapper's ScannerExecutionError
already names its own tool in its message ("Failed to run semgrep...",
"bandit exited with status..."), so wrapping would only be redundant -
and fail-closed, as already settled, means one scanner failing must stop
the whole run rather than being caught and skipped. The first failure
propagates unchanged; scanners after it in the given order simply never
run.

A class, not a function, unlike context_builder.py's single top-level
function: this module has a name in the frozen architecture
(ScannerOrchestrator) that a future LiveFindingsProvider is expected to
hold as a component ("a future LiveFindingsProvider can internally use a
ScannerOrchestrator"), the same way MockFindingsProvider already holds
its own constructed state. No constructor arguments are needed today, so
none are added - an instance method rather than a speculative __init__.

No AI, reporting, CLI, or repository-loading logic - and no ScanMode
logic - lives here, matching every constraint in the frozen architecture
for this module.
"""
from typing import Iterable, List

from sentinelai.backend.context_builder import RepositoryContext
from sentinelai.contracts import ScannerFinding

from .base import Scanner


class ScannerOrchestrator:
    """Runs a caller-selected, caller-ordered set of scanners against a RepositoryContext."""

    def run(self, context: RepositoryContext, scanners: Iterable[Scanner]) -> List[ScannerFinding]:
        """Invoke each of `scanners` exactly once, in the given order, and concatenate their findings."""
        findings: List[ScannerFinding] = []
        for scanner in scanners:
            findings.extend(scanner.scan(context))
        return findings
