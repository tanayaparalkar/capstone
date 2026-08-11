"""
Abstract scanner base - the shape every concrete scanner (Semgrep,
Bandit, GitLeaks, Trivy, OSV) will implement, and nothing else.

Takes a RepositoryContext (backend/context_builder.py), not a raw path:
every filesystem walk a scanner could need - the repository location,
detected languages, declared dependencies, code snippets - is already
exposed by the existing backend modules, so this framework performs none
of its own. A concrete scanner that genuinely needs to invoke an external
tool against the repository on disk still has what it needs, since
RepositoryContext carries the LoadedRepository (and so absolute_path)
through.

No scanner-specific knowledge lives here: no mention of Semgrep, Bandit,
severity mapping, subprocess execution, or Docker, and no API keys or
environment variables are read. Those belong to each concrete scanner
module, none of which exist yet.
"""
from abc import ABC, abstractmethod
from typing import List

from sentinelai.backend.context_builder import RepositoryContext
from sentinelai.contracts import ScannerFinding


class Scanner(ABC):
    """A single static-analysis tool that turns a RepositoryContext into ScannerFinding objects."""

    @abstractmethod
    def scan(self, context: RepositoryContext) -> List[ScannerFinding]:
        """Run this scanner against `context` and return the findings it produces."""
        raise NotImplementedError
