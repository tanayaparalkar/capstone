"""
Provider boundary.

FindingsProvider is the abstraction the CLI depends on. It exists so the
CLI, reporting, and statistics code never talk to a data source directly:
today MockFindingsProvider (mock_provider.py) is the only implementation;
once Nitaanth's backend is ready, a second implementation (e.g. calling
his /scan endpoint) is a drop-in replacement, and nothing above this
boundary needs to change.
"""
from abc import ABC, abstractmethod

from sentinelai.contracts import ScanMode, ScanResult


class FindingsProvider(ABC):
    """Source of a ScanResult for a given repository."""

    @abstractmethod
    def get_scan_result(self, repo_path: str, mode: ScanMode = ScanMode.STANDARD) -> ScanResult:
        """Return a fully-populated ScanResult for the given repository path."""
        raise NotImplementedError
