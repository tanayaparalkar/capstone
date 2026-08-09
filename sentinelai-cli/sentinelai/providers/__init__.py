"""Provider boundary: source of ScanResult objects for the CLI."""
from .base import FindingsProvider
from .mock_provider import MockFindingsProvider

__all__ = ["FindingsProvider", "MockFindingsProvider"]
