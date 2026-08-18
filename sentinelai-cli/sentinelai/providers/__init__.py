"""Provider boundary: source of ScanResult objects for the CLI."""
from .base import FindingsProvider
from .live_provider import LiveFindingsProvider
from .mock_provider import MockFindingsProvider

__all__ = ["FindingsProvider", "MockFindingsProvider", "LiveFindingsProvider"]
