"""
Mock findings provider.

Reads the bundled sample scanner findings and wraps them in a ScanResult.
No AI enrichment is produced here - that's Tanaya's layer's job, so
ai_findings is always empty for the mock provider. This exists purely so
the CLI, reporting, and statistics code can be built and demoed before
Nitaanth's and Tanaya's real modules are ready.
"""
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from sentinelai.contracts import (
    RepositoryInfo,
    ScanMetadata,
    ScanMode,
    ScannerFinding,
    ScannerTier,
    ScanResult,
)
from sentinelai.correlation import correlate_findings

from .base import FindingsProvider

DEFAULT_MOCK_DATA_PATH = Path(__file__).parent / "mock_data" / "scanner_findings.json"


class MockFindingsProvider(FindingsProvider):
    """FindingsProvider backed by a bundled JSON fixture instead of real scanners."""

    def __init__(self, data_path: Optional[Path] = None) -> None:
        self._data_path = data_path or DEFAULT_MOCK_DATA_PATH

    def get_scan_result(
        self,
        repo_path: str,
        mode: ScanMode = ScanMode.STANDARD,
        tier: ScannerTier = ScannerTier.CORE,
    ) -> ScanResult:
        # `tier` is accepted to satisfy the FindingsProvider signature and
        # deliberately ignored: this provider runs no scanners, so there is no
        # scanner set for it to select. It is still recorded in the metadata
        # below so a mock report states which tier was asked for.
        raw = json.loads(self._data_path.read_text(encoding="utf-8"))
        scanner_findings = [ScannerFinding(**item) for item in raw]
        repo_path_obj = Path(repo_path)

        return ScanResult(
            repository=RepositoryInfo(
                name=repo_path_obj.name or str(repo_path_obj),
                path=str(repo_path),
            ),
            metadata=ScanMetadata(
                timestamp=datetime.now(timezone.utc),
                mode=mode,
                scanner_tier=tier,
            ),
            scanner_findings=scanner_findings,
            ai_findings=[],
            correlated_findings=correlate_findings(scanner_findings),
        )
