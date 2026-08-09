"""Unit tests for the mock findings provider."""
from sentinelai.contracts import ScanMode, ScanResult, ScannerFinding
from sentinelai.providers import FindingsProvider, MockFindingsProvider


def test_mock_provider_is_a_findings_provider():
    provider = MockFindingsProvider()
    assert isinstance(provider, FindingsProvider)


def test_mock_provider_returns_scan_result():
    provider = MockFindingsProvider()
    result = provider.get_scan_result("/tmp/some-repo")
    assert isinstance(result, ScanResult)


def test_mock_provider_populates_scanner_findings():
    provider = MockFindingsProvider()
    result = provider.get_scan_result("/tmp/some-repo")
    assert len(result.scanner_findings) == 10
    assert all(isinstance(f, ScannerFinding) for f in result.scanner_findings)


def test_mock_provider_leaves_ai_findings_empty():
    provider = MockFindingsProvider()
    result = provider.get_scan_result("/tmp/some-repo")
    assert result.ai_findings == []


def test_mock_provider_records_repository_path():
    provider = MockFindingsProvider()
    result = provider.get_scan_result("/tmp/some-repo")
    assert result.repository.path == "/tmp/some-repo"
    assert result.repository.name == "some-repo"


def test_mock_provider_respects_requested_mode():
    provider = MockFindingsProvider()
    result = provider.get_scan_result("/tmp/some-repo", mode=ScanMode.FULL)
    assert result.metadata.mode == ScanMode.FULL


def test_mock_provider_scan_result_round_trip():
    provider = MockFindingsProvider()
    result = provider.get_scan_result("/tmp/some-repo")
    restored = ScanResult.model_validate_json(result.model_dump_json())
    assert restored == result
