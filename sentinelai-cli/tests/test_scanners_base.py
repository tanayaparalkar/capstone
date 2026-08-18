"""Tests for sentinelai.scanners.base - the abstract Scanner contract."""
import pytest

from sentinelai.backend.context_builder import build_repository_context
from sentinelai.backend.loader import load_repository
from sentinelai.contracts import ScannerFinding, Severity
from sentinelai.scanners.base import Scanner


class _DummyScanner(Scanner):
    def scan(self, context):
        return []


class _FindingScanner(Scanner):
    def scan(self, context):
        return [
            ScannerFinding(
                finding_id="SENT-001",
                scanner="dummy",
                category="test-category",
                severity=Severity.LOW,
                rule_id="dummy.rule",
                message="test finding",
            )
        ]


def test_scanner_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        Scanner()


def test_concrete_subclass_without_scan_cannot_be_instantiated():
    class _IncompleteScanner(Scanner):
        pass

    with pytest.raises(TypeError):
        _IncompleteScanner()


def test_concrete_subclass_can_be_instantiated_and_scanned(tmp_path):
    context = build_repository_context(load_repository(str(tmp_path)))

    scanner = _DummyScanner()

    assert scanner.scan(context) == []


def test_concrete_subclass_returns_scanner_findings(tmp_path):
    context = build_repository_context(load_repository(str(tmp_path)))

    scanner = _FindingScanner()
    findings = scanner.scan(context)

    assert len(findings) == 1
    assert isinstance(findings[0], ScannerFinding)
    assert findings[0].scanner == "dummy"
