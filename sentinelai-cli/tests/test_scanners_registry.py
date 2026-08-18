"""Tests for sentinelai.scanners.registry - the explicit scanner registry."""
import pytest

from sentinelai.scanners.base import Scanner
from sentinelai.scanners.exceptions import ConfigurationError
from sentinelai.scanners.registry import ScannerRegistration, ScannerRegistry


class _DummyScanner(Scanner):
    def scan(self, context):
        return []


def test_empty_registry_has_no_scanners():
    registry = ScannerRegistry()

    assert registry.list_scanners() == ()


def test_register_and_get():
    registry = ScannerRegistry()
    scanner = _DummyScanner()

    registry.register("dummy", scanner)

    assert registry.get("dummy") is scanner


def test_duplicate_registration_raises_configuration_error():
    registry = ScannerRegistry()
    registry.register("dummy", _DummyScanner())

    with pytest.raises(ConfigurationError, match="dummy"):
        registry.register("dummy", _DummyScanner())


def test_getting_an_unregistered_name_raises_configuration_error():
    registry = ScannerRegistry()

    with pytest.raises(ConfigurationError, match="missing"):
        registry.get("missing")


def test_deterministic_ordering_by_name_regardless_of_registration_order():
    registry = ScannerRegistry()
    registry.register("zeta", _DummyScanner())
    registry.register("alpha", _DummyScanner())
    registry.register("mu", _DummyScanner())

    result = registry.list_scanners()

    assert [registration.name for registration in result] == ["alpha", "mu", "zeta"]


def test_list_scanners_returns_registration_dataclasses():
    registry = ScannerRegistry()
    scanner = _DummyScanner()
    registry.register("dummy", scanner)

    result = registry.list_scanners()

    assert result == (ScannerRegistration(name="dummy", scanner=scanner),)


def test_list_scanners_result_is_immutable():
    registry = ScannerRegistry()
    registry.register("dummy", _DummyScanner())

    result = registry.list_scanners()

    assert isinstance(result, tuple)
    with pytest.raises(Exception):
        result[0].name = "changed"


def test_separate_registries_do_not_share_state():
    first_registry = ScannerRegistry()
    second_registry = ScannerRegistry()

    first_registry.register("dummy", _DummyScanner())

    assert first_registry.list_scanners() != ()
    assert second_registry.list_scanners() == ()
