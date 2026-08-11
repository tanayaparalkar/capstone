"""Tests for sentinelai.scanners.exceptions - the scanner-framework exception hierarchy."""
from sentinelai.core.errors import SentinelAIError
from sentinelai.scanners.exceptions import ConfigurationError, ScannerError, ScannerExecutionError


def test_scanner_error_is_an_exception():
    assert issubclass(ScannerError, Exception)


def test_configuration_error_is_a_scanner_error():
    assert issubclass(ConfigurationError, ScannerError)


def test_scanner_execution_error_is_a_scanner_error():
    assert issubclass(ScannerExecutionError, ScannerError)


def test_configuration_error_and_scanner_execution_error_are_distinct():
    assert not issubclass(ConfigurationError, ScannerExecutionError)
    assert not issubclass(ScannerExecutionError, ConfigurationError)


def test_scanner_error_is_independent_of_core_sentinelai_error():
    # Mirrors backend/loader.py's own exception hierarchy: the scanner framework's
    # errors belong to the scanner framework, not the CLI layer's error hierarchy.
    assert not issubclass(ScannerError, SentinelAIError)
