"""
Tests for the small error hierarchy in sentinelai/core/errors.py, and
that ReportLoadError correctly participates in it (so the CLI can catch
InvalidInputError broadly and still catch ReportLoadError specifically).
"""
from sentinelai.core import InvalidInputError, ProviderError, SentinelAIError
from sentinelai.reporting.loader import ReportLoadError


def test_invalid_input_error_is_a_sentinelai_error():
    assert issubclass(InvalidInputError, SentinelAIError)


def test_provider_error_is_a_sentinelai_error():
    assert issubclass(ProviderError, SentinelAIError)


def test_report_load_error_is_an_invalid_input_error():
    assert issubclass(ReportLoadError, InvalidInputError)
    assert issubclass(ReportLoadError, SentinelAIError)


def test_errors_carry_their_message():
    exc = InvalidInputError("bad severity value")
    assert str(exc) == "bad severity value"
