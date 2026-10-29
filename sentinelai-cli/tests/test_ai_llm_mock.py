"""MockLLMProvider: records every prompt it receives and returns its canned fixture response."""
import json

from sentinelai.ai.llm import LLMProvider, LLMResponse, MockLLMProvider


def test_mock_declares_local_non_retaining_provider():
    assert issubclass(MockLLMProvider, LLMProvider)
    assert MockLLMProvider.name == "mock"
    assert MockLLMProvider.is_local is True
    assert MockLLMProvider.retains_data is False


def test_calls_start_empty():
    assert MockLLMProvider().calls == []


def test_every_prompt_is_recorded_in_order():
    provider = MockLLMProvider()
    provider.complete("first prompt", max_tokens=100, temperature=0.0)
    provider.complete("second prompt", max_tokens=200, temperature=0.5)
    provider.complete("first prompt", max_tokens=100, temperature=0.0)
    assert provider.calls == ["first prompt", "second prompt", "first prompt"]


def test_instances_do_not_share_recorded_calls():
    a, b = MockLLMProvider(), MockLLMProvider()
    a.complete("only on a", max_tokens=10, temperature=0.0)
    assert a.calls == ["only on a"]
    assert b.calls == []


def test_returns_the_bundled_canned_response():
    response = MockLLMProvider().complete("anything", max_tokens=10, temperature=0.0)
    assert isinstance(response, LLMResponse)
    assert response.model == "mock-model"
    assert response.text


def test_canned_response_can_come_from_a_supplied_fixture(tmp_path):
    fixture = tmp_path / "response.json"
    fixture.write_text(json.dumps({"text": "custom", "model": "m", "prompt_tokens": 3}), encoding="utf-8")
    response = MockLLMProvider(data_path=fixture).complete("p", max_tokens=10, temperature=0.0)
    assert response.text == "custom"
    assert response.prompt_tokens == 3
    assert response.completion_tokens is None
