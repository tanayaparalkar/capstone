"""
Mock LLM provider.

Returns a canned LLMResponse loaded from a bundled JSON fixture and
records every prompt it receives on .calls, so other phases can assert
on exactly what would have been sent to a model. Makes no network call;
mirrors providers/mock_provider.py's MockFindingsProvider.
"""
import json
from pathlib import Path
from typing import ClassVar, Optional

from .base import LLMProvider, LLMResponse

DEFAULT_MOCK_RESPONSE_PATH = Path(__file__).parent / "mock_data" / "mock_response.json"


class MockLLMProvider(LLMProvider):
    """LLMProvider backed by a JSON fixture instead of a model."""

    name: ClassVar[str] = "mock"
    is_local: ClassVar[bool] = True
    retains_data: ClassVar[Optional[bool]] = False

    def __init__(self, data_path: Optional[Path] = None) -> None:
        raw = json.loads((data_path or DEFAULT_MOCK_RESPONSE_PATH).read_text(encoding="utf-8"))
        self._response = LLMResponse(**raw)
        self.calls: list[str] = []

    def complete(self, prompt: str, *, max_tokens: int, temperature: float) -> LLMResponse:
        self.calls.append(prompt)
        return self._response
