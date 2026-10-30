"""
Mock LLM provider.

Returns a canned LLMResponse loaded from a bundled JSON fixture and
records every (system, prompt) pair on .calls for test assertions.
"""
import json
from pathlib import Path
from typing import ClassVar, Optional

from .base import LLMProvider, LLMResponse

DEFAULT_MOCK_RESPONSE_PATH = Path(__file__).parent / "mock_data" / "mock_response.json"


class CallRecord(tuple):
    """Stores (system, prompt), but equals a string matching prompt for backwards compatibility."""

    def __new__(cls, system: Optional[str], prompt: str):
        return super().__new__(cls, (system, prompt))

    @property
    def system(self) -> Optional[str]:
        return self[0]

    @property
    def prompt(self) -> str:
        return self[1]

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return self[1] == other
        if isinstance(other, tuple):
            return super().__eq__(other)
        return False


class MockLLMProvider(LLMProvider):
    """LLMProvider backed by a JSON fixture instead of a model."""

    name: ClassVar[str] = "mock"
    is_local: ClassVar[bool] = True
    retains_data: ClassVar[Optional[bool]] = False

    def __init__(self, data_path: Optional[Path] = None) -> None:
        raw = json.loads((data_path or DEFAULT_MOCK_RESPONSE_PATH).read_text(encoding="utf-8"))
        self._response = LLMResponse(**raw)
        self.calls: list[CallRecord] = []

    def complete(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.1,
    ) -> LLMResponse:
        self.calls.append(CallRecord(system, prompt))
        latency = self._response.latency_ms if (self._response.latency_ms and self._response.latency_ms > 0) else 1.0
        return self._response.model_copy(update={"latency_ms": latency})
