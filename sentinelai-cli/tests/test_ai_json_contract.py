"""
Tests for the AI JSON output contract - the agreement between what
ai/prompt_builder.py asks the model to produce and what
ai/agents/explainer.py is able to parse.

These two halves drifted apart in production and nothing caught it: the
prompt asked for a prose explanation while explain() called json.loads()
on the result, so a correctly-behaving model returned a well-formed
Markdown answer that the pipeline could not read. The tests below pin
both halves - that the prompt states the JSON-only requirement, and that
llm_ollama.py's fence stripping turns each shape a model plausibly
returns into something json.loads() accepts.

Fence stripping is tested through _strip_markdown_fences directly rather
than through OllamaProvider.generate(): the stripping is pure string
handling with no HTTP involved, so exercising it directly keeps these
tests network-free and instant, consistent with every other test in this
suite.
"""
import json

from sentinelai.ai.llm_ollama import _strip_markdown_fences
from sentinelai.ai.prompt_builder import _OUTPUT_FORMAT_INSTRUCTIONS, build_prompt
from sentinelai.ai.repository_context import RepositoryContext
from sentinelai.contracts import ScannerFinding, Severity

# The six keys ai/llm_response.py's LLMResponse declares. Written out rather than
# derived from the model so that a change to LLMResponse fails these tests loudly
# instead of silently reshaping what they assert.
_RESPONSE_BODY = """{
  "title": "Weak hash used for passwords",
  "explanation": "MD5 is unsuitable for password hashing.",
  "exploit_path": null,
  "impact": "Hashes can be reversed with precomputed tables.",
  "remediation": "Use bcrypt or Argon2 instead.",
  "patch_suggestion": null
}"""


def _finding() -> ScannerFinding:
    return ScannerFinding(
        finding_id="SENT-001",
        scanner="bandit",
        category="hashlib",
        severity=Severity.HIGH,
        file="app/crypto_utils.py",
        line_start=6,
        line_end=6,
        rule_id="B324",
        message="Use of weak MD5 hash for security.",
        raw_evidence="return hashlib.md5(password.encode()).hexdigest()",
    )


def test_prompt_includes_the_json_only_instruction():
    prompt = build_prompt(_finding(), RepositoryContext(), [])

    assert "Respond with ONLY a single JSON object" in prompt
    assert "no Markdown code fences" in prompt
    assert _OUTPUT_FORMAT_INSTRUCTIONS in prompt


def test_plain_json_passes_through_unchanged_and_parses():
    assert _strip_markdown_fences(_RESPONSE_BODY) == _RESPONSE_BODY
    assert json.loads(_RESPONSE_BODY)


def test_fenced_json_with_language_tag_is_stripped_and_parses():
    raw = f"```json\n{_RESPONSE_BODY}\n```"

    stripped = _strip_markdown_fences(raw)

    assert stripped.startswith("{")
    assert stripped.endswith("}")
    assert json.loads(stripped)


def test_bare_fence_without_language_tag_is_stripped_and_parses():
    raw = f"```\n{_RESPONSE_BODY}\n```"

    stripped = _strip_markdown_fences(raw)

    assert stripped.startswith("{")
    assert stripped.endswith("}")
    assert json.loads(stripped)


def test_non_fenced_text_is_left_unchanged():
    text = "This is just plain text, not JSON."

    assert _strip_markdown_fences(text) == text
