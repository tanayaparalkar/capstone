"""
Tests for the ai_configured=True path in sentinelai/main.py's scan()
command - specifically, that repository context is rebuilt via the
backend and adapted with
sentinelai.ai.repository_context.from_backend_context() before reaching
enrich_findings(). No existing test exercised ai_configured=True at all
(grepped for it project-wide before writing these), so this is the
first time that branch runs in the test suite.

AI settings and the LLM/retriever factories are faked at the same seam
main.py itself uses (sentinelai.main.get_settings /
create_default_retriever / create_llm_generate_fn), not patched at a
lower level: create_default_retriever() and create_llm_generate_fn()
would otherwise try to reach a real Ollama server. confidence_scorer.py
and verifier.py run for real - neither touches the network - so only the
genuinely external calls are replaced.

_get_provider() is pinned to MockFindingsProvider throughout, exactly
like test_cli.py: these tests are about AI wiring, not provider
selection, and using the real LiveFindingsProvider default would attempt
real Semgrep/Bandit/GitLeaks subprocess calls.
"""
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from sentinelai.ai.config import AISettings
from sentinelai.ai.retrieval import RetrievedChunk, Retriever
from sentinelai.main import app
from sentinelai.providers import MockFindingsProvider

runner = CliRunner()


@pytest.fixture(autouse=True)
def _use_mock_provider(monkeypatch):
    import sentinelai.main as main_module

    monkeypatch.setattr(main_module, "_get_provider", lambda: MockFindingsProvider())


class _EmptyRetriever(Retriever):
    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        return []


def _fake_generate(prompt: str) -> str:
    return json.dumps(
        {
            "title": "Fake finding title",
            "explanation": "Fake explanation.",
            "exploit_path": None,
            "impact": None,
            "remediation": "Fake remediation.",
            "patch_suggestion": None,
        }
    )


def _configure_ai(monkeypatch):
    monkeypatch.setattr(
        "sentinelai.main.get_settings",
        lambda: AISettings(llm_model="fake-model", embedding_model="fake-embed-model"),
    )
    monkeypatch.setattr("sentinelai.main.create_default_retriever", lambda: _EmptyRetriever())
    monkeypatch.setattr("sentinelai.main.create_llm_generate_fn", lambda: _fake_generate)


def _write(path: Path, content: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_ai_findings_receive_repository_context_derived_from_the_backend(monkeypatch, tmp_path):
    _write(tmp_path / "main.py")
    _write(tmp_path / "app.js")
    _configure_ai(monkeypatch)

    result = runner.invoke(app, ["scan", str(tmp_path), "--format", "json"])

    assert result.exit_code == 0
    data = json.loads(result.output)
    ai_findings = data["findings"]["ai_enriched"]
    assert len(ai_findings) > 0
    for finding in ai_findings:
        assert finding["repository_context"] is not None
        assert tmp_path.name in finding["repository_context"]
        assert "Python" in finding["repository_context"]
        assert "JavaScript" in finding["repository_context"]


def test_ai_findings_repository_context_reflects_an_empty_repository(monkeypatch, tmp_path):
    _configure_ai(monkeypatch)

    result = runner.invoke(app, ["scan", str(tmp_path), "--format", "json"])

    assert result.exit_code == 0
    data = json.loads(result.output)
    ai_findings = data["findings"]["ai_enriched"]
    assert len(ai_findings) > 0
    for finding in ai_findings:
        assert finding["repository_context"] == f"Repository '{tmp_path.name}'."


def test_ai_disabled_by_default_never_reaches_the_new_code_path(tmp_path):
    # No AI configuration patched: get_settings() reads real (unset) env vars, so
    # ai_configured is False and repository context is never rebuilt at all -
    # confirms the new code is additive, not reached on the existing default path.
    result = runner.invoke(app, ["scan", str(tmp_path), "--format", "json"])

    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["findings"]["ai_enriched"] == []
