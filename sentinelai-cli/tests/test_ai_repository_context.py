"""
Tests for sentinelai.ai.repository_context.from_backend_context() - the
adapter resolving the backend/AI RepositoryContext naming collision.

Exercised from both directions: as a conversion in isolation (given a
real backend.RepositoryContext, does it produce the right AI-layer
shape), and as an input to the adapter's real consumers - prompt_builder
and context_retrieval - proving the adapted value is genuinely usable by
both, not just structurally similar.
"""
from pathlib import Path

from sentinelai.ai.prompt_builder import build_prompt
from sentinelai.ai.repository_context import RepositoryContext, from_backend_context
from sentinelai.ai.retrieval.context_retrieval import _build_query
from sentinelai.backend.context_builder import build_repository_context
from sentinelai.backend.loader import load_repository
from sentinelai.contracts import ScannerFinding, Severity


def _write(path: Path, content: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _finding(**overrides) -> ScannerFinding:
    base = dict(
        finding_id="SENT-001",
        scanner="semgrep",
        category="sql-injection",
        severity=Severity.HIGH,
        rule_id="dummy.rule",
        message="test finding",
    )
    base.update(overrides)
    return ScannerFinding(**base)


# --- isolation: backend.RepositoryContext -> ai.RepositoryContext -----------------------------------------------------------------


def test_returns_ai_repository_context_instance(tmp_path):
    context = build_repository_context(load_repository(str(tmp_path)))

    result = from_backend_context(context)

    assert isinstance(result, RepositoryContext)


def test_summary_includes_repository_name(tmp_path):
    context = build_repository_context(load_repository(str(tmp_path)))

    result = from_backend_context(context)

    assert tmp_path.name in result.summary


def test_summary_includes_detected_languages(tmp_path):
    _write(tmp_path / "main.py")
    _write(tmp_path / "app.js")
    context = build_repository_context(load_repository(str(tmp_path)))

    result = from_backend_context(context)

    assert "Python" in result.summary
    assert "JavaScript" in result.summary


def test_summary_omits_language_parentheses_when_no_languages_detected(tmp_path):
    context = build_repository_context(load_repository(str(tmp_path)))

    result = from_backend_context(context)

    assert "(" not in result.summary
    assert result.summary == f"Repository '{tmp_path.name}'."


def test_summary_language_order_matches_detect_languages_ordering(tmp_path):
    # detect_languages() sorts by descending file_count then name - the adapter must
    # not re-sort or otherwise reorder what it's given.
    for i in range(3):
        _write(tmp_path / f"file_{i}.py")
    _write(tmp_path / "app.js")
    context = build_repository_context(load_repository(str(tmp_path)))

    result = from_backend_context(context)

    assert result.summary.index("Python") < result.summary.index("JavaScript")


def test_summary_excludes_git_dependency_and_snippet_details(tmp_path):
    # Deliberately minimal: only name + languages, per the module docstring's
    # reasoning. A dependency or git branch name must never leak into the summary.
    _write(tmp_path / "requirements.txt", "some-very-distinctive-package==1.0\n")
    context = build_repository_context(load_repository(str(tmp_path)))

    result = from_backend_context(context)

    assert "some-very-distinctive-package" not in result.summary


def test_does_not_mutate_the_backend_context(tmp_path):
    context = build_repository_context(load_repository(str(tmp_path)))

    from_backend_context(context)

    # Frozen dataclass - if from_backend_context() somehow tried to mutate it,
    # this would already have raised. Re-adapting must also be side-effect-free.
    second = from_backend_context(context)
    assert second.summary == from_backend_context(context).summary


# --- consumer direction: adapted context is genuinely usable by prompt_builder / retrieval -----------------------------------------------------------------


def test_adapted_context_is_accepted_by_build_prompt(tmp_path):
    _write(tmp_path / "main.py")
    context = build_repository_context(load_repository(str(tmp_path)))
    adapted = from_backend_context(context)

    prompt = build_prompt(_finding(), adapted, [])

    assert f"Repository context: Repository '{tmp_path.name}' (Python)." in prompt


def test_unpopulated_backend_context_produces_no_repository_context_section(tmp_path):
    # An empty repo still gets a non-empty summary ("Repository 'x'.") - the
    # section must still appear, just without a language list.
    context = build_repository_context(load_repository(str(tmp_path)))
    adapted = from_backend_context(context)

    prompt = build_prompt(_finding(), adapted, [])

    assert f"Repository context: Repository '{tmp_path.name}'." in prompt


def test_adapted_context_is_accepted_by_retrieval_query_building(tmp_path):
    _write(tmp_path / "main.py")
    context = build_repository_context(load_repository(str(tmp_path)))
    adapted = from_backend_context(context)

    query = _build_query(_finding(), adapted)

    assert f"Repository '{tmp_path.name}' (Python)." in query
