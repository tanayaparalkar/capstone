"""
Source context inside the shared evidence block: ai/prompt_builder.py.

The section exists so the Remediation analyst can see the code a fix has to
fit into - the enclosing function, the imports, the indentation - rather than
only the single matched line the scanner reported.

The property worth guarding most carefully is not the formatting but *who
sees it*: build_evidence_block is the one place all three agents get their
evidence from, and the critic stage is only meaningful if every agent judged
identical material. A change that gave source context to the remediation
analyst alone would break that silently, so it is asserted directly.
"""
from unittest.mock import patch

from sentinelai.ai.agents.critic import build_critic_prompt
from sentinelai.ai.agents.evidence_analyst import build_evidence_prompt
from sentinelai.ai.agents.exploit_remediation_analyst import build_exploit_remediation_prompt
from sentinelai.ai.agents.schemas import EvidenceAssessment, ExploitRemediationAssessment
from sentinelai.ai.config import AISettings
from sentinelai.ai.prompt_builder import build_evidence_block
from sentinelai.ai.repository_context import RepositoryContext
from sentinelai.contracts import ScannerFinding, Severity

from agent_fakes import VALID_EVIDENCE, VALID_EXPLOIT_REMEDIATION

_SOURCE = (
    "import hashlib\n"
    "\n"
    "\n"
    "def hash_password(password):\n"
    "    return hashlib.md5(password.encode()).hexdigest()\n"
)
_HEADER = "Source context"


def _finding(tmp_path, line_start=5, line_end=None, name="crypto.py"):
    path = tmp_path / name
    path.write_text(_SOURCE, encoding="utf-8")
    return ScannerFinding(
        finding_id="SENT-001",
        scanner="bandit",
        category="weak-cryptography",
        severity=Severity.HIGH,
        file=str(path),
        line_start=line_start,
        line_end=line_end,
        rule_id="B324",
        message="Use of weak MD5 hash for security.",
        raw_evidence="hashlib.md5(password.encode())",
        cwe="CWE-327",
    )


def _block(finding, context_lines=20):
    """Render the evidence block with a pinned context width, bypassing the cached singleton."""
    with patch(
        "sentinelai.ai.prompt_builder.get_settings",
        return_value=AISettings(code_context_lines=context_lines),
    ):
        return build_evidence_block(finding, RepositoryContext(summary="Repository 'x'."), [])


# --- the section itself -----------------------------------------------------------------------------------------------


def test_source_context_is_included_when_the_file_is_readable(tmp_path):
    block = _block(_finding(tmp_path))

    assert _HEADER in block
    assert "def hash_password(password):" in block, "the enclosing function must be visible"
    assert "import hashlib" in block, "imports must be visible - a fix may need to change them"


def test_the_finding_line_is_marked(tmp_path):
    """A 20-line window buries the reported line unless it is marked."""
    block = _block(_finding(tmp_path, line_start=5))

    assert "> 5 |     return hashlib.md5(password.encode()).hexdigest()" in block
    assert "  1 | import hashlib" in block, "context lines must NOT carry the marker"


def test_a_multi_line_span_marks_every_line_in_it(tmp_path):
    block = _block(_finding(tmp_path, line_start=4, line_end=5))

    assert "> 4 | def hash_password(password):" in block
    assert "> 5 |     return hashlib.md5(password.encode()).hexdigest()" in block


def test_line_numbers_are_absolute_file_positions_not_window_offsets(tmp_path):
    """A diff generated later is anchored to these numbers; window-relative ones would be wrong."""
    block = _block(_finding(tmp_path, line_start=5), context_lines=1)

    assert "  4 | def hash_password(password):" in block
    assert "> 5 |" in block
    assert "  1 |" not in block, "context_lines=1 must not reach line 1"


def test_context_width_is_configurable(tmp_path):
    narrow = _block(_finding(tmp_path, line_start=5), context_lines=0)
    wide = _block(_finding(tmp_path, line_start=5), context_lines=20)

    assert "import hashlib" not in narrow
    assert "import hashlib" in wide


# --- absence is silent ------------------------------------------------------------------------------------------------


def test_no_section_when_the_file_cannot_be_read(tmp_path):
    """The prompt must fall back to exactly what it contained before this section existed."""
    finding = _finding(tmp_path)
    missing = finding.model_copy(update={"file": str(tmp_path / "gone.py")})

    block = _block(missing)

    assert _HEADER not in block
    assert "Evidence:" in block, "the rest of the evidence block is unaffected"


def test_no_section_for_a_line_less_dependency_finding(tmp_path):
    """Trivy/OSV findings are manifest-scoped and carry no line - not an error."""
    finding = _finding(tmp_path).model_copy(update={"line_start": None, "line_end": None})

    assert _HEADER not in _block(finding)


# --- the invariant that matters ---------------------------------------------------------------------------------------


def test_all_three_agents_receive_the_same_source_context(tmp_path):
    """build_evidence_block is the single seam; none of the three agents may diverge."""
    finding = _finding(tmp_path)
    ctx = RepositoryContext(summary="Repository 'x'.")
    evidence = EvidenceAssessment.model_validate(VALID_EVIDENCE)
    assessment = ExploitRemediationAssessment.model_validate(VALID_EXPLOIT_REMEDIATION)

    with patch(
        "sentinelai.ai.prompt_builder.get_settings",
        return_value=AISettings(code_context_lines=20),
    ):
        prompts = [
            build_evidence_prompt(finding, ctx, []),
            build_exploit_remediation_prompt(finding, evidence, ctx, []),
            build_critic_prompt(finding, evidence, assessment, ctx, []),
        ]

    for prompt in prompts:
        assert _HEADER in prompt
        assert "> 5 |     return hashlib.md5(password.encode()).hexdigest()" in prompt
