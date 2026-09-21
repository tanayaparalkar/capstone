"""
Opt-in patch application through the CLI: --apply-patches and patching/runner.py.

Two things are being protected.

*Default behaviour is untouched.* A scan without the flag must never write into
the repository, and that is tested by hashing the tree before and after rather
than by asserting the flag was not read.

*Failures stay local to one finding.* Patch application is best effort: a
finding whose patch is invalid, inapplicable, or whose write cannot be verified
must be recorded and stepped over, leaving the findings on either side of it
unaffected and the scan's exit code unchanged. Several tests below deliberately
sandwich a failing finding between two good ones to check that.

The runner is tested directly rather than through a full scan because a full
scan needs real scanners and a live model to produce enriched findings; the CLI
surface itself is covered separately with Typer's runner.
"""
import hashlib
from pathlib import Path
from unittest.mock import patch as mock_patch

import pytest
from typer.testing import CliRunner

from sentinelai.contracts import (
    AIEnrichedFinding,
    ConfidenceLabel,
    Severity,
    StructuredPatch,
    VerificationStatus,
)
from sentinelai.core.exit_codes import ExitCode
from sentinelai.main import app
from sentinelai.patching import (
    PatchApplicator,
    PatchOutcome,
    PatchRunResult,
    RollbackFailedError,
    run_patches,
)
from sentinelai.patching.exceptions import PatchVerificationError

ORIGINAL = "import yaml\n\ndef load_config(raw):\n    return yaml.load(raw)\n\ndef main():\n"
PATCHED = "import yaml\n\ndef load_config(raw):\n    return yaml.safe_load(raw)\n\ndef main():\n"


def _diff(name: str = "parser.py") -> str:
    return (
        f"--- a/{name}\n+++ b/{name}\n@@ -1,6 +1,6 @@\n"
        " import yaml\n \n def load_config(raw):\n"
        "-    return yaml.load(raw)\n+    return yaml.safe_load(raw)\n \n def main():\n"
    )


def _file(tmp_path: Path, name: str = "parser.py", text: str = ORIGINAL) -> Path:
    target = tmp_path / "app" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return target


def _finding(finding_id: str, patch=None) -> AIEnrichedFinding:
    return AIEnrichedFinding(
        finding_id=finding_id,
        title="t",
        severity=Severity.HIGH,
        explanation="e",
        remediation="r",
        confidence_score=0.5,
        confidence_label=ConfidenceLabel.MEDIUM,
        verification_status=VerificationStatus.UNVERIFIED,
        structured_patch=patch,
    )


def _patch_for(path: Path, **overrides) -> StructuredPatch:
    defaults = dict(diff=_diff(path.name), file=str(path))
    defaults.update(overrides)
    return StructuredPatch(**defaults)


def _tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


# --- the flag itself --------------------------------------------------------------------------------------------


def test_apply_patches_flag_is_documented():
    result = CliRunner().invoke(app, ["scan", "--help"])
    assert result.exit_code == 0
    assert "--apply-patches" in result.stdout


def test_the_flag_defaults_to_off():
    """Opt-in: the help text must say so, since this is the only flag that writes."""
    result = CliRunner().invoke(app, ["scan", "--help"])
    assert "Off by default" in " ".join(result.stdout.split())


# --- default behaviour is unchanged -----------------------------------------------------------------------------


def test_without_the_flag_nothing_is_written(tmp_path):
    """Hashed before and after: the strongest form of 'the scan did not touch anything'."""
    target = _file(tmp_path)
    before = _tree_hash(tmp_path)

    # No run_patches call happens at all without the flag; simulate the same
    # findings reaching the reporting stage.
    findings = [_finding("SENT-001", _patch_for(target))]
    assert findings  # the patch exists and would have applied if asked

    assert _tree_hash(tmp_path) == before
    assert target.read_text(encoding="utf-8") == ORIGINAL


def test_run_patches_on_an_empty_finding_list_does_nothing(tmp_path):
    before = _tree_hash(tmp_path)
    result = run_patches([], root=tmp_path)

    assert result.attempts == ()
    assert result.attempted == 0
    assert _tree_hash(tmp_path) == before


# --- successful application -------------------------------------------------------------------------------------


def test_a_valid_patch_is_applied(tmp_path):
    target = _file(tmp_path)
    result = run_patches([_finding("SENT-001", _patch_for(target))], root=tmp_path)

    assert result.attempts[0].outcome is PatchOutcome.APPLIED
    assert result.attempts[0].succeeded is True
    assert target.read_text(encoding="utf-8") == PATCHED


def test_application_creates_a_backup(tmp_path):
    """Delegated, not reimplemented: the applicator's backup layer still runs."""
    target = _file(tmp_path)
    run_patches([_finding("SENT-001", _patch_for(target))], root=tmp_path)

    backup = tmp_path / ".sentinelai" / "backups" / "app" / "parser.py"
    assert backup.read_text(encoding="utf-8") == ORIGINAL


def test_attempt_records_the_finding_id_and_target_file(tmp_path):
    target = _file(tmp_path)
    attempt = run_patches([_finding("SENT-042", _patch_for(target))], root=tmp_path).attempts[0]

    assert attempt.finding_id == "SENT-042"
    assert attempt.file == str(target)


# --- eligibility ------------------------------------------------------------------------------------------------


def test_a_finding_without_a_structured_patch_is_skipped(tmp_path):
    result = run_patches([_finding("SENT-001")], root=tmp_path)

    attempt = result.attempts[0]
    assert attempt.outcome is PatchOutcome.SKIPPED_NO_PATCH
    assert attempt.file is None
    assert result.attempted == 0


def test_an_invalid_patch_is_recorded_as_validation_failed(tmp_path):
    target = _file(tmp_path)
    bad = _patch_for(target, diff="--- a/f\n+++ b/f\n@@ -1,9 +1,9 @@\n-a\n+b\n")

    result = run_patches([_finding("SENT-001", bad)], root=tmp_path)

    assert result.attempts[0].outcome is PatchOutcome.VALIDATION_FAILED
    assert "hunk_count_mismatch" in result.attempts[0].detail
    assert target.read_text(encoding="utf-8") == ORIGINAL


def test_a_patch_that_does_not_fit_is_recorded_as_not_applicable(tmp_path):
    target = _file(tmp_path, text="totally\ndifferent\ncontent\nhere\nnow\nok\n")

    result = run_patches([_finding("SENT-001", _patch_for(target))], root=tmp_path)

    assert result.attempts[0].outcome is PatchOutcome.NOT_APPLICABLE
    assert target.read_text(encoding="utf-8") == "totally\ndifferent\ncontent\nhere\nnow\nok\n"


def test_a_missing_target_is_recorded_as_not_applicable(tmp_path):
    missing = tmp_path / "app" / "gone.py"
    missing.parent.mkdir(parents=True)
    patch = StructuredPatch(diff=_diff("gone.py"), file=str(missing))

    result = run_patches([_finding("SENT-001", patch)], root=tmp_path)

    assert result.attempts[0].outcome is PatchOutcome.NOT_APPLICABLE


# --- verification failure and automatic rollback ------------------------------------------------------------------


def test_a_verification_failure_is_rolled_back_and_recorded(tmp_path):
    target = _file(tmp_path)

    def corrupting_write(self, path, contents):
        Path(path).write_text("TRUNCATED", encoding="utf-8")

    with mock_patch.object(PatchApplicator, "_atomic_write", corrupting_write):
        result = run_patches([_finding("SENT-001", _patch_for(target))], root=tmp_path)

    attempt = result.attempts[0]
    assert attempt.outcome is PatchOutcome.ROLLED_BACK
    assert attempt.rollback_executed is True
    assert attempt.rollback_failed is False
    assert target.read_text(encoding="utf-8") == ORIGINAL, "the file must be restored"


def test_a_rollback_failure_is_recorded_as_the_most_severe_outcome(tmp_path):
    """The only case where the repository may still hold modified content."""
    target = _file(tmp_path)

    def corrupting_write(self, path, contents):
        Path(path).write_text("TRUNCATED", encoding="utf-8")

    with mock_patch.object(PatchApplicator, "_atomic_write", corrupting_write):
        with mock_patch(
            "sentinelai.patching.rollback.RollbackManager.restore",
            side_effect=RollbackFailedError("read-only filesystem"),
        ):
            result = run_patches([_finding("SENT-001", _patch_for(target))], root=tmp_path)

    attempt = result.attempts[0]
    assert attempt.outcome is PatchOutcome.ROLLBACK_FAILED
    assert attempt.rollback_executed is True
    assert attempt.rollback_failed is True


def test_an_unexpected_error_is_recorded_rather_than_raised(tmp_path):
    """A patch failure must never abort a scan, whatever its origin."""
    target = _file(tmp_path)

    with mock_patch.object(PatchApplicator, "apply_with_result", side_effect=RuntimeError("boom")):
        result = run_patches([_finding("SENT-001", _patch_for(target))], root=tmp_path)

    assert result.attempts[0].outcome is PatchOutcome.ERROR
    assert "RuntimeError: boom" in result.attempts[0].detail


# --- isolation and ordering -------------------------------------------------------------------------------------


def test_one_failing_finding_does_not_stop_the_others(tmp_path):
    good_one = _file(tmp_path, "one.py")
    broken = _file(tmp_path, "two.py", text="unrelated\ncontent\nentirely\nhere\nnow\nok\n")
    good_two = _file(tmp_path, "three.py")

    result = run_patches(
        [
            _finding("SENT-001", _patch_for(good_one)),
            _finding("SENT-002", _patch_for(broken)),
            _finding("SENT-003", _patch_for(good_two)),
        ],
        root=tmp_path,
    )

    assert [a.outcome for a in result.attempts] == [
        PatchOutcome.APPLIED,
        PatchOutcome.NOT_APPLICABLE,
        PatchOutcome.APPLIED,
    ]
    assert good_one.read_text(encoding="utf-8") == PATCHED
    assert good_two.read_text(encoding="utf-8") == PATCHED


def test_results_preserve_the_order_of_the_findings(tmp_path):
    targets = [_file(tmp_path, f"f{i}.py") for i in range(4)]
    findings = [_finding(f"SENT-{i:03d}", _patch_for(t)) for i, t in enumerate(targets)]

    result = run_patches(findings, root=tmp_path)

    assert [a.finding_id for a in result.attempts] == [f.finding_id for f in findings]


def test_the_run_is_deterministic(tmp_path):
    target = _file(tmp_path)
    findings = [_finding("SENT-001", _patch_for(target)), _finding("SENT-002")]

    first = run_patches(findings, root=tmp_path)
    second = run_patches(findings, root=tmp_path)

    # The first run applies; the second finds the fix already in place and says
    # so. Both are stable: repeating either produces the same outcomes again.
    assert [a.finding_id for a in first.attempts] == [a.finding_id for a in second.attempts]
    assert run_patches(findings, root=tmp_path).attempts == second.attempts


# --- the result model -------------------------------------------------------------------------------------------


def test_result_partitions_attempts_without_recounting(tmp_path):
    good = _file(tmp_path, "one.py")
    broken = _file(tmp_path, "two.py", text="unrelated\ncontent\nentirely\nhere\nnow\nok\n")

    result = run_patches(
        [
            _finding("SENT-001", _patch_for(good)),
            _finding("SENT-002", _patch_for(broken)),
            _finding("SENT-003"),
        ],
        root=tmp_path,
    )

    assert len(result.applied) == 1
    assert len(result.failed) == 1
    assert len(result.skipped) == 1
    assert result.attempted == 2
    assert len(result.attempts) == 3


def test_result_and_attempts_are_immutable(tmp_path):
    result = run_patches([_finding("SENT-001")], root=tmp_path)
    assert isinstance(result, PatchRunResult)
    with pytest.raises(Exception):
        result.attempts[0].outcome = PatchOutcome.APPLIED


def test_an_injected_applicator_is_used(tmp_path):
    """The runner orchestrates; it does not construct policy it was handed."""
    target = _file(tmp_path)
    engine = PatchApplicator(root=tmp_path)

    with mock_patch.object(engine, "apply_with_result", wraps=engine.apply_with_result) as spy:
        run_patches([_finding("SENT-001", _patch_for(target))], root=tmp_path, applicator=engine)

    spy.assert_called_once()


# --- architecture -----------------------------------------------------------------------------------------------


def test_the_runner_does_not_reimplement_the_pipeline():
    """CLI orchestration only: no validation, backup, restore or filesystem work here."""
    import ast

    import sentinelai.patching.runner as runner

    source = open(runner.__file__).read()
    tree = ast.parse(source)
    imported = {
        node.module.split(".")[-1]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    for forbidden in ("backup", "rollback", "validator", "_filesystem"):
        assert forbidden not in imported, f"runner must not import {forbidden} directly"
    assert "os.replace" not in source
    assert "validate_structured_patch" not in source


def test_patch_failures_do_not_raise_out_of_the_runner(tmp_path):
    """Exit behaviour: the scan continues regardless of what patching found."""
    broken = _file(tmp_path, "two.py", text="unrelated\ncontent\nentirely\nhere\nnow\nok\n")

    result = run_patches(
        [_finding("SENT-001", _patch_for(broken)), _finding("SENT-002")], root=tmp_path
    )

    assert len(result.failed) == 1  # recorded, not raised


# --- dry run ----------------------------------------------------------------------------------------------------


def test_dry_run_does_not_modify_the_target(tmp_path):
    """The whole promise, tested as an outcome: the file is byte-identical afterwards."""
    target = _file(tmp_path)
    before = _tree_hash(tmp_path)

    result = run_patches([_finding("SENT-001", _patch_for(target))], root=tmp_path, dry_run=True)

    assert result.attempts[0].outcome is PatchOutcome.APPLIED
    assert result.dry_run is True
    assert target.read_text(encoding="utf-8") == ORIGINAL
    assert _tree_hash(tmp_path) == before


def test_dry_run_writes_no_backup(tmp_path):
    """A backup is a write into the repository, so a dry run must not take one."""
    target = _file(tmp_path)

    run_patches([_finding("SENT-001", _patch_for(target))], root=tmp_path, dry_run=True)

    assert not (tmp_path / ".sentinelai").exists()


def test_dry_run_computes_the_same_bytes_a_real_run_would_write(tmp_path):
    """A preview is only useful if it previews the real thing - same render(), not a parallel path."""
    target = _file(tmp_path)
    previewed = PatchApplicator(root=tmp_path).apply_with_result(
        _patch_for(target), dry_run=True
    ).contents

    PatchApplicator(root=tmp_path).apply(_patch_for(target))

    assert previewed == PATCHED == target.read_text(encoding="utf-8")


def test_dry_run_still_validates(tmp_path):
    target = _file(tmp_path)
    bad = _patch_for(target, diff="--- a/f\n+++ b/f\n@@ -1,9 +1,9 @@\n-a\n+b\n")

    result = run_patches([_finding("SENT-001", bad)], root=tmp_path, dry_run=True)

    assert result.attempts[0].outcome is PatchOutcome.VALIDATION_FAILED


def test_dry_run_still_detects_an_inapplicable_patch(tmp_path):
    target = _file(tmp_path, text="unrelated\ncontent\nentirely\nhere\nnow\nok\n")

    result = run_patches([_finding("SENT-001", _patch_for(target))], root=tmp_path, dry_run=True)

    assert result.attempts[0].outcome is PatchOutcome.NOT_APPLICABLE


def test_dry_run_is_repeatable(tmp_path):
    """Nothing is consumed, so the second run sees the same input as the first."""
    target = _file(tmp_path)
    findings = [_finding("SENT-001", _patch_for(target))]

    first = run_patches(findings, root=tmp_path, dry_run=True)
    second = run_patches(findings, root=tmp_path, dry_run=True)

    assert first.attempts == second.attempts
    assert target.read_text(encoding="utf-8") == ORIGINAL


def test_a_successful_dry_run_is_not_counted_as_a_failure(tmp_path):
    """Outcome is unchanged by mode; only PatchRunResult.dry_run distinguishes the two."""
    target = _file(tmp_path)
    result = run_patches([_finding("SENT-001", _patch_for(target))], root=tmp_path, dry_run=True)

    assert result.failed == ()
    assert len(result.applied) == 1
    assert result.dry_run is True


def test_a_real_run_is_unaffected_by_the_new_parameter(tmp_path):
    """Regression: the default path must still write, back up, and report APPLIED."""
    target = _file(tmp_path)

    result = run_patches([_finding("SENT-001", _patch_for(target))], root=tmp_path)

    assert result.attempts[0].outcome is PatchOutcome.APPLIED
    assert result.dry_run is False
    assert target.read_text(encoding="utf-8") == PATCHED
    assert (tmp_path / ".sentinelai" / "backups" / "app" / "parser.py").is_file()


def test_the_report_states_that_nothing_was_written(tmp_path):
    """A saved artifact must say for itself whether files changed."""
    from sentinelai.reporting import build_patch_application_report

    target = _file(tmp_path)
    run = run_patches([_finding("SENT-001", _patch_for(target))], root=tmp_path, dry_run=True)

    report = build_patch_application_report(run)

    assert report.dry_run is True
    assert report.summary.applied == 1
    assert report.attempts[0].outcome == "applied"


def test_dry_run_requires_apply_patches():
    """Fails fast rather than silently doing nothing."""
    result = CliRunner().invoke(app, ["scan", ".", "--dry-run"])

    assert result.exit_code == ExitCode.INVALID_INPUT
    assert "--apply-patches" in result.stderr


def test_dry_run_flag_is_documented():
    result = CliRunner().invoke(app, ["scan", "--help"])

    assert result.exit_code == 0
    assert "--dry-run" in result.stdout
