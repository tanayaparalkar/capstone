"""
Backup and repository safety: sentinelai/patching/backup.py and safety.py.

Two guarantees carry the weight here.

*A backup always precedes a write.* Tested by ordering rather than by
inspection - a backup failure is forced and the target is then read back to
prove the write never happened. An unbacked-up write is the one outcome this
layer exists to prevent, so it is tested as an outcome, not as a call sequence.

*A backup is never overwritten.* This is what makes the stored copy the state
before the FIRST modification rather than before the most recent one, which is
what a future restore needs. Applying two different patches to one file and
checking the backup still holds the pristine original is the direct test of it.

The safety checks are tested for what they do NOT do as much as what they do:
a dirty repository produces a warning object and nothing else - no print, no
log, no prompt, no abort.
"""
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch as mock_patch

import pytest
from git.exc import GitCommandNotFound

from sentinelai.contracts import StructuredPatch
from sentinelai.core.errors import SentinelAIError
from sentinelai.patching import (
    BACKUP_DIRECTORY,
    BackupError,
    BackupManager,
    BackupRecord,
    PatchApplicator,
    PatchError,
    RepositoryState,
    inspect_repository,
)

ORIGINAL = "import yaml\n\ndef load_config(raw):\n    return yaml.load(raw)\n\ndef main():\n"
PATCHED = "import yaml\n\ndef load_config(raw):\n    return yaml.safe_load(raw)\n\ndef main():\n"
DIFF = (
    "--- a/parser.py\n+++ b/parser.py\n@@ -1,6 +1,6 @@\n"
    " import yaml\n \n def load_config(raw):\n"
    "-    return yaml.load(raw)\n+    return yaml.safe_load(raw)\n \n def main():\n"
)


def _repo(tmp_path: Path, relative: str = "app/parser.py", text: str = ORIGINAL) -> Path:
    target = tmp_path / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return target


def _patch(path: Path, **overrides) -> StructuredPatch:
    defaults = dict(diff=DIFF, file=str(path))
    defaults.update(overrides)
    return StructuredPatch(**defaults)


def _git_repo(tmp_path: Path) -> Path:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.email", "t@e.st"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "Test"], check=True)
    return tmp_path


def _commit_all(tmp_path: Path) -> None:
    subprocess.run(["git", "-C", str(tmp_path), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-qm", "initial"], check=True)


# --- backup creation --------------------------------------------------------------------------------------------


def test_backup_copies_the_file_contents(tmp_path):
    target = _repo(tmp_path)
    record = BackupManager(tmp_path).backup(target)

    assert record.backup_path.read_text(encoding="utf-8") == ORIGINAL


def test_backup_preserves_relative_repository_layout(tmp_path):
    target = _repo(tmp_path, "app/main.py")
    record = BackupManager(tmp_path).backup(target)

    assert record.backup_path == tmp_path / ".sentinelai" / "backups" / "app" / "main.py"


def test_nested_backup_directories_are_created_automatically(tmp_path):
    target = _repo(tmp_path, "a/b/c/deep.py")
    record = BackupManager(tmp_path).backup(target)

    assert record.backup_path.is_file()
    assert record.backup_path.parent == tmp_path / ".sentinelai" / "backups" / "a" / "b" / "c"


def test_backup_directory_constant_matches_the_documented_layout():
    assert BACKUP_DIRECTORY == Path(".sentinelai") / "backups"


def test_backup_preserves_file_permissions(tmp_path):
    target = _repo(tmp_path)
    os.chmod(target, 0o640)
    record = BackupManager(tmp_path).backup(target)
    assert (os.stat(record.backup_path).st_mode & 0o777) == 0o640


def test_backup_leaves_no_partial_files(tmp_path):
    target = _repo(tmp_path)
    record = BackupManager(tmp_path).backup(target)
    assert [p.name for p in record.backup_path.parent.iterdir()] == ["parser.py"]


def test_backup_path_for_computes_without_creating(tmp_path):
    target = _repo(tmp_path)
    expected = BackupManager(tmp_path).backup_path_for(target)

    assert expected == tmp_path / ".sentinelai" / "backups" / "app" / "parser.py"
    assert not expected.exists(), "backup_path_for must be pure computation"


# --- never overwriting ------------------------------------------------------------------------------------------


def test_an_existing_backup_is_never_overwritten(tmp_path):
    target = _repo(tmp_path)
    manager = BackupManager(tmp_path)
    first = manager.backup(target)

    target.write_text("MODIFIED SINCE\n", encoding="utf-8")
    second = manager.backup(target)

    assert second.already_existed is True
    assert second.backup_path == first.backup_path
    assert second.backup_path.read_text(encoding="utf-8") == ORIGINAL, "must hold the FIRST state"


def test_already_existed_is_false_on_first_backup(tmp_path):
    assert BackupManager(tmp_path).backup(_repo(tmp_path)).already_existed is False


def test_two_patches_leave_the_pristine_original_recoverable(tmp_path):
    """The property a future restore depends on."""
    target = _repo(tmp_path)
    applicator = PatchApplicator(root=tmp_path)
    applicator.apply(_patch(target))

    second = DIFF.replace("-    return yaml.load(raw)", "-    return yaml.safe_load(raw)").replace(
        "+    return yaml.safe_load(raw)", "+    return yaml.safe_load(raw, Loader=None)"
    )
    applicator.apply(_patch(target, diff=second))

    backup = BackupManager(tmp_path).backup_path_for(target)
    assert backup.read_text(encoding="utf-8") == ORIGINAL


# --- metadata ---------------------------------------------------------------------------------------------------


def test_record_describes_original_and_backup(tmp_path):
    target = _repo(tmp_path)
    record = BackupManager(tmp_path).backup(target)

    assert isinstance(record, BackupRecord)
    assert record.original_path == target.resolve()
    assert record.backup_path.is_file()


def test_record_carries_a_utc_timestamp(tmp_path):
    record = BackupManager(tmp_path).backup(_repo(tmp_path))
    assert isinstance(record.created_at, datetime)
    assert record.created_at.tzinfo is timezone.utc


def test_record_is_immutable(tmp_path):
    record = BackupManager(tmp_path).backup(_repo(tmp_path))
    with pytest.raises(Exception):
        record.backup_path = Path("/elsewhere")


def test_existing_backup_reports_the_backup_time_not_the_call_time(tmp_path):
    target = _repo(tmp_path)
    manager = BackupManager(tmp_path)
    first = manager.backup(target)
    second = manager.backup(target)

    assert second.created_at == first.created_at


# --- backup failure ---------------------------------------------------------------------------------------------


def test_target_outside_the_backup_root_is_an_error(tmp_path):
    outside = _repo(tmp_path, "outside.py")
    contained = tmp_path / "repo"
    contained.mkdir()

    with pytest.raises(BackupError) as excinfo:
        BackupManager(contained).backup(outside)

    assert "outside the backup root" in str(excinfo.value)


def test_backing_up_a_missing_file_is_an_error(tmp_path):
    with pytest.raises(BackupError):
        BackupManager(tmp_path).backup(tmp_path / "absent.py")


def test_backup_error_is_a_sentinelai_error():
    """No parallel hierarchy."""
    assert issubclass(BackupError, PatchError)
    assert issubclass(BackupError, SentinelAIError)


# --- ordering: validate -> backup -> apply ----------------------------------------------------------------------


def test_a_backup_failure_prevents_the_write(tmp_path):
    """The central guarantee, tested as an outcome rather than a call sequence."""
    target = _repo(tmp_path)

    with mock_patch.object(BackupManager, "backup", side_effect=BackupError("no space")):
        with pytest.raises(BackupError):
            PatchApplicator(root=tmp_path).apply(_patch(target))

    assert target.read_text(encoding="utf-8") == ORIGINAL, "no write may happen without a backup"


def test_backup_happens_before_the_write(tmp_path):
    target = _repo(tmp_path)
    order = []

    real_backup = BackupManager.backup

    def recording_backup(self, path):
        order.append("backup")
        return real_backup(self, path)

    real_write = PatchApplicator._atomic_write

    def recording_write(self, path, contents):
        order.append("write")
        # Still performs the real write: Phase 2.3 reads the file back to verify
        # it, so a stub that only records would fail verification and mask the
        # ordering this test exists to check.
        real_write(path, contents)

    with mock_patch.object(BackupManager, "backup", recording_backup):
        with mock_patch.object(PatchApplicator, "_atomic_write", recording_write):
            PatchApplicator(root=tmp_path).apply(_patch(target))

    assert order == ["backup", "write"]


def test_validation_failure_happens_before_any_backup(tmp_path):
    target = _repo(tmp_path)
    bad = _patch(target, diff="--- a/f\n+++ b/f\n@@ -1,9 +1,9 @@\n-a\n+b\n")

    with mock_patch.object(BackupManager, "backup") as backup:
        with pytest.raises(SentinelAIError):
            PatchApplicator(root=tmp_path).apply(bad)

    backup.assert_not_called()
    assert not (tmp_path / ".sentinelai").exists()


def test_an_unrenderable_patch_leaves_no_backup(tmp_path):
    """Backup sits after render, so a patch that cannot be applied creates no clutter."""
    target = _repo(tmp_path, text="completely\ndifferent\ncontent\nhere\nnow\nok\n")

    with pytest.raises(SentinelAIError):
        PatchApplicator(root=tmp_path).apply(_patch(target))

    assert not (tmp_path / ".sentinelai").exists()


def test_result_carries_the_backup_record(tmp_path):
    target = _repo(tmp_path)
    result = PatchApplicator(root=tmp_path).apply_with_result(_patch(target))

    assert result.contents == PATCHED
    assert result.backup is not None
    assert result.backup.backup_path.read_text(encoding="utf-8") == ORIGINAL


def test_no_backup_is_taken_without_a_root(tmp_path):
    """With no root there is no repository layout to mirror."""
    target = _repo(tmp_path)
    result = PatchApplicator().apply_with_result(_patch(target))

    assert result.backup is None
    assert result.repository.state is RepositoryState.NOT_CHECKED


def test_an_injected_backup_manager_is_used(tmp_path):
    """The applicator orchestrates backup; it does not own the implementation."""
    target = _repo(tmp_path)
    elsewhere = tmp_path / "custom"
    elsewhere.mkdir()

    result = PatchApplicator(
        root=tmp_path, backup_manager=BackupManager(tmp_path, backup_directory=Path("custom"))
    ).apply_with_result(_patch(target))

    assert result.backup.backup_path == elsewhere / "app" / "parser.py"


# --- repository safety ------------------------------------------------------------------------------------------


def test_a_clean_git_repository_reports_clean(tmp_path):
    _repo(tmp_path)
    _git_repo(tmp_path)
    _commit_all(tmp_path)

    status = inspect_repository(tmp_path)

    assert status.state is RepositoryState.CLEAN
    assert status.should_warn is False
    assert status.changed_paths == ()


def test_a_dirty_git_repository_reports_dirty_with_the_paths(tmp_path):
    _repo(tmp_path)
    _git_repo(tmp_path)
    _commit_all(tmp_path)
    (tmp_path / "app" / "parser.py").write_text("changed\n", encoding="utf-8")

    status = inspect_repository(tmp_path)

    assert status.state is RepositoryState.DIRTY
    assert status.should_warn is True
    assert any("parser.py" in entry for entry in status.changed_paths)
    assert "uncommitted change" in status.detail


def test_a_dirty_repository_does_not_abort_the_patch(tmp_path):
    """A warning is information for a caller, never a veto."""
    target = _repo(tmp_path)
    _git_repo(tmp_path)

    result = PatchApplicator(root=tmp_path).apply_with_result(_patch(target))

    assert result.repository.should_warn is True
    assert target.read_text(encoding="utf-8") == PATCHED, "the patch still applied"


def test_a_non_git_directory_is_not_an_error(tmp_path):
    _repo(tmp_path)
    status = inspect_repository(tmp_path)

    assert status.state is RepositoryState.NOT_A_REPOSITORY
    assert status.should_warn is False


def test_patching_proceeds_normally_outside_a_git_repository(tmp_path):
    target = _repo(tmp_path)
    result = PatchApplicator(root=tmp_path).apply_with_result(_patch(target))

    assert result.repository.state is RepositoryState.NOT_A_REPOSITORY
    assert target.read_text(encoding="utf-8") == PATCHED


def test_missing_git_executable_is_not_an_error(tmp_path):
    with mock_patch("sentinelai.patching.safety.Repo", side_effect=GitCommandNotFound("git", "not found")):
        status = inspect_repository(tmp_path)

    assert status.state is RepositoryState.GIT_UNAVAILABLE
    assert status.should_warn is False


def test_missing_git_executable_does_not_stop_a_patch(tmp_path):
    target = _repo(tmp_path)
    with mock_patch("sentinelai.patching.safety.Repo", side_effect=GitCommandNotFound("git", "not found")):
        result = PatchApplicator(root=tmp_path).apply_with_result(_patch(target))

    assert result.repository.state is RepositoryState.GIT_UNAVAILABLE
    assert target.read_text(encoding="utf-8") == PATCHED


def test_inspection_never_raises_for_a_missing_directory(tmp_path):
    assert inspect_repository(tmp_path / "nope").state in {
        RepositoryState.NOT_A_REPOSITORY,
        RepositoryState.GIT_UNAVAILABLE,
    }


def test_safety_module_neither_prints_nor_logs():
    """This phase returns structured information and nothing else."""
    import ast

    import sentinelai.patching.safety as safety

    tree = ast.parse(open(safety.__file__).read())
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert "logging" not in imported
    calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "print" not in calls
    assert "input" not in calls


def test_backup_module_has_no_git_or_subprocess_dependency():
    """Backup is pure filesystem; the git dependency is quarantined in safety.py."""
    import ast

    import sentinelai.patching.backup as backup

    tree = ast.parse(open(backup.__file__).read())
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    for forbidden in ("git", "subprocess", "logging", "urllib", "socket"):
        assert forbidden not in imported, f"backup.py must not import {forbidden}"


def test_backup_module_contains_no_restoration_logic():
    """backup.py stays backup-only; restoration lives in rollback.py.

    This replaces a Phase 2.2 tripwire that asserted NOTHING in the package
    could restore a backup. Phase 2.3 added RollbackManager, so that assertion
    is correctly obsolete - it existed to fail loudly the moment rollback
    arrived, and it did. The separation it was really protecting is the module
    boundary, which is what this now checks instead.
    """
    import ast

    import sentinelai.patching.backup as backup

    tree = ast.parse(open(backup.__file__).read())
    functions = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert not any(
        name.lower().startswith(("restore", "rollback", "undo", "recover")) for name in functions
    )
