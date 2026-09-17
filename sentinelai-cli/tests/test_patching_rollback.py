"""
Deterministic rollback: sentinelai/patching/rollback.py.

The guarantee under test is narrow and absolute - after a failure the file on
disk holds exactly the bytes the backup holds, and the backup itself is
untouched. Everything here is written to check that as an observable outcome
rather than as a sequence of calls: files are read back and compared byte for
byte, permissions are stat'd, and directories are listed for leftovers.

The awkward case worth naming is that Phase 2.1's write is already atomic, so a
failed write leaves the target unmodified and rollback has nothing to undo. The
failure automatic rollback actually exists for is a VERIFICATION failure: the
write reported success but the file does not hold what was intended. Those two
paths are tested separately because only one of them can leave a modified file
behind.
"""
import ast
import os
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch as mock_patch

import pytest

from sentinelai.patching import _filesystem
from sentinelai.contracts import StructuredPatch
from sentinelai.core.errors import SentinelAIError
from sentinelai.patching import (
    BackupManager,
    PatchApplicator,
    PatchError,
    PatchVerificationError,
    RollbackError,
    RollbackFailedError,
    RollbackManager,
    RollbackNotFoundError,
    RollbackRecord,
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


# --- manual rollback API ----------------------------------------------------------------------------------------


def test_restore_returns_the_file_to_its_backed_up_bytes(tmp_path):
    target = _repo(tmp_path)
    BackupManager(tmp_path).backup(target)
    target.write_text("SOMETHING ELSE\n", encoding="utf-8")

    record = RollbackManager(tmp_path).restore(target)

    assert target.read_bytes() == ORIGINAL.encode()
    assert record.succeeded is True


def test_restore_after_a_real_patch_undoes_it(tmp_path):
    target = _repo(tmp_path)
    PatchApplicator(root=tmp_path).apply(_patch(target))
    assert target.read_text(encoding="utf-8") == PATCHED

    RollbackManager(tmp_path).restore(target)

    assert target.read_text(encoding="utf-8") == ORIGINAL


def test_restore_is_byte_exact_including_line_endings(tmp_path):
    crlf = ORIGINAL.replace("\n", "\r\n")
    target = tmp_path / "app" / "parser.py"
    target.parent.mkdir(parents=True)
    with open(target, "w", encoding="utf-8", newline="") as handle:
        handle.write(crlf)
    BackupManager(tmp_path).backup(target)
    target.write_text("clobbered\n", encoding="utf-8")

    RollbackManager(tmp_path).restore(target)

    assert target.read_bytes() == crlf.encode()


def test_restore_restores_permissions(tmp_path):
    target = _repo(tmp_path)
    os.chmod(target, 0o640)
    BackupManager(tmp_path).backup(target)
    os.chmod(target, 0o600)
    target.write_text("changed\n", encoding="utf-8")

    RollbackManager(tmp_path).restore(target)

    assert (os.stat(target).st_mode & 0o777) == 0o640


def test_can_restore_reports_availability_without_raising(tmp_path):
    target = _repo(tmp_path)
    manager = RollbackManager(tmp_path)

    assert manager.can_restore(target) is False
    BackupManager(tmp_path).backup(target)
    assert manager.can_restore(target) is True


def test_can_restore_is_false_for_a_target_outside_the_root(tmp_path):
    outside = _repo(tmp_path, "outside.py")
    contained = tmp_path / "repo"
    contained.mkdir()
    assert RollbackManager(contained).can_restore(outside) is False


def test_backup_path_for_is_exposed_and_pure(tmp_path):
    target = _repo(tmp_path)
    expected = RollbackManager(tmp_path).backup_path_for(target)
    assert expected == tmp_path / ".sentinelai" / "backups" / "app" / "parser.py"
    assert not expected.exists()


def test_an_injected_backup_manager_is_used_for_restoration(tmp_path):
    """A custom backup directory must be restored from the same place it was written to."""
    target = _repo(tmp_path)
    manager = BackupManager(tmp_path, backup_directory=Path("custom"))
    manager.backup(target)
    target.write_text("changed\n", encoding="utf-8")

    RollbackManager(tmp_path, backup_manager=manager).restore(target)

    assert target.read_text(encoding="utf-8") == ORIGINAL


# --- metadata ---------------------------------------------------------------------------------------------------


def test_record_describes_the_restoration(tmp_path):
    target = _repo(tmp_path)
    BackupManager(tmp_path).backup(target)

    record = RollbackManager(tmp_path).restore(target)

    assert isinstance(record, RollbackRecord)
    assert record.restored_path == target.resolve()
    assert record.backup_path == tmp_path / ".sentinelai" / "backups" / "app" / "parser.py"
    assert isinstance(record.restored_at, datetime)
    assert record.restored_at.tzinfo is timezone.utc


def test_record_is_immutable(tmp_path):
    target = _repo(tmp_path)
    BackupManager(tmp_path).backup(target)
    record = RollbackManager(tmp_path).restore(target)
    with pytest.raises(Exception):
        record.succeeded = False


# --- the backup is never modified -------------------------------------------------------------------------------


def test_restoring_does_not_modify_the_backup(tmp_path):
    target = _repo(tmp_path)
    backup_path = BackupManager(tmp_path).backup(target).backup_path
    before_bytes, before_mtime = backup_path.read_bytes(), backup_path.stat().st_mtime
    target.write_text("changed\n", encoding="utf-8")

    RollbackManager(tmp_path).restore(target)

    assert backup_path.read_bytes() == before_bytes
    assert backup_path.stat().st_mtime == before_mtime


def test_restoring_does_not_consume_the_backup(tmp_path):
    target = _repo(tmp_path)
    BackupManager(tmp_path).backup(target)
    RollbackManager(tmp_path).restore(target)
    assert RollbackManager(tmp_path).can_restore(target) is True


def test_repeated_rollbacks_are_idempotent(tmp_path):
    target = _repo(tmp_path)
    BackupManager(tmp_path).backup(target)
    target.write_text("changed\n", encoding="utf-8")

    first = RollbackManager(tmp_path).restore(target)
    after_first = target.read_bytes()
    second = RollbackManager(tmp_path).restore(target)

    assert target.read_bytes() == after_first == ORIGINAL.encode()
    assert first.backup_path == second.backup_path


# --- missing and unusable backups -------------------------------------------------------------------------------


def test_missing_backup_raises_not_found(tmp_path):
    target = _repo(tmp_path)
    with pytest.raises(RollbackNotFoundError) as excinfo:
        RollbackManager(tmp_path).restore(target)
    assert "no backup exists" in str(excinfo.value)


def test_a_backup_that_is_not_a_regular_file_raises_not_found(tmp_path):
    target = _repo(tmp_path)
    backup_path = RollbackManager(tmp_path).backup_path_for(target)
    backup_path.mkdir(parents=True)

    with pytest.raises(RollbackNotFoundError):
        RollbackManager(tmp_path).restore(target)


def test_an_unreadable_backup_raises_rollback_failed(tmp_path):
    """A corrupted or unreadable backup must not be reported as a successful restore."""
    target = _repo(tmp_path)
    BackupManager(tmp_path).backup(target)
    target.write_text("changed\n", encoding="utf-8")

    with mock_patch.object(Path, "read_bytes", side_effect=OSError("I/O error")):
        with pytest.raises(RollbackFailedError) as excinfo:
            RollbackManager(tmp_path).restore(target)

    assert "could not read the backup" in str(excinfo.value)


def test_a_failed_restore_write_raises_rollback_failed(tmp_path):
    target = _repo(tmp_path)
    BackupManager(tmp_path).backup(target)

    with mock_patch("sentinelai.patching._filesystem.os.replace", side_effect=OSError("read-only fs")):
        with pytest.raises(RollbackFailedError) as excinfo:
            RollbackManager(tmp_path).restore(target)

    assert "could not restore" in str(excinfo.value)


def test_a_failed_restore_leaves_no_temporary_files(tmp_path):
    target = _repo(tmp_path)
    BackupManager(tmp_path).backup(target)

    with mock_patch("sentinelai.patching._filesystem.os.replace", side_effect=OSError("boom")):
        with pytest.raises(RollbackFailedError):
            RollbackManager(tmp_path).restore(target)

    assert sorted(p.name for p in target.parent.iterdir()) == ["parser.py"]


def test_a_successful_restore_leaves_no_temporary_files(tmp_path):
    target = _repo(tmp_path)
    BackupManager(tmp_path).backup(target)
    RollbackManager(tmp_path).restore(target)
    assert sorted(p.name for p in target.parent.iterdir()) == ["parser.py"]


def test_a_failed_restore_leaves_the_backup_available_for_retry(tmp_path):
    target = _repo(tmp_path)
    BackupManager(tmp_path).backup(target)
    target.write_text("changed\n", encoding="utf-8")

    with mock_patch("sentinelai.patching._filesystem.os.replace", side_effect=OSError("boom")):
        with pytest.raises(RollbackFailedError):
            RollbackManager(tmp_path).restore(target)

    RollbackManager(tmp_path).restore(target)
    assert target.read_text(encoding="utf-8") == ORIGINAL


# --- automatic rollback -----------------------------------------------------------------------------------------


def test_verification_failure_triggers_automatic_rollback(tmp_path):
    """The case rollback exists for: the write reported success but the file is wrong."""
    target = _repo(tmp_path)

    def corrupting_write(self, path, contents):
        Path(path).write_text("TRUNCATED", encoding="utf-8")

    with mock_patch.object(PatchApplicator, "_atomic_write", corrupting_write):
        with pytest.raises(PatchVerificationError):
            PatchApplicator(root=tmp_path).apply(_patch(target))

    assert target.read_text(encoding="utf-8") == ORIGINAL, "the file must be restored"


def test_the_raised_error_carries_the_rollback_record(tmp_path):
    target = _repo(tmp_path)

    def corrupting_write(self, path, contents):
        Path(path).write_text("TRUNCATED", encoding="utf-8")

    with mock_patch.object(PatchApplicator, "_atomic_write", corrupting_write):
        with pytest.raises(PatchVerificationError) as excinfo:
            PatchApplicator(root=tmp_path).apply(_patch(target))

    assert excinfo.value.rollback is not None
    assert excinfo.value.rollback.restored_path == target.resolve()


def test_a_write_failure_also_rolls_back(tmp_path):
    """Already atomic, so there is nothing to undo - restoring anyway must still be a no-op."""
    target = _repo(tmp_path)

    with mock_patch("sentinelai.patching._filesystem.os.replace", side_effect=OSError("disk full")):
        with pytest.raises(PatchError):
            PatchApplicator(root=tmp_path).apply(_patch(target))

    assert target.read_text(encoding="utf-8") == ORIGINAL


def test_no_rollback_is_attempted_without_a_backup(tmp_path):
    """With no root there is no backup, so there is nothing to restore from."""
    target = _repo(tmp_path)

    def corrupting_write(self, path, contents):
        Path(path).write_text("TRUNCATED", encoding="utf-8")

    with mock_patch.object(PatchApplicator, "_atomic_write", corrupting_write):
        with pytest.raises(PatchVerificationError) as excinfo:
            PatchApplicator().apply(_patch(target))

    assert excinfo.value.rollback is None


def test_a_rollback_failure_propagates_rather_than_being_swallowed(tmp_path):
    """The worst outcome must be loud: the patch failed AND the original is not back."""
    target = _repo(tmp_path)

    def corrupting_write(self, path, contents):
        Path(path).write_text("TRUNCATED", encoding="utf-8")

    with mock_patch.object(PatchApplicator, "_atomic_write", corrupting_write):
        with mock_patch.object(RollbackManager, "restore", side_effect=RollbackFailedError("no")):
            with pytest.raises(RollbackFailedError):
                PatchApplicator(root=tmp_path).apply(_patch(target))


def test_a_successful_apply_performs_no_rollback(tmp_path):
    target = _repo(tmp_path)
    with mock_patch.object(RollbackManager, "restore") as restore:
        PatchApplicator(root=tmp_path).apply(_patch(target))
    restore.assert_not_called()
    assert target.read_text(encoding="utf-8") == PATCHED


def test_automatic_rollback_ordering_is_validate_backup_write_verify(tmp_path):
    target = _repo(tmp_path)
    order = []
    real_backup, real_write = BackupManager.backup, PatchApplicator._atomic_write

    def rec_backup(self, path):
        order.append("backup")
        return real_backup(self, path)

    def rec_write(self, path, contents):
        order.append("write")
        real_write(path, contents)

    def rec_verify(path, expected):
        order.append("verify")
        raise PatchVerificationError("forced")

    def rec_restore(self, path):
        order.append("rollback")
        return RollbackRecord(Path(path), Path(path), datetime.now(timezone.utc), True)

    with mock_patch.object(BackupManager, "backup", rec_backup), mock_patch.object(
        PatchApplicator, "_atomic_write", rec_write
    ), mock_patch.object(PatchApplicator, "_verify_written", staticmethod(rec_verify)), mock_patch.object(
        RollbackManager, "restore", rec_restore
    ):
        with pytest.raises(PatchVerificationError):
            PatchApplicator(root=tmp_path).apply(_patch(target))

    assert order == ["backup", "write", "verify", "rollback"]


# --- hierarchy and purity ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "error", [RollbackError, RollbackNotFoundError, RollbackFailedError, PatchVerificationError]
)
def test_every_error_is_a_sentinelai_error(error):
    assert issubclass(error, PatchError)
    assert issubclass(error, SentinelAIError)


def test_rollback_module_has_no_ai_cli_reporting_or_subprocess_dependency():
    import sentinelai.patching.rollback as rollback

    tree = ast.parse(open(rollback.__file__).read())
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
    for forbidden in ("ai", "subprocess", "logging", "typer", "rich", "urllib", "socket", "git"):
        assert forbidden not in imported, f"rollback.py must not import {forbidden}"


def test_backup_module_still_owns_no_restoration_logic():
    """backup.py remains backup-only; rollback.py owns restoration."""
    import sentinelai.patching.backup as backup

    source = open(backup.__file__).read()
    tree = ast.parse(source)
    functions = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert not any(
        name.lower().startswith(("restore", "rollback", "undo", "recover")) for name in functions
    )


# --- shared filesystem primitive (refactor guard) ---------------------------------------------------------------


def test_both_application_and_rollback_use_the_shared_primitive(tmp_path):
    """One implementation of atomic replacement, proven by intercepting it once.

    If either caller regrew its own copy of the sequence, only one of the two
    calls below would be recorded and this would fail.
    """
    target = _repo(tmp_path)
    calls = []
    real = _filesystem.atomic_replace

    def recording(target_path, data, mode, suffix):
        calls.append(suffix)
        return real(target_path, data, mode, suffix)

    with mock_patch("sentinelai.patching._filesystem.atomic_replace", recording):
        with mock_patch("sentinelai.patching.applicator.atomic_replace", recording):
            with mock_patch("sentinelai.patching.rollback.atomic_replace", recording):
                PatchApplicator(root=tmp_path).apply(_patch(target))
                RollbackManager(tmp_path).restore(target)

    assert calls == [".sentinelai-tmp", ".sentinelai-restore"], calls


def test_neither_caller_retains_its_own_replacement_sequence():
    """os.replace and tempfile must appear in exactly one module of the package."""
    import sentinelai.patching.applicator as applicator
    import sentinelai.patching.rollback as rollback

    for module in (applicator, rollback):
        tree = ast.parse(open(module.__file__).read())
        attributes = {
            f"{node.value.id}.{node.attr}"
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
        }
        assert "os.replace" not in attributes, f"{module.__name__} must delegate the rename"
        assert "tempfile.mkstemp" not in attributes, f"{module.__name__} must delegate temp creation"


def test_backup_is_deliberately_not_on_the_shared_primitive():
    """backup.py ends with os.link, not os.replace - a different guarantee, not a duplicate."""
    import sentinelai.patching.backup as backup

    source = open(backup.__file__).read()
    assert "os.link(" in source
    assert "atomic_replace" not in source
