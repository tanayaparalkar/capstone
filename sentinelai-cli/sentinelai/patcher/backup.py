"""Backup and safety layer for SentinelAI code remediation."""
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

BACKUPS_DIR_NAME = ".sentinelai/backups"
MANIFEST_FILE_NAME = "backup_manifest.json"


def check_git_clean(repo_root: Path) -> Tuple[bool, str]:
    """Check if the git repository has unstaged or uncommitted changes."""
    git_bin = shutil.which("git")
    if not git_bin:
        return True, ""

    try:
        proc = subprocess.run(
            [git_bin, "status", "--porcelain"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=5,
        )
        if proc.returncode == 0:
            lines = proc.stdout.strip().splitlines()
            if lines:
                return False, f"Git repository has {len(lines)} uncommitted change(s). Backups will be taken before patching."
    except Exception:
        pass

    return True, ""


def get_backup_dir(repo_root: Path) -> Path:
    """Get and ensure the backup directory exists."""
    backup_dir = repo_root / BACKUPS_DIR_NAME
    backup_dir.mkdir(parents=True, exist_ok=True)
    return backup_dir


def create_snapshot(file_path: Path, repo_root: Path) -> Path:
    """Create an atomic snapshot of a file in .sentinelai/backups/."""
    backup_dir = get_backup_dir(repo_root)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    rel_path = file_path.relative_to(repo_root) if file_path.is_absolute() else file_path
    safe_name = str(rel_path).replace("/", "_").replace("\\", "_")
    backup_path = backup_dir / f"{ts}_{safe_name}.bak"

    shutil.copy2(file_path, backup_path)

    # Record in manifest
    manifest_path = backup_dir / MANIFEST_FILE_NAME
    manifest: List[Dict[str, str]] = []
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest = []

    manifest.append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "original_file": str(rel_path),
        "backup_file": str(backup_path.relative_to(repo_root)),
    })
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return backup_path


def restore_backup(backup_path: Path, target_path: Path) -> bool:
    """Restore a file from its backup copy."""
    if not backup_path.exists():
        return False
    shutil.copy2(backup_path, target_path)
    return True


def restore_latest_backup_session(repo_root: Path) -> List[Tuple[str, bool]]:
    """Undo the most recent set of applied patches from the manifest."""
    backup_dir = repo_root / BACKUPS_DIR_NAME
    manifest_path = backup_dir / MANIFEST_FILE_NAME
    if not manifest_path.exists():
        return []

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return []

    if not manifest:
        return []

    results = []
    # Process from newest to oldest
    reversed_entries = list(reversed(manifest))
    restored_targets = set()

    for entry in reversed_entries:
        orig = entry.get("original_file")
        bak = entry.get("backup_file")
        if not orig or not bak or orig in restored_targets:
            continue

        target_path = repo_root / orig
        backup_path = repo_root / bak

        if backup_path.exists():
            success = restore_backup(backup_path, target_path)
            results.append((orig, success))
            restored_targets.add(orig)

    # Clear manifest after full undo
    manifest_path.write_text("[]", encoding="utf-8")
    return results
