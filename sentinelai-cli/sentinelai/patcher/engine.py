"""Execution engine for applying, validating, and rolling back code patches."""
import ast
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from .models import Patch, PatchResult, PatchStatus


def create_backup(file_path: Path) -> Path:
    """Create a temporary backup of a file before modifying it."""
    backup_file = tempfile.NamedTemporaryFile(
        prefix=f"{file_path.name}.", suffix=".sentinelai_bak", delete=False
    )
    backup_path = Path(backup_file.name)
    backup_file.close()
    shutil.copy2(file_path, backup_path)
    return backup_path


def restore_backup(backup_path: Path, target_path: Path) -> bool:
    """Restore a file from its backup."""
    if not backup_path.exists():
        return False
    shutil.copy2(backup_path, target_path)
    return True


def validate_code_syntax(file_path: Path, content: str) -> Optional[str]:
    """Validate syntax of code before saving. Returns error message if invalid."""
    if file_path.suffix.lower() == ".py":
        try:
            ast.parse(content, filename=str(file_path))
        except SyntaxError as e:
            return f"Python syntax error at line {e.lineno}: {e.msg}"
        except Exception as e:
            return f"Syntax validation error: {str(e)}"
    return None


def apply_patch(patch: Patch, validate_syntax: bool = True) -> PatchResult:
    """Safely apply a patch to disk with syntax validation and backup."""
    if not patch.file_path.exists():
        return PatchResult(
            patch=patch,
            status=PatchStatus.FAILED,
            message=f"Target file does not exist: {patch.file_path}",
        )

    try:
        content = patch.file_path.read_text(encoding="utf-8")
    except Exception as e:
        return PatchResult(
            patch=patch,
            status=PatchStatus.FAILED,
            message=f"Could not read target file: {e}",
        )

    # Make backup
    backup_path = create_backup(patch.file_path)

    # Strategy 1: Direct substring replacement if original_snippet is unique
    new_content = None
    if patch.original_snippet and patch.original_snippet in content:
        if content.count(patch.original_snippet) == 1:
            new_content = content.replace(patch.original_snippet, patch.replacement_snippet, 1)

    # Strategy 2: Line range replacement
    if new_content is None and patch.line_start is not None:
        lines = content.splitlines(keepends=True)
        l_start = max(1, patch.line_start)
        l_end = min(len(lines), patch.line_end or l_start)

        # Build replacement lines with matching trailing newline
        repl_str = patch.replacement_snippet
        if not repl_str.endswith("\n"):
            repl_str += "\n"

        lines[l_start - 1 : l_end] = [repl_str]
        new_content = "".join(lines)

    if new_content is None:
        return PatchResult(
            patch=patch,
            status=PatchStatus.FAILED,
            message="Could not locate snippet in target file",
            backup_path=backup_path,
        )

    # Validate syntax
    if validate_syntax:
        err = validate_code_syntax(patch.file_path, new_content)
        if err:
            return PatchResult(
                patch=patch,
                status=PatchStatus.FAILED,
                message=f"Patch rejected: {err}",
                backup_path=backup_path,
            )

    # Write file
    try:
        patch.file_path.write_text(new_content, encoding="utf-8")
    except Exception as e:
        restore_backup(backup_path, patch.file_path)
        return PatchResult(
            patch=patch,
            status=PatchStatus.FAILED,
            message=f"Failed to write patched content: {e}",
            backup_path=backup_path,
        )

    return PatchResult(
        patch=patch,
        status=PatchStatus.APPLIED,
        message="Patch successfully applied to source code",
        backup_path=backup_path,
    )


def rollback_patch(result: PatchResult) -> bool:
    """Roll back an applied patch using its backup."""
    if result.backup_path and result.backup_path.exists():
        return restore_backup(result.backup_path, result.patch.file_path)
    return False
