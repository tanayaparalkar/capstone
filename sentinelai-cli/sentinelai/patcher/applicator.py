"""Atomic patch applicator with diff and replacement chunk support."""
import ast
from pathlib import Path
from typing import Optional

from .backup import create_snapshot, restore_backup
from .models import Patch, PatchResult, PatchStatus


def validate_code_syntax(file_path: Path, content: str) -> Optional[str]:
    """Pre-flight AST validation for Python files."""
    if file_path.suffix.lower() == ".py":
        try:
            ast.parse(content, filename=str(file_path))
        except SyntaxError as e:
            return f"Python syntax error at line {e.lineno}: {e.msg}"
        except Exception as e:
            return f"Syntax error: {str(e)}"
    return None


def apply_replacement_chunk(
    content: str,
    original_snippet: Optional[str],
    replacement_snippet: str,
    line_start: Optional[int] = None,
    line_end: Optional[int] = None,
) -> Optional[str]:
    """Apply a replacement chunk via exact string matching or line ranges."""
    # 1. Direct unique substring replacement
    if original_snippet and original_snippet in content:
        if content.count(original_snippet) == 1:
            return content.replace(original_snippet, replacement_snippet, 1)

    # 2. Line range replacement
    if line_start is not None:
        lines = content.splitlines(keepends=True)
        l_start = max(1, line_start)
        l_end = min(len(lines), line_end or l_start)

        repl_str = replacement_snippet
        if not repl_str.endswith("\n"):
            repl_str += "\n"

        lines[l_start - 1 : l_end] = [repl_str]
        return "".join(lines)

    return None


def apply_patch_atomically(
    patch: Patch,
    repo_root: Optional[Path] = None,
    validate_syntax: bool = True,
) -> PatchResult:
    """Atomically apply a patch to disk with snapshot backup and pre-flight validation."""
    if not patch.file_path.exists():
        return PatchResult(
            patch=patch,
            status=PatchStatus.FAILED,
            message=f"Target file does not exist: {patch.file_path}",
        )

    root = repo_root or patch.file_path.parent

    try:
        content = patch.file_path.read_text(encoding="utf-8")
    except Exception as e:
        return PatchResult(
            patch=patch,
            status=PatchStatus.FAILED,
            message=f"Could not read target file: {e}",
        )

    # Create atomic snapshot in .sentinelai/backups/
    try:
        backup_path = create_snapshot(patch.file_path, root)
    except Exception:
        # Fallback to in-memory/temp backup
        from .engine import create_backup
        backup_path = create_backup(patch.file_path)

    # Apply replacement chunk
    new_content = apply_replacement_chunk(
        content=content,
        original_snippet=patch.original_snippet,
        replacement_snippet=patch.replacement_snippet,
        line_start=patch.line_start,
        line_end=patch.line_end,
    )

    if new_content is None:
        return PatchResult(
            patch=patch,
            status=PatchStatus.FAILED,
            message="Could not locate code snippet in target file",
            backup_path=backup_path,
        )

    # Pre-flight syntax validation before saving
    if validate_syntax:
        err = validate_code_syntax(patch.file_path, new_content)
        if err:
            return PatchResult(
                patch=patch,
                status=PatchStatus.FAILED,
                message=f"Patch rejected by pre-flight validation: {err}",
                backup_path=backup_path,
            )

    # Atomic write to file
    try:
        patch.file_path.write_text(new_content, encoding="utf-8")
    except Exception as e:
        restore_backup(backup_path, patch.file_path)
        return PatchResult(
            patch=patch,
            status=PatchStatus.FAILED,
            message=f"Failed to write patched file: {e}",
            backup_path=backup_path,
        )

    return PatchResult(
        patch=patch,
        status=PatchStatus.APPLIED,
        message="Patch atomically applied to source code",
        backup_path=backup_path,
    )
