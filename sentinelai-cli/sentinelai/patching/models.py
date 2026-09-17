"""
Result types for deterministic patch validation.

Plain frozen dataclasses rather than Pydantic models, and deliberately not
placed in contracts/. A validation result is an internal answer to "is this
patch structurally usable", computed on demand and never serialized into a
report - nothing in the report schema carries it, and putting it in contracts/
would imply a compatibility commitment this phase has not made. backend/
code_context.py sets the same precedent for an internal, non-serialized value
object.

PatchIssueCode exists so a caller can branch on *what* was wrong without
matching English. The human-readable message travels alongside it rather than
replacing it, because the codes are for programs and the messages are for the
person reading a failure.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple


class PatchIssueCode(str, Enum):
    """Every distinct reason a patch can be rejected. Stable identifiers - callers may branch on these."""

    # Whole-patch
    EMPTY_DIFF = "empty_diff"
    NO_CHANGES = "no_changes"

    # File headers
    MISSING_OLD_FILE_HEADER = "missing_old_file_header"
    MISSING_NEW_FILE_HEADER = "missing_new_file_header"

    # Hunks
    NO_HUNKS = "no_hunks"
    MALFORMED_HUNK_HEADER = "malformed_hunk_header"
    HUNK_COUNT_MISMATCH = "hunk_count_mismatch"
    MALFORMED_LINE_PREFIX = "malformed_line_prefix"

    # Replacement block
    REPLACEMENT_SYNTAX_ERROR = "replacement_syntax_error"


@dataclass(frozen=True)
class PatchIssue:
    """One reason a patch was rejected.

    `line` is 1-based and indexes the text that was validated - the diff string
    or the replacement block - not the target source file. It is None for issues
    that belong to the patch as a whole rather than to a position in it.
    """

    code: PatchIssueCode
    message: str
    line: Optional[int] = None


@dataclass(frozen=True)
class PatchValidation:
    """Outcome of validating one patch.

    `is_valid` is derived from `issues` rather than set independently, so the two
    can never disagree - a result reporting no issues is valid by construction.

    A valid result means the patch is *structurally* well formed: the headers
    parse, the hunks are internally consistent, every line carries a legal
    prefix, and any Python replacement block is syntactically parseable. It does
    NOT mean the patch applies to any particular file, that the target file
    exists, or that the change is correct. Checking applicability requires
    reading the target file and is not this phase's work.
    """

    issues: Tuple[PatchIssue, ...] = field(default_factory=tuple)

    @property
    def is_valid(self) -> bool:
        return not self.issues

    @property
    def codes(self) -> Tuple[PatchIssueCode, ...]:
        """Issue codes in the order they were found. Convenient for assertions and logging."""
        return tuple(issue.code for issue in self.issues)

    def __bool__(self) -> bool:
        return self.is_valid
