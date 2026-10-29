"""Secret redaction applied to every code slice before it is placed in a prompt."""
from .patterns import PATTERN_NAMES, redaction_marker
from .redactor import SecretIndex, redact_text

__all__ = ["PATTERN_NAMES", "SecretIndex", "redact_text", "redaction_marker"]
