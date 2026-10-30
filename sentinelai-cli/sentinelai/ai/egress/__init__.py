"""The verification gate every payload passes before it may leave the machine."""
from .guard import EgressViolation, assert_safe

__all__ = ["EgressViolation", "assert_safe"]
