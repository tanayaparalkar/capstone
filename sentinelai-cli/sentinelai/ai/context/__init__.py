"""Bounded source context around a finding, for prompts."""
from .slicer import CodeSlice, CodeSliceError, slice_around

__all__ = ["CodeSlice", "CodeSliceError", "slice_around"]
