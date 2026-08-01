"""
Data loading layer.

Phase 1: reads mock findings from a bundled JSON file.
Phase 4: replace the body of `load_findings()` with a call to Nitaanth's
         /scan API endpoint, keeping the same return type (list[Finding])
         so nothing downstream (display.py, formatters.py, main.py) has
         to change.
"""
import json
from pathlib import Path

from .models import Finding

MOCK_DATA_PATH = Path(__file__).parent / "mock_findings.json"


def load_findings(repo_path: str) -> list[Finding]:
    """
    Return findings for the given repo path.

    `repo_path` is currently unused (Phase 1 always returns the same mock
    set) but is kept in the signature now so the Phase 4 swap to a real
    API call doesn't change the function's interface.
    """
    with open(MOCK_DATA_PATH, encoding="utf-8") as f:
        raw = json.load(f)
    return [Finding(**item) for item in raw]
