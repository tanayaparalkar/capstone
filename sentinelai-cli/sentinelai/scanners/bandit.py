"""
Bandit scanner wrapper - invokes the installed `bandit` executable
against a repository and converts its JSON output into ScannerFinding
objects. Nothing else.

Mirrors semgrep.py's shape (thin subprocess wrapper, defensive .get()
field extraction, same four-case exception mapping) with two deliberate
deviations, both forced by real differences between the two tools rather
than stylistic preference:

- Exit code handling. Semgrep returns 0 regardless of findings unless
  explicitly told to fail on results; Bandit's own documented convention
  is the opposite - exit 0 means no issues found, exit 1 means issues
  were found, and only 1 truly signals success-with-findings, not
  failure. Treating exit 1 as an error (as semgrep.py correctly does for
  itself) would raise ScannerExecutionError on every Bandit scan that
  found anything, which is the normal, successful case this wrapper
  exists to handle. Only a return code outside {0, 1} is treated as
  Bandit itself failing.
- No optional "config" constructor parameter. Semgrep needs external rule
  selection to do anything meaningful, which is a real decision this
  module declines to make on a future consumer's behalf. Bandit ships
  its own built-in check set and runs meaningfully with zero
  configuration, so there is no equivalent decision to defer - adding an
  unused parameter here would be speculative, not consistent design.

Rule "-r" (recursive) and "-f json" are hardcoded, not optional: unlike
Semgrep's rule selection, there is no legitimate alternative value for
either - a directory target requires -r to be scanned at all, and JSON
output is this wrapper's only supported format. Hardcoding a mechanically
required flag is not "managing configuration."

Bandit's `issue_confidence` has no home in ScannerFinding (there is no
confidence field on the shared contract, and this wrapper does not add
one - see module list below). It is preserved by appending it to the
existing `message` field instead of being dropped, since the contract
already supports free-text there; if it did not, it would simply be
ignored rather than inventing a field for it.

Explicitly out of scope, deferred to a later module: installing Bandit,
managing virtual environments, Docker, reading environment variables,
deduplicating or merging findings across scanners, and invoking any
other scanner.
"""
import json
import subprocess
from typing import Any, Dict, List, Optional, Tuple

from pydantic import ValidationError

from sentinelai.backend.context_builder import RepositoryContext
from sentinelai.contracts import ScannerFinding, Severity

from .base import Scanner
from .exceptions import ScannerExecutionError

_SEVERITY_MAP = {
    "LOW": Severity.LOW,
    "MEDIUM": Severity.MEDIUM,
    "HIGH": Severity.HIGH,
}


class BanditScanner(Scanner):
    """Runs `bandit -r -f json` against a repository and converts its results into ScannerFinding objects."""

    def __init__(self, executable: str = "bandit") -> None:
        self._executable = executable

    def scan(self, context: RepositoryContext) -> List[ScannerFinding]:
        command = [self._executable, "-r", "-f", "json", str(context.repository.absolute_path)]

        try:
            completed = subprocess.run(command, capture_output=True, encoding="utf-8", check=False)
        except (OSError, UnicodeDecodeError) as exc:
            raise ScannerExecutionError(f"Failed to run bandit ('{self._executable}'): {exc}") from exc

        # Bandit's own convention: exit 0 = no issues, exit 1 = issues found (both mean
        # bandit ran successfully) - see module docstring. Only anything else means
        # bandit itself failed.
        if completed.returncode not in (0, 1):
            raise ScannerExecutionError(
                f"bandit exited with status {completed.returncode}: {completed.stderr.strip()}"
            )

        stdout = completed.stdout
        json_start = stdout.find("{")
        if json_start == -1:
            raise ScannerExecutionError(f"Bandit produced no JSON output:\n{stdout}")

        try:
            data = json.loads(stdout[json_start:])
            results = data.get("results") or []
            return [_convert_result(result, index) for index, result in enumerate(results)]
        except (json.JSONDecodeError, TypeError, AttributeError, KeyError, ValidationError) as exc:
            raise ScannerExecutionError(f"Failed to parse bandit JSON output: {exc}") from exc


def _convert_result(result: Dict[str, Any], index: int) -> ScannerFinding:
    line_start, line_end = _extract_line_range(result)

    return ScannerFinding(
        finding_id=f"bandit-{index}",
        scanner="bandit",
        category=result.get("test_name") or result.get("test_id", "unknown"),
        severity=_map_severity(result.get("issue_severity")),
        file=result.get("filename"),
        line_start=line_start,
        line_end=line_end,
        rule_id=result.get("test_id", "unknown"),
        message=_build_message(result),
        raw_evidence=result.get("code"),
        cwe=_extract_cwe(result),
    )


def _extract_line_range(result: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
    line_range = result.get("line_range")
    if isinstance(line_range, list) and line_range and all(isinstance(n, int) and n >= 1 for n in line_range):
        return min(line_range), max(line_range)

    line_number = _positive_int(result.get("line_number"))
    if line_number is None:
        return None, None
    return line_number, line_number


def _positive_int(value: Any) -> Optional[int]:
    return value if isinstance(value, int) and value >= 1 else None


def _map_severity(value: Any) -> Severity:
    if not isinstance(value, str):
        return Severity.LOW
    return _SEVERITY_MAP.get(value.upper(), Severity.LOW)


def _build_message(result: Dict[str, Any]) -> str:
    message = result.get("issue_text", "")
    confidence = result.get("issue_confidence")
    if isinstance(confidence, str) and confidence:
        return f"{message} (confidence: {confidence})"
    return message


def _extract_cwe(result: Dict[str, Any]) -> Optional[str]:
    issue_cwe = result.get("issue_cwe")
    if isinstance(issue_cwe, dict):
        cwe_id = issue_cwe.get("id")
        return f"CWE-{cwe_id}" if isinstance(cwe_id, int) else None
    if isinstance(issue_cwe, int):
        return f"CWE-{issue_cwe}"
    return None
