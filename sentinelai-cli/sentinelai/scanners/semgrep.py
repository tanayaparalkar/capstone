"""
Semgrep scanner wrapper - invokes the installed `semgrep` executable
against a repository and converts its JSON output into ScannerFinding
objects. Nothing else.

Explicitly out of scope, all deferred to a later module: installing
Semgrep, downloading or authoring rulesets, Docker, reading environment
variables, deduplicating or merging findings across scanners, and
invoking any other scanner. This wrapper only shells out, parses, and
converts.

Rule selection ("--config") is accepted as an optional constructor
parameter rather than hardcoded: this module has no opinion on which
rules Semgrep should run (that is a configuration/orchestration decision
for a later module), so if no config is given, no --config flag is
passed at all and Semgrep's own installed default behavior applies
unmodified - the wrapper does not decide, download, or manage a ruleset
itself.

Field mapping is deliberately defensive: every field is read with .get()
and a fallback rather than direct indexing, because Semgrep's JSON schema
has fields that are genuinely optional or vary by rule (e.g. "cwe" is
sometimes a list, sometimes a single string, sometimes absent). A finding
missing an optional field should still produce a valid ScannerFinding,
not a crash.

Severity mapping is fixed and total: Semgrep's `extra.severity` uses the
triad ERROR / WARNING / INFO (case varies by version). Mapped as
ERROR -> HIGH, WARNING -> MEDIUM, INFO -> LOW; anything missing or
unrecognized maps to LOW rather than being assumed dangerous - a finding
Semgrep didn't rate should not be inflated into a high-severity one by
this wrapper's own guesswork. There is no Semgrep-native equivalent of
CRITICAL, so that value is simply never produced by this scanner.

finding_id uses Semgrep's own `extra.fingerprint` when present (a stable,
content-based identifier Semgrep computes for each finding) and falls
back to a positional "semgrep-{index}" id when it is absent - both are
deterministic for a given JSON input, which is all this wrapper needs;
any cross-scanner ID scheme is an aggregation concern, not this module's.
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
    "ERROR": Severity.HIGH,
    "WARNING": Severity.MEDIUM,
    "INFO": Severity.LOW,
}


class SemgrepScanner(Scanner):
    """Runs `semgrep --json` against a repository and converts its results into ScannerFinding objects."""

    def __init__(self, executable: str = "semgrep", config: Optional[str] = None) -> None:
        self._executable = executable
        self._config = config

    def scan(self, context: RepositoryContext) -> List[ScannerFinding]:
        command = [self._executable]
        if self._config is not None:
            command += ["--config", self._config]
        command += ["--json", "--quiet", str(context.repository.absolute_path)]

        try:
            completed = subprocess.run(command, capture_output=True, encoding="utf-8", check=False)
        except (OSError, UnicodeDecodeError) as exc:
            raise ScannerExecutionError(f"Failed to run semgrep ('{self._executable}'): {exc}") from exc

        if completed.returncode != 0:
            raise ScannerExecutionError(
                f"semgrep exited with status {completed.returncode}: {completed.stderr.strip()}"
            )

        try:
            data = json.loads(completed.stdout)
            results = data.get("results") or []
            return [_convert_result(result, index) for index, result in enumerate(results)]
        except (json.JSONDecodeError, TypeError, AttributeError, KeyError, ValidationError) as exc:
            raise ScannerExecutionError(f"Failed to parse semgrep JSON output: {exc}") from exc


def _convert_result(result: Dict[str, Any], index: int) -> ScannerFinding:
    extra = result.get("extra") or {}
    metadata = extra.get("metadata") or {}
    line_start, line_end = _extract_line_range(result)

    return ScannerFinding(
        finding_id=extra.get("fingerprint") or f"semgrep-{index}",
        scanner="semgrep",
        category=metadata.get("category") or result.get("check_id", "unknown"),
        severity=_map_severity(extra.get("severity")),
        file=result.get("path"),
        line_start=line_start,
        line_end=line_end,
        rule_id=result.get("check_id", "unknown"),
        message=extra.get("message", ""),
        raw_evidence=extra.get("lines"),
        cwe=_extract_cwe(metadata),
    )


def _extract_line_range(result: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
    start_line = _positive_int((result.get("start") or {}).get("line"))
    end_line = _positive_int((result.get("end") or {}).get("line"))

    if start_line is None:
        return None, None
    if end_line is None or end_line < start_line:
        end_line = start_line
    return start_line, end_line


def _positive_int(value: Any) -> Optional[int]:
    return value if isinstance(value, int) and value >= 1 else None


def _map_severity(value: Any) -> Severity:
    if not isinstance(value, str):
        return Severity.LOW
    return _SEVERITY_MAP.get(value.upper(), Severity.LOW)


def _extract_cwe(metadata: Dict[str, Any]) -> Optional[str]:
    cwe = metadata.get("cwe")
    if isinstance(cwe, list):
        return cwe[0] if cwe and isinstance(cwe[0], str) else None
    if isinstance(cwe, str):
        return cwe
    return None
