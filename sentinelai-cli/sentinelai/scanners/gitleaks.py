"""
Gitleaks scanner wrapper - invokes the installed `gitleaks` executable
against a repository and converts its JSON output into ScannerFinding
objects. Nothing else.

Every claim below was verified empirically against a real gitleaks 8.30.1
install (`brew install gitleaks`), not assumed from memory - the CLI and
JSON schema differ from both Semgrep's and Bandit's in ways that would
have produced real bugs if guessed:

- Subcommand: gitleaks has no bare top-level scan; `dir` scans a
  directory's current file contents, `git` scans commit history. `dir`
  is used, matching the scope semgrep.py/bandit.py already have (the
  working tree as it exists now, not git log).
- JSON shape: gitleaks' report is a bare top-level JSON array of finding
  objects (`[{...}, {...}]`), not an object with a "results" key the way
  Semgrep's and Bandit's are. Parsing assumes a list directly; anything
  else (e.g. `null`, `{}`) is treated as no findings rather than an
  error, mirroring the same "missing results" leniency the other two
  wrappers apply to their own wrapped shape.
- Exit codes are genuinely ambiguous, unlike Bandit's clean {0, 1} split.
  Verified directly: exit 0 = no leaks (valid JSON `[]`); exit 1 = leaks
  found (valid JSON with results) - matching gitleaks' own documented
  `--exit-code` default. But exit 1 is *also* returned for at least one
  real failure mode (an unreadable/missing target path), where stdout is
  empty rather than JSON. So exit code alone cannot disambiguate success
  from failure the way it can for Bandit. The two-tier check below
  handles this: any exit code outside {0, 1} is an unambiguous failure;
  for 0 or 1, whether stdout actually parses as JSON is what decides
  success vs. failure - which is also exactly why "invalid JSON raises
  ScannerExecutionError" already covers the ambiguous case correctly.
- No severity field exists anywhere in gitleaks' output - confirmed by
  inspecting real results, not just the docs. Per the explicit
  instruction to pick one deterministic value rather than invent
  per-rule heuristics, every finding is assigned Severity.HIGH: gitleaks
  only reports exposed secrets/credentials, a category the security
  industry treats as uniformly high-severity regardless of which pattern
  matched, so a fixed value is not arbitrary here the way it would be
  for a general-purpose linter.
- No CWE field exists either, but this wrapper does not substitute a
  guessed one the way it does for severity: there was no equivalent
  instruction to invent a deterministic default for CWE, so `cwe` is
  simply None for every finding, consistent with "convert only fields
  that actually exist."
- `--redact` is passed unconditionally, not made configurable. Verified
  empirically that it replaces both `Match` and `Secret` with the literal
  string "REDACTED" in gitleaks' own JSON output before this wrapper ever
  sees it. This is a security property, not a formatting preference: a
  secrets scanner that itself stores and forwards live credential values
  in its findings would be creating the exact exposure it's meant to
  detect. There is no current consumer needing the raw value, and
  validating whether a secret is genuinely live is explicitly out of
  scope for this wrapper regardless.

finding_id uses gitleaks' own `Fingerprint` field (a stable
"{file}:{ruleID}:{startLine}" identifier gitleaks computes itself) when
present, falling back to a positional "gitleaks-{index}" id - the same
pattern semgrep.py uses for its own native fingerprint field.

Explicitly out of scope, deferred to a later module: installing
Gitleaks, Docker, reading environment variables, deduplicating or
merging findings across scanners, invoking any other scanner, and
validating whether a detected secret is real/active.
"""
import json
import subprocess
from typing import Any, Dict, List, Optional, Tuple

from pydantic import ValidationError

from sentinelai.backend.context_builder import RepositoryContext
from sentinelai.contracts import ScannerFinding, Severity

from .base import Scanner
from .exceptions import ScannerExecutionError


class GitleaksScanner(Scanner):
    """Runs `gitleaks dir -f json --redact` against a repository and converts its results into ScannerFinding objects."""

    def __init__(self, executable: str = "gitleaks") -> None:
        self._executable = executable

    def scan(self, context: RepositoryContext) -> List[ScannerFinding]:
        command = [
            self._executable,
            "dir",
            str(context.repository.absolute_path),
            "-f",
            "json",
            "-r",
            "-",
            "--redact",
        ]

        try:
            completed = subprocess.run(command, capture_output=True, encoding="utf-8", check=False)
        except (OSError, UnicodeDecodeError) as exc:
            raise ScannerExecutionError(f"Failed to run gitleaks ('{self._executable}'): {exc}") from exc

        # Exit code outside {0, 1} is an unambiguous gitleaks failure. 0/1 alone do not
        # guarantee success (see module docstring) - the JSON parse below is what
        # actually disambiguates a real failure that happens to also exit 1.
        if completed.returncode not in (0, 1):
            raise ScannerExecutionError(
                f"gitleaks exited with status {completed.returncode}: {completed.stderr.strip()}"
            )

        try:
            data = json.loads(completed.stdout)
            results = data if isinstance(data, list) else []
            return [_convert_result(result, index) for index, result in enumerate(results)]
        except (json.JSONDecodeError, TypeError, AttributeError, KeyError, ValidationError) as exc:
            raise ScannerExecutionError(f"Failed to parse gitleaks JSON output: {exc}") from exc


def _convert_result(result: Dict[str, Any], index: int) -> ScannerFinding:
    line_start, line_end = _extract_line_range(result)

    return ScannerFinding(
        finding_id=result.get("Fingerprint") or f"gitleaks-{index}",
        scanner="gitleaks",
        category=result.get("RuleID", "unknown"),
        severity=Severity.HIGH,
        file=result.get("File"),
        line_start=line_start,
        line_end=line_end,
        rule_id=result.get("RuleID", "unknown"),
        message=result.get("Description", ""),
        raw_evidence=result.get("Match"),
        cwe=None,
    )


def _extract_line_range(result: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
    start_line = _positive_int(result.get("StartLine"))
    end_line = _positive_int(result.get("EndLine"))

    if start_line is None:
        return None, None
    if end_line is None or end_line < start_line:
        end_line = start_line
    return start_line, end_line


def _positive_int(value: Any) -> Optional[int]:
    return value if isinstance(value, int) and value >= 1 else None
