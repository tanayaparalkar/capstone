"""
Trivy scanner wrapper - invokes the installed `trivy` executable in
filesystem/vulnerability mode against a repository and converts its JSON
output into ScannerFinding objects. Nothing else.

Every claim below was verified against a real Trivy 0.74.0 install
(`brew install trivy`) run over the project's benchmark repository, not
assumed from documentation:

- Scanner selection. Trivy's default is `--scanners vuln,secret`. Leaving
  it at the default would make Trivy a second secret scanner alongside
  GitLeaks, duplicating every secret finding under a different tool name.
  `--scanners vuln` is therefore mechanically required for this wrapper's
  purpose, not a configuration preference, and is hardcoded for the same
  reason bandit.py hardcodes `-r -f json`. Misconfiguration scanning is
  likewise never enabled.

- Exit codes. Trivy returns 0 whether or not vulnerabilities were found;
  only `--exit-code N` (which this wrapper never passes) changes that.
  Confirmed both ways: 12 vulnerabilities -> exit 0, and `--exit-code 5`
  with the same input -> exit 5. So unlike Bandit and GitLeaks, findings
  never make the exit code non-zero, and any non-zero status here means
  Trivy itself failed (e.g. a missing target path -> exit 1, "FATAL ...
  fs scan error").

- `Results` is `null`, not `[]`, when the tree contains no manifest at
  all - so `data.get("Results") or []` is required rather than indexing.
  A manifest with no known vulnerabilities instead yields a Result with
  `Packages` but no `Vulnerabilities` key at all. Both are normal, empty
  outcomes and return [] rather than raising.

- Line numbers are deliberately discarded. Trivy exposes package-level
  `Locations[].StartLine`, and using it was measured to be actively
  harmful: in the benchmark repository `CVE-2020-14343` and
  `CVE-2020-1747` both map to pyyaml at line 2 with CWE-20, and
  `CVE-2018-18074` and `CVE-2024-47081` both map to requests at line 3
  with CWE-522. Under the existing correlation rules (same normalized
  CWE + same file + lines within tolerance) each of those pairs would be
  merged into one issue, hiding a real distinct CVE behind another. A
  vulnerability in a dependency is a property of the dependency, not of
  the line that happens to declare it, so line_start/line_end are None
  and these findings stay singletons.

- Field availability across all 12 real vulnerabilities: PkgName,
  InstalledVersion, FixedVersion, VulnerabilityID, Severity, Title,
  Description, PrimaryURL, References, Status 12/12; CweIDs 11/12 (and
  it is a *list*, e.g. ["CWE-20"]); PkgPath 0/12, which is why `file`
  comes from the Result's Target rather than the package.

`file` is the manifest path normalized relative to the repository root
(Trivy already reports Target that way, e.g. "requirements.txt"; an
absolute Target is made relative when it lies under the root). Note this
differs from semgrep.py/bandit.py/gitleaks.py, which pass their tools'
absolute paths through unchanged. The difference is intentional per the
Phase 2 specification and is safe: correlation ignores these findings
entirely (they carry no line_start), so path shape cannot affect
grouping.

Trivy's own `Fingerprint` (a sha256) is not used as the finding_id. The
composite below - manifest + package + version + advisory - is already
unique, is stable across runs and across Trivy releases, and is legible
in a report table, none of which an opaque hash offers.

Explicitly out of scope: installing or updating Trivy, managing its
vulnerability database, Docker/image/repository scanning modes, secret
or misconfiguration scanning, reading environment variables, and
deduplicating findings against any other scanner.
"""
import json
import os
import subprocess
from typing import Any, Dict, List, Optional

from pydantic import ValidationError

from sentinelai.backend.context_builder import RepositoryContext
from sentinelai.contracts import ScannerFinding, Severity

from .base import Scanner
from .exceptions import ScannerExecutionError

_SEVERITY_MAP = {
    "CRITICAL": Severity.CRITICAL,
    "HIGH": Severity.HIGH,
    "MEDIUM": Severity.MEDIUM,
    "LOW": Severity.LOW,
}

_CATEGORY = "vulnerable-dependency"

# Cap on how many advisory URLs are carried into raw_evidence. Trivy's
# References list runs to dozens of entries for some CVEs; raw_evidence is
# meant to be a concise deterministic summary a reader (or the AI layer)
# can actually take in, not a transcript of the whole record. PrimaryURL is
# always kept and always first.
_MAX_REFERENCES = 3


class TrivyScanner(Scanner):
    """Runs `trivy fs --scanners vuln --format json` and converts its results into ScannerFinding objects."""

    def __init__(self, executable: str = "trivy") -> None:
        self._executable = executable

    def scan(self, context: RepositoryContext) -> List[ScannerFinding]:
        repository_root = str(context.repository.absolute_path)
        command = [self._executable, "fs", "--scanners", "vuln", "--format", "json", repository_root]

        try:
            completed = subprocess.run(command, capture_output=True, encoding="utf-8", check=False)
        except (OSError, UnicodeDecodeError) as exc:
            raise ScannerExecutionError(f"Failed to run trivy ('{self._executable}'): {exc}") from exc

        # Findings never change trivy's exit code (see module docstring), so
        # any non-zero status is trivy itself failing.
        if completed.returncode != 0:
            raise ScannerExecutionError(
                f"trivy exited with status {completed.returncode}: {completed.stderr.strip()}"
            )

        try:
            data = json.loads(completed.stdout)
            return _convert_results(data, repository_root)
        except (json.JSONDecodeError, TypeError, AttributeError, KeyError, ValidationError) as exc:
            raise ScannerExecutionError(f"Failed to parse trivy JSON output: {exc}") from exc


def _convert_results(data: Any, repository_root: str) -> List[ScannerFinding]:
    # `Results` is null for a tree with no manifests, and each Result omits
    # `Vulnerabilities` entirely when its packages are all clean.
    results = (data or {}).get("Results") or []
    findings: List[ScannerFinding] = []
    seen_ids: Dict[str, int] = {}

    for result in results:
        target = _normalize_target((result or {}).get("Target"), repository_root)
        for vulnerability in (result or {}).get("Vulnerabilities") or []:
            findings.append(_convert_vulnerability(vulnerability, target, seen_ids))
    return findings


def _convert_vulnerability(vulnerability: Dict[str, Any], target: str, seen_ids: Dict[str, int]) -> ScannerFinding:
    package = _text(vulnerability.get("PkgName")) or "unknown"
    installed = _text(vulnerability.get("InstalledVersion")) or "unknown"
    advisory = _text(vulnerability.get("VulnerabilityID")) or "unknown"

    return ScannerFinding(
        finding_id=_finding_id(target, package, installed, advisory, seen_ids),
        scanner="trivy",
        category=_CATEGORY,
        severity=_map_severity(vulnerability.get("Severity")),
        file=target,
        # A dependency vulnerability has no line - see module docstring.
        line_start=None,
        line_end=None,
        rule_id=advisory,
        message=_build_message(vulnerability, package, installed, advisory),
        raw_evidence=_build_evidence(vulnerability, target, package, installed, advisory),
        cwe=_extract_cwe(vulnerability),
    )


def _finding_id(target: str, package: str, installed: str, advisory: str, seen_ids: Dict[str, int]) -> str:
    """A deterministic id built from manifest + package + version + advisory.

    Positional ids (`trivy-0`, `trivy-1`) would renumber every finding
    whenever an unrelated advisory is published or fixed, so the same
    vulnerability would not keep the same id between two runs of the same
    repository. The composite below does, and it stays readable in a report
    table: `trivy-requirements.txt:flask@0.12.2:CVE-2018-1000656`.

    finding_id is the join key between ScannerFinding and AIEnrichedFinding
    in the statistics layer and in all five renderers, each of which builds a
    `{finding_id: ...}` dict - duplicates silently collapse entries rather
    than failing loudly (the bug semgrep.py's _finding_id docstring records).
    The composite is unique in every real output observed, but uniqueness here
    is guaranteed structurally rather than assumed: a repeated key gets a
    `#N` suffix instead of overwriting its predecessor.
    """
    base = f"trivy-{target}:{package}@{installed}:{advisory}"
    count = seen_ids.get(base, 0)
    seen_ids[base] = count + 1
    return base if count == 0 else f"{base}#{count + 1}"


def _normalize_target(target: Any, repository_root: str) -> str:
    """Trivy's Target relative to the repository root.

    Trivy already reports Target relative to the scanned path
    ("requirements.txt"); an absolute Target that lies under the root is made
    relative, and anything else is passed through unchanged rather than
    guessed at.
    """
    text = _text(target)
    if text is None:
        return "unknown"
    if not os.path.isabs(text):
        return text
    try:
        relative = os.path.relpath(text, repository_root)
    except ValueError:  # e.g. different drives on Windows
        return text
    return text if relative.startswith(os.pardir) else relative


def _map_severity(value: Any) -> Severity:
    if not isinstance(value, str):
        return Severity.LOW
    return _SEVERITY_MAP.get(value.upper(), Severity.LOW)


def _extract_cwe(vulnerability: Dict[str, Any]) -> Optional[str]:
    """The first entry of Trivy's CweIDs list, which is already in "CWE-20" form.

    Only the first is taken because ScannerFinding.cwe is a single value; the
    rest are not lost, they are carried in raw_evidence below.
    """
    cwe_ids = vulnerability.get("CweIDs")
    if isinstance(cwe_ids, list):
        for candidate in cwe_ids:
            text = _text(candidate)
            if text is not None:
                return text
    return _text(cwe_ids)


def _build_message(vulnerability: Dict[str, Any], package: str, installed: str, advisory: str) -> str:
    fixed = _text(vulnerability.get("FixedVersion"))
    headline = _text(vulnerability.get("Title")) or _text(vulnerability.get("Description")) or advisory

    remedy = f"Fixed in {fixed}." if fixed else "No fixed version is available."
    return f"{package} {installed} is affected by {advisory}: {headline} {remedy}"


def _build_evidence(
    vulnerability: Dict[str, Any], target: str, package: str, installed: str, advisory: str
) -> str:
    """A concise, deterministic package/advisory/fix summary - not the raw JSON record.

    Field order is fixed and every value comes straight from Trivy, so two
    runs over unchanged input produce byte-identical evidence.
    """
    cwe_ids = [text for text in (_text(c) for c in vulnerability.get("CweIDs") or []) if text]

    lines = [
        f"manifest: {target}",
        f"package: {package}",
        f"installed: {installed}",
        f"fixed: {_text(vulnerability.get('FixedVersion')) or 'none available'}",
        f"advisory: {advisory}",
        f"severity: {_text(vulnerability.get('Severity')) or 'UNKNOWN'}",
    ]
    if cwe_ids:
        lines.append(f"cwe: {', '.join(cwe_ids)}")
    status = _text(vulnerability.get("Status"))
    if status:
        lines.append(f"status: {status}")
    for url in _references(vulnerability):
        lines.append(f"reference: {url}")
    return "\n".join(lines)


def _references(vulnerability: Dict[str, Any]) -> List[str]:
    """PrimaryURL first, then References, de-duplicated, capped, order preserved."""
    urls: List[str] = []
    primary = _text(vulnerability.get("PrimaryURL"))
    if primary:
        urls.append(primary)
    for reference in vulnerability.get("References") or []:
        text = _text(reference)
        if text and text not in urls:
            urls.append(text)
    return urls[:_MAX_REFERENCES]


def _text(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None
