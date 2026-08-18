"""
OSV-Scanner wrapper - invokes the installed `osv-scanner` executable
against a repository and converts its JSON output into ScannerFinding
objects. Nothing else.

Every claim below was verified against a real OSV-Scanner 2.5.1 install
(`brew install osv-scanner`, osv-scalibr 0.5.2) run over the project's
benchmark repository:

- Command form. 2.x removed the v1 top-level invocation
  (`osv-scanner --format json --lockfile=...`); the current form is the
  `scan source` subcommand, which extracts every package source it
  recognizes. The v1 form is not accepted and is not used here.

- `--recursive` is required, not optional. Without it `scan source`
  examines only the *top level* of the target directory and does not
  descend, which is a silent trap: pointing it at a repository root whose
  manifests live in subdirectories reports "No package sources found" and
  exits 128 - indistinguishable, from the outside, from a repository that
  genuinely has no dependencies. Measured on this project: the repository
  root exits 128 without the flag and exits 1 with it, discovering both
  `sentinelai-cli/requirements.txt` and
  `sentinelai-manual-test/requirements.txt`. Pointing the same command
  directly at a directory that *does* hold the manifest at its top level
  is unaffected - 25 groups either way - so the flag fixes the nested case
  without changing the flat one.

  This does not weaken the fail-closed contract below. A tree with no
  recognized manifest at any depth still exits 128 with `--recursive`;
  the flag widens where OSV looks, it does not make an empty search
  succeed. `--allow-no-lockfiles`, which would turn that failure into a
  success, is deliberately not used.

- Exit codes. 0 means the scan succeeded and found nothing; 1 means the
  scan succeeded and found vulnerabilities - the same convention Bandit
  and GitLeaks use, and the opposite of Trivy's. Both are success. Any
  other status is a real failure and carries an actionable message on
  stderr, which is preserved verbatim in the raised error: 127 for an
  unresolvable path ("failed to resolve path") or for `--offline` with no
  cached database ("no offline version of the OSV database is
  available"), and 128 for a tree containing no package source at all
  ("No package sources found").

  Exit 128 is worth calling out because stdout is *empty* in that case,
  not `{"results": []}` - a JSON parse would raise before the status was
  ever consulted. The status check below runs first, so that path raises
  ScannerExecutionError naming the real cause instead of a decode error.
  Per the Phase 2 specification this is treated as a failure rather than
  as an empty result: a repository with no manifest fails closed under
  `--extended`, consistent with how the orchestrator treats every other
  scanner problem.

- One finding per *group*, not per vulnerability. OSV reports the same
  underlying issue once per advisory database that carries it, and then
  states the equivalence itself in `groups[].ids`. In the benchmark
  repository that is 50 raw `vulnerabilities` entries against 25
  `groups`: every group was a PYSEC/GHSA pair for one issue. Emitting per
  vulnerability would double-report all 25 under two different
  identifiers. This wrapper emits one ScannerFinding per group and
  carries the full id/alias set in the message and evidence, so nothing
  is lost.

- Representative advisory id. `sorted(ids)[0]` - deterministic, and
  measured to be the useful choice rather than an arbitrary one: it
  selects the GHSA record in 25/25 groups, and the GHSA record is the one
  that carries `database_specific.cwe_ids`. Taking OSV's own first id
  instead would have yielded a CWE in 0/25 groups, because those are
  PYSEC records and PYSEC carries no CWE. CWE lookup still falls back
  across the remaining members in sorted order rather than relying on
  that alignment holding for every ecosystem.

- Severity. There is no severity enum anywhere in the output. `groups[].
  max_severity` is a CVSS *score* string ("9.8", "4.3"), present for
  25/25 groups, and is mapped through the standard CVSS v3 qualitative
  bands. Vulnerability-level `severity` (CVSS vectors) is present for
  only 39/50 records and is not used.

- Fixed versions come from `affected[].ranges[].events[].fixed`, filtered
  to `type == "ECOSYSTEM"`. The filter matters: GIT ranges put commit
  hashes in the same field (e.g. requests' PYSEC-2018-28 yields both
  "c45d7c49ea75..." and "2.20.0"), and a commit hash is not something a
  reader can pin a requirements file to.

- Transitive packages. OSV emits a second result whose `source.type` is
  "unknown" holding dependencies that are not in the manifest at all
  (idna 2.7.0, urllib3 1.23.0 in the benchmark, resolved from the
  registry rather than from the local environment). These are real
  findings and are converted like any other; they simply share the
  manifest path their `source.path` reports.

- No line information exists anywhere in the output, and none is
  invented. `line_start`/`line_end` are None, so these findings are never
  correlatable under the existing rules and remain singleton groups.

`file` is `source.path` normalized relative to the repository root
(OSV reports it absolute). As in trivy.py this differs from the three
code scanners, which pass absolute paths through; correlation ignores
these findings entirely, so the difference cannot affect grouping.

Explicitly out of scope: installing or updating OSV-Scanner, managing its
database, offline/local-database modes, call-graph analysis, license
scanning, and deduplicating findings against any other scanner - Trivy in
particular reports the same direct-dependency advisories under their CVE
identifiers, which is a correlation-policy question, not this wrapper's.
"""
import json
import os
import subprocess
from typing import Any, Dict, List, Optional, Tuple

from pydantic import ValidationError

from sentinelai.backend.context_builder import RepositoryContext
from sentinelai.contracts import ScannerFinding, Severity

from .base import Scanner
from .exceptions import ScannerExecutionError

_CATEGORY = "vulnerable-dependency"

# Standard CVSS v3 qualitative severity bands, applied to groups[].max_severity.
_SEVERITY_BANDS: Tuple[Tuple[float, Severity], ...] = (
    (9.0, Severity.CRITICAL),
    (7.0, Severity.HIGH),
    (4.0, Severity.MEDIUM),
)

# See trivy.py's _MAX_REFERENCES - same reasoning, same concise-evidence goal.
_MAX_REFERENCES = 3

# osv-scanner ran successfully: 0 = nothing found, 1 = vulnerabilities found.
_SUCCESS_RETURN_CODES = (0, 1)


class OSVScanner(Scanner):
    """Runs `osv-scanner scan source --format json --recursive` and converts its groups into ScannerFinding objects."""

    def __init__(self, executable: str = "osv-scanner") -> None:
        self._executable = executable

    def scan(self, context: RepositoryContext) -> List[ScannerFinding]:
        repository_root = str(context.repository.absolute_path)
        # --recursive: without it OSV inspects only the top level of the target
        # and a repository whose manifests are nested fails as "No package
        # sources found". See the module docstring for the measurement.
        command = [
            self._executable,
            "scan",
            "source",
            "--format",
            "json",
            "--recursive",
            repository_root,
        ]

        try:
            completed = subprocess.run(command, capture_output=True, encoding="utf-8", check=False)
        except (OSError, UnicodeDecodeError) as exc:
            raise ScannerExecutionError(f"Failed to run osv-scanner ('{self._executable}'): {exc}") from exc

        # Checked before parsing: at exit 128 stdout is empty, so parsing first
        # would report a decode error instead of "No package sources found".
        if completed.returncode not in _SUCCESS_RETURN_CODES:
            raise ScannerExecutionError(
                f"osv-scanner exited with status {completed.returncode}: {completed.stderr.strip()}"
            )

        try:
            data = json.loads(completed.stdout)
            return _convert_results(data, repository_root)
        except (json.JSONDecodeError, TypeError, AttributeError, KeyError, ValidationError) as exc:
            raise ScannerExecutionError(f"Failed to parse osv-scanner JSON output: {exc}") from exc


def _convert_results(data: Any, repository_root: str) -> List[ScannerFinding]:
    results = (data or {}).get("results") or []
    findings: List[ScannerFinding] = []
    seen_ids: Dict[str, int] = {}

    for result in results:
        source_path = _normalize_source((((result or {}).get("source")) or {}).get("path"), repository_root)
        for entry in (result or {}).get("packages") or []:
            findings.extend(_convert_package(entry or {}, source_path, seen_ids))
    return findings


def _convert_package(entry: Dict[str, Any], source_path: str, seen_ids: Dict[str, int]) -> List[ScannerFinding]:
    package = entry.get("package") or {}
    name = _text(package.get("name")) or "unknown"
    version = _text(package.get("version")) or "unknown"
    ecosystem = _text(package.get("ecosystem")) or "unknown"

    by_id = {
        vulnerability["id"]: vulnerability
        for vulnerability in entry.get("vulnerabilities") or []
        if isinstance(vulnerability, dict) and _text(vulnerability.get("id"))
    }

    findings = []
    for group in entry.get("groups") or []:
        finding = _convert_group(group or {}, by_id, source_path, name, version, ecosystem, seen_ids)
        if finding is not None:
            findings.append(finding)
    return findings


def _convert_group(
    group: Dict[str, Any],
    by_id: Dict[str, Dict[str, Any]],
    source_path: str,
    name: str,
    version: str,
    ecosystem: str,
    seen_ids: Dict[str, int],
) -> Optional[ScannerFinding]:
    advisory_ids = sorted({text for text in (_text(i) for i in group.get("ids") or []) if text})
    if not advisory_ids:
        # A group with no ids identifies nothing and cannot be reported
        # against an advisory; skipped rather than invented.
        return None

    representative = advisory_ids[0]
    members = [by_id[advisory_id] for advisory_id in advisory_ids if advisory_id in by_id]
    aliases = sorted({text for member in members for text in (_text(a) for a in member.get("aliases") or []) if text})
    fixed_versions = _fixed_versions(members)

    return ScannerFinding(
        finding_id=_finding_id(source_path, name, version, representative, seen_ids),
        scanner="osv-scanner",
        category=_CATEGORY,
        severity=_map_severity(group.get("max_severity")),
        file=source_path,
        # OSV reports no line information and none is synthesized - see module docstring.
        line_start=None,
        line_end=None,
        rule_id=representative,
        message=_build_message(members, advisory_ids, name, version, ecosystem, fixed_versions),
        raw_evidence=_build_evidence(
            group, members, advisory_ids, aliases, source_path, name, version, ecosystem, fixed_versions
        ),
        cwe=_extract_cwe(members),
    )


def _finding_id(source_path: str, name: str, version: str, representative: str, seen_ids: Dict[str, int]) -> str:
    """Deterministic id: manifest + package + installed version + representative advisory.

    Same reasoning as trivy.py's _finding_id - stable across runs even as the
    advisory database grows, readable in a report table, and structurally
    de-duplicated rather than assumed unique, because finding_id is the join
    key every renderer and the statistics layer index on.
    """
    base = f"osv-{source_path}:{name}@{version}:{representative}"
    count = seen_ids.get(base, 0)
    seen_ids[base] = count + 1
    return base if count == 0 else f"{base}#{count + 1}"


def _normalize_source(path: Any, repository_root: str) -> str:
    """source.path relative to the repository root; OSV reports it absolute."""
    text = _text(path)
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
    """CVSS score string -> Severity, via the standard v3 qualitative bands."""
    try:
        score = float(value)
    except (TypeError, ValueError):
        return Severity.LOW
    for threshold, severity in _SEVERITY_BANDS:
        if score >= threshold:
            return severity
    return Severity.LOW


def _extract_cwe(members: List[Dict[str, Any]]) -> Optional[str]:
    """First CWE across the group's advisories, representative first.

    Members arrive in sorted-id order, so the representative is checked first
    and the result is deterministic. The fallback across the remaining members
    is what makes this robust: PYSEC records carry no CWE at all, so a lookup
    that stopped at one advisory would return None for whole ecosystems.
    """
    for member in members:
        database_specific = member.get("database_specific")
        if not isinstance(database_specific, dict):
            continue
        for candidate in database_specific.get("cwe_ids") or []:
            text = _text(candidate)
            if text is not None:
                return text
    return None


def _fixed_versions(members: List[Dict[str, Any]]) -> List[str]:
    """Fixed versions from affected[].ranges[].events[].fixed, ECOSYSTEM ranges only.

    GIT ranges use the same `fixed` key for commit hashes, which are not
    versions anyone can pin a manifest to - see module docstring.
    """
    versions = set()
    for member in members:
        for affected in member.get("affected") or []:
            if not isinstance(affected, dict):
                continue
            for entry in affected.get("ranges") or []:
                if not isinstance(entry, dict) or entry.get("type") != "ECOSYSTEM":
                    continue
                for event in entry.get("events") or []:
                    if isinstance(event, dict):
                        text = _text(event.get("fixed"))
                        if text is not None:
                            versions.add(text)
    return sorted(versions)


def _build_message(
    members: List[Dict[str, Any]],
    advisory_ids: List[str],
    name: str,
    version: str,
    ecosystem: str,
    fixed_versions: List[str],
) -> str:
    headline = None
    for member in members:
        headline = _text(member.get("summary")) or _text(member.get("details"))
        if headline:
            break
    headline = headline or advisory_ids[0]

    remedy = f"Fixed in {', '.join(fixed_versions)}." if fixed_versions else "No fixed version is available."
    return (
        f"{name} {version} ({ecosystem}) is affected by {', '.join(advisory_ids)}: {headline} {remedy}"
    )


def _build_evidence(
    group: Dict[str, Any],
    members: List[Dict[str, Any]],
    advisory_ids: List[str],
    aliases: List[str],
    source_path: str,
    name: str,
    version: str,
    ecosystem: str,
    fixed_versions: List[str],
) -> str:
    """A concise, deterministic package/advisory/group/fix summary - not the raw JSON.

    Field order is fixed and every collection is sorted, so two runs over
    unchanged input produce byte-identical evidence.
    """
    lines = [
        f"manifest: {source_path}",
        f"package: {name}",
        f"ecosystem: {ecosystem}",
        f"installed: {version}",
        f"fixed: {', '.join(fixed_versions) if fixed_versions else 'none available'}",
        f"advisory: {advisory_ids[0]}",
        f"group ids: {', '.join(advisory_ids)}",
    ]
    if aliases:
        lines.append(f"aliases: {', '.join(aliases)}")
    max_severity = _text(group.get("max_severity"))
    if max_severity:
        lines.append(f"cvss score: {max_severity}")
    for url in _references(members):
        lines.append(f"reference: {url}")
    return "\n".join(lines)


def _references(members: List[Dict[str, Any]]) -> List[str]:
    """Advisory URLs in member order, de-duplicated and capped."""
    urls: List[str] = []
    for member in members:
        for reference in member.get("references") or []:
            url = _text(reference.get("url")) if isinstance(reference, dict) else _text(reference)
            if url and url not in urls:
                urls.append(url)
    return urls[:_MAX_REFERENCES]


def _text(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None
