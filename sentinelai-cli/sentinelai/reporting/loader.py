"""
Loading a previously-saved SentinelAI JSON report back into a ScanResult
+ ScanStatistics, so `sentinelai report` can render JSON/Markdown/HTML/
SARIF from an existing scan without rerunning scanners or AI reasoning.

Round-trip architecture: the existing JSON report schema (JSONReport)
already carries everything needed to reconstruct a ScanResult -
`repository` (RepositoryInfo), `scan` (ScanMetadata), and
`findings.scanner`/`findings.ai_enriched` (list[ScannerFinding] /
list[AIEnrichedFinding]) map field-for-field onto ScanResult's own
fields. No second persistence format was introduced.

Statistics strategy: the input file's `statistics` block is never
trusted - calculate_statistics() is always re-run on the reconstructed
ScanResult, so a hand-edited or corrupted `statistics` section in the
input can never produce an inconsistent report. Because
calculate_statistics() is a pure, deterministic function of ScanResult,
this reproduces byte-identical statistics to the original scan whenever
the findings/metadata themselves are untampered - "recompute" and "same
as the original" are the same outcome here, not a tradeoff.

All failure modes (missing file, malformed JSON, schema mismatch,
invalid enum values, unsupported schema_version) raise ReportLoadError -
a plain exception with a clean, actionable message - so the CLI layer
can catch exactly one exception type and print a one-line error instead
of a stack trace. Nothing in the loaded content is ever executed or
interpreted as code; it is only parsed as data and validated by Pydantic.

ReportLoadError subclasses core.errors.InvalidInputError: every failure
mode here is "the given input can't be used as given," the same
category as a bad CLI flag - so the CLI maps it to ExitCode.INVALID_INPUT
alongside argument-validation errors, not a distinct exit code.
"""
import json
from pathlib import Path
from typing import Optional

from pydantic import ValidationError

from ..contracts import ScanResult
from ..core.errors import InvalidInputError
from ..statistics import ScanStatistics, calculate_statistics
from .models import REPORT_SCHEMA_VERSION, JSONReport, PatchApplicationReport


class ReportLoadError(InvalidInputError):
    """Raised for any problem loading a saved scan result."""


def load_scan_result(path: Path) -> tuple:
    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ReportLoadError(f"could not read '{path}': {exc}") from exc

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ReportLoadError(f"'{path}' is not valid JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise ReportLoadError(f"'{path}' does not contain a SentinelAI report object.")

    schema_version = data.get("schema_version")
    if schema_version != REPORT_SCHEMA_VERSION:
        raise ReportLoadError(
            f"unsupported SentinelAI report schema version: {schema_version!r} "
            f"(expected {REPORT_SCHEMA_VERSION!r})"
        )

    try:
        report = JSONReport.model_validate(data)
    except ValidationError as exc:
        raise ReportLoadError(f"'{path}' does not match the SentinelAI report schema:\n{exc}") from exc

    result = ScanResult(
        repository=report.repository,
        metadata=report.scan,
        scanner_findings=report.findings.scanner,
        ai_findings=report.findings.ai_enriched,
        # Reconstructed, not recomputed: correlation is deterministic, but re-running
        # it here would silently re-derive groups from a possibly hand-edited report
        # rather than reporting what the scan actually found. Absent in a
        # pre-correlation report, where it validates to [].
        correlated_findings=report.findings.correlated,
    )
    statistics: ScanStatistics = calculate_statistics(result)
    return result, statistics


def load_patch_application(path: Path) -> Optional[PatchApplicationReport]:
    """Read a saved report's patch section, or None when it has none.

    A separate function rather than a third element on load_scan_result's tuple:
    that signature is consumed by existing callers and widening it would break
    every one of them for a value most do not want. A report written before this
    field existed, or by a scan run without --apply-patches, returns None.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReportLoadError(f"could not read '{path}': {exc}") from exc

    if not isinstance(data, dict) or data.get("patch_application") is None:
        return None

    try:
        return PatchApplicationReport.model_validate(data["patch_application"])
    except ValidationError as exc:
        raise ReportLoadError(
            f"'{path}' has a patch_application section that does not match the schema:\n{exc}"
        ) from exc
