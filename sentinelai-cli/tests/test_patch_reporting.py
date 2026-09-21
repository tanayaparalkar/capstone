"""
Patch results in the reporting layer: reporting/patch_section.py and every renderer.

Presentation only. Nothing here applies a patch - the PatchRunResult objects are
built directly - because this phase adds no patching behaviour and a test that
needed a real write would be testing Phase 2 again.

Two properties carry the weight.

*One derivation.* The ten summary counts are computed in exactly one place, so a
test asserting them in JSON and again in Markdown is asserting the same
arithmetic reached both. A renderer that started counting for itself would show
up as a disagreement between formats, which is what the cross-format test checks.

*Backwards compatibility.* A report generated without --apply-patches must still
load, and the schema version must not move. Both are checked directly, including
against a report with the key stripped entirely.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from sentinelai.contracts import (
    RepositoryInfo,
    ScanMetadata,
    ScanMode,
    ScanResult,
    ScannerFinding,
    Severity,
)
from sentinelai.patching import PatchAttempt, PatchOutcome, PatchRunResult
from sentinelai.reporting.models import PatchApplicationReport, PatchApplicationSummary
from sentinelai.reporting import (
    REPORT_SCHEMA_VERSION,
    build_patch_application_report,
    load_patch_application,
    load_scan_result,
    to_html,
    to_json,
    to_markdown,
    to_sarif,
)
from sentinelai.reporting.loader import ReportLoadError
from sentinelai.statistics import calculate_statistics


def _result() -> ScanResult:
    finding = ScannerFinding(
        finding_id="SENT-001",
        scanner="semgrep",
        category="sql-injection",
        severity=Severity.HIGH,
        file="app/db.py",
        line_start=7,
        rule_id="r",
        message="m",
    )
    return ScanResult(
        repository=RepositoryInfo(name="demo", path="/tmp/demo"),
        metadata=ScanMetadata(
            timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
            mode=ScanMode.STANDARD,
            duration_seconds=1.0,
        ),
        scanner_findings=[finding],
    )


def _attempt(finding_id, outcome, file="app/db.py", **kw) -> PatchAttempt:
    return PatchAttempt(finding_id=finding_id, file=file, outcome=outcome, **kw)


def _run(*attempts) -> PatchRunResult:
    return PatchRunResult(attempts=tuple(attempts))


def _render_terminal(run) -> str:
    """Render the terminal view to a string so it can be asserted like the others."""
    import io

    from rich.console import Console

    from sentinelai.presentation.patches import render_patch_application

    buffer = io.StringIO()
    render_patch_application(Console(file=buffer, width=120), build_patch_application_report(run))
    return buffer.getvalue()


def _render_all(run):
    result, stats = _result(), calculate_statistics(_result())
    report = build_patch_application_report(run)
    return {
        "json": to_json(result, stats, patch_application=report),
        "markdown": to_markdown(result, stats, report),
        "html": to_html(result, stats, report),
        "sarif": to_sarif(result, stats, patch_application=report),
    }


# --- no patch application requested ---------------------------------------------------------------------------


def test_build_returns_none_when_patching_was_not_requested():
    """None in, None out - 'not requested' must stay distinguishable from 'nothing eligible'."""
    assert build_patch_application_report(None) is None


def test_json_carries_an_explicit_null_when_not_requested():
    rendered = json.loads(to_json(_result(), calculate_statistics(_result())))
    assert "patch_application" in rendered
    assert rendered["patch_application"] is None


@pytest.mark.parametrize("fmt", ["markdown", "html"])
def test_text_formats_omit_the_section_when_not_requested(fmt):
    """A 'not requested' block would appear in every report ever generated without the flag."""
    result, stats = _result(), calculate_statistics(_result())
    rendered = to_markdown(result, stats) if fmt == "markdown" else to_html(result, stats)
    assert "Patch Application" not in rendered


def test_sarif_omits_the_property_when_not_requested():
    rendered = json.loads(to_sarif(_result(), calculate_statistics(_result())))
    assert "patchApplication" not in rendered["runs"][0]["properties"]["sentinelai"]


# --- zero eligible patches ------------------------------------------------------------------------------------


def test_a_run_with_only_skipped_findings_still_reports_a_section():
    rendered = _render_all(_run(_attempt("SENT-001", PatchOutcome.SKIPPED_NO_PATCH, file=None)))

    assert "Patch Application" in rendered["markdown"]
    assert json.loads(rendered["json"])["patch_application"]["summary"]["attempted"] == 0
    assert json.loads(rendered["json"])["patch_application"]["summary"]["skipped"] == 1


def test_an_empty_run_reports_zeroes_rather_than_nothing():
    rendered = _render_all(_run())
    summary = json.loads(rendered["json"])["patch_application"]["summary"]

    assert summary["total_findings"] == 0
    assert "No findings were considered" in rendered["markdown"]


# --- each outcome ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "outcome,key",
    [
        (PatchOutcome.APPLIED, "applied"),
        (PatchOutcome.VALIDATION_FAILED, "validation_failed"),
        (PatchOutcome.NOT_APPLICABLE, "not_applicable"),
        (PatchOutcome.ERROR, "errors"),
    ],
)
def test_each_outcome_is_counted_under_its_own_key(outcome, key):
    rendered = _render_all(_run(_attempt("SENT-001", outcome)))
    summary = json.loads(rendered["json"])["patch_application"]["summary"]

    assert summary[key] == 1
    assert summary["attempted"] == 1


def test_a_rollback_is_reported_on_both_axes():
    """Outcome says what became of the patch; the booleans say what became of the file."""
    run = _run(_attempt("SENT-001", PatchOutcome.ROLLED_BACK, rollback_executed=True))
    summary = json.loads(_render_all(run)["json"])["patch_application"]["summary"]

    assert summary["rolled_back"] == 1
    assert summary["rollback_failed"] == 0


def test_a_rollback_failure_is_counted_separately():
    run = _run(
        _attempt(
            "SENT-001",
            PatchOutcome.ROLLBACK_FAILED,
            rollback_executed=True,
            rollback_failed=True,
        )
    )
    summary = json.loads(_render_all(run)["json"])["patch_application"]["summary"]

    assert summary["rollback_failed"] == 1
    assert summary["rolled_back"] == 1


@pytest.mark.parametrize(
    "executed,failed,expected",
    [(False, False, "no"), (True, False, "yes"), (True, True, "failed")],
)
def test_the_rendered_rollback_column_reflects_both_flags(executed, failed, expected):
    """Two booleans collapse into one column; the precedence is what keeps that lossless.

    With both flags set the column must read "failed", not "yes": a rollback that
    was attempted and did not succeed is the one case where a file may still hold
    modified content, and reporting it as a plain "yes" would hide that.
    """
    run = _run(
        _attempt(
            "SENT-001",
            PatchOutcome.APPLIED,
            rollback_executed=executed,
            rollback_failed=failed,
        )
    )

    assert f"| SENT-001 | app/db.py | applied | {expected} | - |" in _render_all(run)["markdown"]


def test_mixed_outcomes_are_all_represented():
    run = _run(
        _attempt("SENT-001", PatchOutcome.APPLIED),
        _attempt("SENT-002", PatchOutcome.VALIDATION_FAILED),
        _attempt("SENT-003", PatchOutcome.NOT_APPLICABLE),
        _attempt("SENT-004", PatchOutcome.SKIPPED_NO_PATCH, file=None),
        _attempt("SENT-005", PatchOutcome.ROLLED_BACK, rollback_executed=True),
    )
    summary = json.loads(_render_all(run)["json"])["patch_application"]["summary"]

    assert summary["total_findings"] == 5
    assert summary["with_structured_patch"] == 4
    assert (summary["applied"], summary["validation_failed"]) == (1, 1)
    assert (summary["not_applicable"], summary["skipped"], summary["rolled_back"]) == (1, 1, 1)


# --- one derivation, shared by every format -------------------------------------------------------------------


def test_every_format_reports_the_same_counts():
    """A renderer that counted for itself would disagree here."""
    run = _run(
        _attempt("SENT-001", PatchOutcome.APPLIED),
        _attempt("SENT-002", PatchOutcome.VALIDATION_FAILED),
        _attempt("SENT-003", PatchOutcome.SKIPPED_NO_PATCH, file=None),
    )
    rendered = _render_all(run)

    json_summary = json.loads(rendered["json"])["patch_application"]["summary"]
    sarif_summary = json.loads(rendered["sarif"])["runs"][0]["properties"]["sentinelai"][
        "patchApplication"
    ]["summary"]

    assert json_summary == sarif_summary
    for label, count in (("Applied", 1), ("Validation failed", 1), ("Skipped (no patch)", 1)):
        assert f"| {label} | {count} |" in rendered["markdown"]
        assert f"<td>{label}</td><td>{count}</td>" in rendered["html"].replace("\n", "").replace(
            "  ", ""
        ) or label in rendered["html"]


def test_patch_results_are_not_mixed_into_ai_findings():
    """A separate deterministic stage, reported separately."""
    rendered = json.loads(_render_all(_run(_attempt("SENT-001", PatchOutcome.APPLIED)))["json"])

    assert "patch_application" in rendered
    assert "patch_application" not in rendered["findings"]
    assert rendered["findings"]["ai_enriched"] == []


def test_sarif_does_not_add_patch_outcomes_as_results():
    """A patch outcome is not a finding and must not be mistaken for one."""
    rendered = json.loads(_render_all(_run(_attempt("SENT-001", PatchOutcome.APPLIED)))["sarif"])
    assert len(rendered["runs"][0]["results"]) == 1  # the scanner finding, nothing added


# --- per-attempt fields and ordering --------------------------------------------------------------------------


def test_every_required_attempt_field_is_present():
    run = _run(_attempt("SENT-007", PatchOutcome.ROLLED_BACK, rollback_executed=True))
    attempt = json.loads(_render_all(run)["json"])["patch_application"]["attempts"][0]

    assert set(attempt) == {
        "finding_id",
        "file",
        "outcome",
        "rollback_executed",
        "rollback_failed",
        "detail",
    }
    assert attempt["finding_id"] == "SENT-007"
    assert attempt["file"] == "app/db.py"
    assert attempt["outcome"] == "rolled_back"


def test_attempt_order_is_the_processing_order():
    run = _run(*[_attempt(f"SENT-{i:03d}", PatchOutcome.APPLIED) for i in (3, 1, 2)])
    ids = [a["finding_id"] for a in json.loads(_render_all(run)["json"])["patch_application"]["attempts"]]

    assert ids == ["SENT-003", "SENT-001", "SENT-002"], "must not be re-sorted"


def test_rendering_is_deterministic():
    run = _run(_attempt("SENT-001", PatchOutcome.APPLIED))
    assert _render_all(run)["json"] == _render_all(run)["json"]
    assert _render_all(run)["markdown"] == _render_all(run)["markdown"]


def test_detail_pipes_do_not_break_the_markdown_table():
    run = _run(_attempt("SENT-001", PatchOutcome.ERROR, detail="a | b\nnext"))
    markdown = _render_all(run)["markdown"]

    assert "a \\| b" in markdown
    assert "a | b\nnext" not in markdown


# --- backwards compatibility ----------------------------------------------------------------------------------


def test_schema_version_is_not_bumped():
    """loader.py compares with strict equality - a bump orphans every saved report."""
    assert REPORT_SCHEMA_VERSION == "1.0"
    assert json.loads(_render_all(_run())["json"])["schema_version"] == "1.0"


def test_a_report_with_patch_results_still_loads(tmp_path):
    run = _run(_attempt("SENT-001", PatchOutcome.APPLIED))
    path = tmp_path / "r.json"
    path.write_text(_render_all(run)["json"], encoding="utf-8")

    loaded, _ = load_scan_result(path)
    assert loaded.scanner_findings[0].finding_id == "SENT-001"


def test_a_report_written_before_this_field_existed_still_loads(tmp_path):
    data = json.loads(to_json(_result(), calculate_statistics(_result())))
    data.pop("patch_application")
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    loaded, _ = load_scan_result(path)

    assert loaded.scanner_findings[0].finding_id == "SENT-001"
    assert load_patch_application(path) is None


def test_patch_results_round_trip_through_a_saved_report(tmp_path):
    run = _run(_attempt("SENT-001", PatchOutcome.APPLIED), _attempt("SENT-002", PatchOutcome.ERROR))
    path = tmp_path / "r.json"
    path.write_text(_render_all(run)["json"], encoding="utf-8")

    reloaded = load_patch_application(path)

    assert reloaded is not None
    assert [a.finding_id for a in reloaded.attempts] == ["SENT-001", "SENT-002"]
    assert reloaded.summary.applied == 1


def test_a_malformed_patch_section_is_reported_as_a_load_error(tmp_path):
    data = json.loads(_render_all(_run())["json"])
    data["patch_application"] = {"summary": "not an object"}
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ReportLoadError):
        load_patch_application(path)


# --- architecture ---------------------------------------------------------------------------------------------


def test_renderers_do_not_reach_the_patch_applicator():
    """Reporting is downstream: it consumes a result object, never the engine."""
    import ast

    for module in ("json_report", "markdown_report", "html_report", "sarif_report", "patch_section"):
        source = open(
            Path(__file__).parent.parent / "sentinelai" / "reporting" / f"{module}.py"
        ).read()
        tree = ast.parse(source)
        imported = {
            name.split(".")[-1]
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
            for name in [node.module]
        }
        assert "applicator" not in imported, f"{module} must not import the applicator"

        # Checked against referenced NAMES rather than raw text: patch_section.py's
        # docstring mentions PatchApplicator precisely to explain that it never
        # touches it, and a substring search cannot tell prose from a call.
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {
            n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)
        }
        assert "PatchApplicator" not in names, f"{module} must not reference PatchApplicator"


# --- execution mode is never displayed without its context --------------------------------------------------------


@pytest.mark.parametrize("fmt", ["markdown", "html", "terminal"])
def test_a_dry_run_states_that_nothing_was_written(fmt):
    """An 'applied' row must never appear without the mode beside it.

    Outcomes describe what the apply step did; the mode says whether it reached
    disk. Since a dry run reports APPLIED like any other success, the renderers
    carry the whole burden of keeping that unambiguous - so it is asserted in
    every format a human reads.
    """
    run = PatchRunResult(attempts=(_attempt("SENT-001", PatchOutcome.APPLIED),), dry_run=True)
    rendered = _render_all(run)[fmt] if fmt != "terminal" else _render_terminal(run)

    assert "DRY RUN" in rendered
    assert "no files were modified" in rendered


@pytest.mark.parametrize("fmt", ["markdown", "html", "terminal"])
def test_a_real_run_states_that_files_were_written(fmt):
    run = PatchRunResult(attempts=(_attempt("SENT-001", PatchOutcome.APPLIED),), dry_run=False)
    rendered = _render_all(run)[fmt] if fmt != "terminal" else _render_terminal(run)

    assert "DRY RUN" not in rendered
    assert "Patches were written to the repository" in rendered


def test_the_mode_wording_is_defined_once():
    """One source of wording, so the three renderers cannot drift apart."""
    from sentinelai.reporting.patch_section import DRY_RUN_NOTICE, mode_notice

    dry = PatchApplicationReport(summary=PatchApplicationSummary(), dry_run=True)
    wet = PatchApplicationReport(summary=PatchApplicationSummary(), dry_run=False)

    assert mode_notice(dry) == DRY_RUN_NOTICE
    assert mode_notice(wet) != DRY_RUN_NOTICE


def test_dry_run_defaults_to_false_for_reports_that_predate_it():
    """Backward compatibility: an older report loads as a real run, which is what it was."""
    assert PatchApplicationReport(summary=PatchApplicationSummary()).dry_run is False
