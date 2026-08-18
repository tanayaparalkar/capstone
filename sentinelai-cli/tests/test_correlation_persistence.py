"""
Correlation across the full report lifecycle:
scan -> JSON -> `sentinelai report` reload -> rendered output and statistics.

Correlation was initially computed at scan time but dropped by the JSON
envelope, so a saved report reloaded with `sentinelai report` showed
every raw finding and zero correlated issues. That is a silent
downgrade: the numbers changed depending on whether a report was
rendered directly or from a saved file, which is exactly the drift the
single-source-of-truth statistics engine exists to prevent.

The fixture mirrors the real benchmark shape - 17 raw findings that
correlate into 13 issues, 4 of them corroborated by two scanners - so
the reload assertions are checked against the same numbers the paper
reports rather than against a toy case.
"""
import json

import pytest

from sentinelai.contracts import (
    CorrelationRule,
    RepositoryInfo,
    ScanMetadata,
    ScanMode,
    ScanResult,
    ScannerFinding,
    Severity,
)
from sentinelai.correlation import correlate_findings
from sentinelai.reporting import to_html, to_json, to_markdown, to_sarif
from sentinelai.reporting.loader import load_scan_result
from sentinelai.reporting.models import REPORT_SCHEMA_VERSION, JSONReport
from sentinelai.statistics import calculate_statistics


def _f(fid, scanner, cwe, file, line, severity=Severity.MEDIUM, category="blacklist") -> ScannerFinding:
    return ScannerFinding(
        finding_id=fid, scanner=scanner, category=category, severity=severity,
        file=file, line_start=line, line_end=line, rule_id="R1",
        message="m", raw_evidence="e", cwe=cwe,
    )


def _benchmark_shaped_findings() -> list[ScannerFinding]:
    """17 raw findings that correlate to 13 issues with 4 multi-scanner groups.

    Mirrors the real benchmark: four co-located same-CWE pairs (one of them on
    adjacent lines), one different-CWE pair that must not merge, one CWE-less
    GitLeaks hit that must not merge, and singletons.
    """
    return [
        # 4 merging pairs -> 4 groups
        _f("bandit-0", "bandit", "CWE-327", "crypto.py", 6, Severity.HIGH),
        _f("semgrep-0", "semgrep", "CWE-327: Broken Crypto", "crypto.py", 6),
        _f("bandit-1", "bandit", "CWE-89", "db.py", 6),
        _f("semgrep-1", "semgrep", "CWE-89: SQLi", "db.py", 7, Severity.HIGH),  # adjacent
        _f("bandit-4", "bandit", "CWE-502", "tasks.py", 13),
        _f("semgrep-2", "semgrep", "CWE-502: Deserialization", "tasks.py", 13),
        _f("bandit-6", "bandit", "CWE-78", "tasks.py", 21, Severity.HIGH),
        _f("semgrep-3", "semgrep", "CWE-78: OS Command", "tasks.py", 21, Severity.HIGH),
        # different CWE on one line -> must stay 2
        _f("bandit-9", "bandit", "CWE-78", "tasks.py", 29),
        _f("semgrep-4", "semgrep", "CWE-95: Eval Injection", "tasks.py", 29),
        # gitleaks has no CWE -> must stay 2
        _f("gitleaks-0", "gitleaks", None, "settings.py", 14, Severity.HIGH, "generic-api-key"),
        _f("semgrep-5", "semgrep", "CWE-798: Hard-coded Credentials", "settings.py", 14, Severity.HIGH),
        # singletons
        _f("bandit-2", "bandit", "CWE-502", "tasks.py", 6, Severity.LOW),
        _f("bandit-3", "bandit", "CWE-78", "tasks.py", 7, Severity.LOW),
        _f("bandit-5", "bandit", "CWE-78", "tasks.py", 17, Severity.HIGH),
        _f("bandit-7", "bandit", "CWE-20", "tasks.py", 25),
        _f("bandit-8", "bandit", "CWE-259", "settings.py", 15, Severity.LOW),
    ]


def _live_result(findings=None) -> ScanResult:
    findings = findings if findings is not None else _benchmark_shaped_findings()
    return ScanResult(
        repository=RepositoryInfo(name="sentinelai-manual-test", path="/tmp/bench"),
        metadata=ScanMetadata(timestamp="2026-08-18T12:00:00Z", mode=ScanMode.STANDARD),
        scanner_findings=findings,
        correlated_findings=correlate_findings(findings),
    )


def _save(result, tmp_path, name="report.json"):
    path = tmp_path / name
    path.write_text(to_json(result, calculate_statistics(result)), encoding="utf-8")
    return path


def _render(result, renderer):
    return renderer(result, calculate_statistics(result))


# --- the fixture matches the benchmark numbers ---------------------------------------------------------------------------


def test_fixture_reproduces_the_benchmark_correlation_shape():
    result = _live_result()
    stats = calculate_statistics(result)

    assert stats.total_findings == 17
    assert stats.correlated_findings == 13
    assert stats.multi_scanner_findings == 4


# --- round trip ------------------------------------------------------------------------------------------------------------


def test_correlation_survives_json_round_trip(tmp_path):
    live = _live_result()

    reloaded, _ = load_scan_result(_save(live, tmp_path))

    assert sorted(reloaded.correlated_findings, key=lambda c: c.correlation_id) == sorted(
        live.correlated_findings, key=lambda c: c.correlation_id
    )


def test_statistics_after_reload_still_show_13_and_4(tmp_path):
    live = _live_result()
    before = calculate_statistics(live)

    _, after = load_scan_result(_save(live, tmp_path))

    assert (after.total_findings, after.correlated_findings, after.multi_scanner_findings) == (17, 13, 4)
    assert (before.total_findings, before.correlated_findings, before.multi_scanner_findings) == (17, 13, 4)


def test_every_correlation_field_is_preserved_exactly(tmp_path):
    live = _live_result()
    reloaded, _ = load_scan_result(_save(live, tmp_path))

    original = {c.correlation_id: c for c in live.correlated_findings}
    for group in reloaded.correlated_findings:
        source = original[group.correlation_id]
        assert group.canonical_finding_id == source.canonical_finding_id
        assert group.source_finding_ids == source.source_finding_ids
        assert group.scanners == source.scanners
        assert group.file == source.file
        assert (group.line_start, group.line_end) == (source.line_start, source.line_end)
        assert group.cwe == source.cwe
        assert group.severity is source.severity
        assert group.category == source.category
        assert group.rule is source.rule
        assert group.correlation_reason == source.correlation_reason


def test_serialized_json_contains_every_correlation_field(tmp_path):
    payload = json.loads(_save(_live_result(), tmp_path).read_text(encoding="utf-8"))
    groups = payload["findings"]["correlated"]

    assert len(groups) == 13
    expected = {
        "correlation_id", "canonical_finding_id", "source_finding_ids", "scanners",
        "file", "line_start", "line_end", "cwe", "severity", "category",
        "rule", "correlation_reason",
    }
    assert expected <= set(groups[0])


def test_normalized_cwe_is_what_is_persisted(tmp_path):
    # Raw scanner strings stay on the raw findings; the group carries only the
    # normalized form.
    payload = json.loads(_save(_live_result(), tmp_path).read_text(encoding="utf-8"))
    merged = next(g for g in payload["findings"]["correlated"] if len(g["source_finding_ids"]) > 1)

    assert merged["cwe"].startswith("CWE-")
    assert ":" not in merged["cwe"]
    raw_semgrep = next(f for f in payload["findings"]["scanner"] if f["finding_id"] == "semgrep-0")
    assert ":" in raw_semgrep["cwe"]  # untouched on the raw finding


def test_raw_findings_are_unchanged_by_the_round_trip(tmp_path):
    # Compared as sets: build_json_report() deliberately sorts findings by
    # finding_id for a deterministic report, so the reloaded order differs from
    # the order the scanners produced. The findings themselves must be identical.
    live = _live_result()

    reloaded, _ = load_scan_result(_save(live, tmp_path))

    key = lambda f: f.finding_id  # noqa: E731
    assert sorted(reloaded.scanner_findings, key=key) == sorted(live.scanner_findings, key=key)


def test_serialized_groups_are_ordered_deterministically(tmp_path):
    payload = json.loads(_save(_live_result(), tmp_path).read_text(encoding="utf-8"))
    ids = [g["correlation_id"] for g in payload["findings"]["correlated"]]

    assert ids == sorted(ids)


# --- backward compatibility ----------------------------------------------------------------------------------------------


def test_pre_correlation_report_loads_with_empty_correlation(tmp_path):
    payload = json.loads(_save(_live_result(), tmp_path).read_text(encoding="utf-8"))
    del payload["findings"]["correlated"]
    path = tmp_path / "old.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    result, stats = load_scan_result(path)

    assert result.correlated_findings == []
    assert len(result.scanner_findings) == 17  # raw findings unaffected
    assert stats.correlated_findings == 0
    assert stats.multi_scanner_findings == 0


def test_schema_version_is_unchanged_so_old_reports_stay_loadable(tmp_path):
    # The loader rejects any version that is not an exact match, so bumping it
    # would make every previously-saved report unloadable.
    assert REPORT_SCHEMA_VERSION == "1.0"
    payload = json.loads(_save(_live_result(), tmp_path).read_text(encoding="utf-8"))
    assert payload["schema_version"] == "1.0"


def test_envelope_defaults_correlated_to_empty():
    report = JSONReport.model_validate(
        {
            "schema_version": REPORT_SCHEMA_VERSION,
            "repository": {"name": "d", "path": "/tmp"},
            "scan": {"timestamp": "2026-08-18T12:00:00Z", "mode": "standard"},
            "statistics": calculate_statistics(_live_result()).model_dump(mode="json"),
            "findings": {"scanner": [], "ai_enriched": []},
        }
    )

    assert report.findings.correlated == []


# --- renderers driven from the RELOADED result ------------------------------------------------------------------------------


@pytest.fixture
def reloaded(tmp_path):
    result, _ = load_scan_result(_save(_live_result(), tmp_path))
    return result


def test_markdown_from_reloaded_result_shows_correlation(reloaded):
    md = _render(reloaded, to_markdown)

    assert "**Total Findings:** 17" in md
    assert "**Correlated Issues:** 13" in md
    assert "4 confirmed by 2+ scanners" in md
    assert "**Correlated Issue:** CORR-" in md
    assert "**Correlation Basis:**" in md


def test_html_from_reloaded_result_shows_correlation(reloaded):
    html = _render(reloaded, to_html)

    assert "Correlated Issues" in html
    assert ">13<" in html
    assert "CORR-" in html
    assert "Correlation Basis" in html


def test_sarif_from_reloaded_result_shows_correlation(reloaded):
    sarif = json.loads(_render(reloaded, to_sarif))
    props = {
        r["properties"]["sentinelai"]["findingId"]: r["properties"]["sentinelai"]
        for r in sarif["runs"][0]["results"]
    }

    assert len(sarif["runs"][0]["results"]) == 17  # one per RAW finding
    merged = props["semgrep-0"]
    assert merged["canonicalFindingId"] == "bandit-0"
    assert sorted(merged["sourceFindingIds"]) == ["bandit-0", "semgrep-0"]
    assert merged["correlatedScanners"] == ["bandit", "semgrep"]
    stats = sarif["runs"][0]["properties"]["sentinelai"]["statistics"]
    assert (stats["correlated_findings"], stats["multi_scanner_findings"]) == (13, 4)


def test_terminal_from_reloaded_result_still_renders(reloaded):
    from rich.console import Console

    from sentinelai.presentation.findings import render_findings_table

    console = Console(record=True, width=200)
    render_findings_table(console, reloaded)
    text = console.export_text()

    # Every raw finding is still listed - correlation never hides one.
    for finding in reloaded.scanner_findings:
        assert finding.finding_id in text


def test_json_regenerated_from_reload_is_semantically_identical(tmp_path):
    """scan -> json -> reload -> json is a fixed point in content.

    Compared as parsed JSON rather than as bytes. `statistics.scanner_counts` is
    built by iterating findings, so its key *insertion order* follows the order
    the scanners ran in on a live result and finding_id order after a reload.
    The counts are equal either way; only the key order in the serialized object
    differs, which JSON does not treat as significant. This is pre-existing
    behaviour of the statistics engine, unrelated to correlation.
    """
    live = _live_result()
    first = _save(live, tmp_path, "first.json").read_text(encoding="utf-8")
    reloaded, stats = load_scan_result(tmp_path / "first.json")

    assert json.loads(to_json(reloaded, stats)) == json.loads(first)


def test_adjacent_line_rule_survives_the_round_trip(tmp_path):
    reloaded, _ = load_scan_result(_save(_live_result(), tmp_path))

    adjacent = [c for c in reloaded.correlated_findings if c.rule is CorrelationRule.SAME_CWE_ADJACENT_LINES]

    assert len(adjacent) == 1
    assert sorted(adjacent[0].source_finding_ids) == ["bandit-1", "semgrep-1"]
    assert "adjacent lines" in adjacent[0].correlation_reason
