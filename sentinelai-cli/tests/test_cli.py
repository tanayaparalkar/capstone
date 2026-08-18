"""
CLI integration tests.

Run the real Typer app via CliRunner against the real MockFindingsProvider
(no mocking of contracts/data) so these tests exercise the actual
CLI -> FindingsProvider -> ScanResult -> display/formatters flow. Provider
usage is verified by monkeypatching MockFindingsProvider.get_scan_result
to record calls while still delegating to the original implementation.

_get_provider() itself now defaults to LiveFindingsProvider in production
(sentinelai/providers/live_provider.py) - real scanner execution has its
own dedicated test suite (tests/test_live_provider.py) and no business
running through 74 CLI-behavior assertions that were never testing "which
provider," only rendering/filtering/formatting given a ScanResult. The
autouse fixture below pins _get_provider() back to MockFindingsProvider
for every test in this file, so all of them keep exercising deterministic,
tool-free mock data exactly as before. The one test that verifies the
production default actually changed lives in test_provider_selection.py
instead - not here, since an autouse fixture scoped to this module would
otherwise patch it before it runs, defeating its own point.
"""
import json

import pytest
from typer.testing import CliRunner

from sentinelai.contracts import ScanMode, ScannerTier
from sentinelai.core import ExitCode
from sentinelai.main import app
from sentinelai.providers import MockFindingsProvider

runner = CliRunner()


@pytest.fixture(autouse=True)
def _use_mock_provider(monkeypatch):
    """Every test in this file exercises the CLI against MockFindingsProvider, not the live default."""
    import sentinelai.main as main_module

    monkeypatch.setattr(main_module, "_get_provider", lambda: MockFindingsProvider())


def _spy_get_scan_result(monkeypatch):
    """Wrap MockFindingsProvider.get_scan_result to record (repo_path, mode, tier) calls."""
    calls = []
    original = MockFindingsProvider.get_scan_result

    def wrapper(self, repo_path, mode=ScanMode.STANDARD, tier=ScannerTier.CORE):
        calls.append((repo_path, mode, tier))
        return original(self, repo_path, mode=mode, tier=tier)

    monkeypatch.setattr(MockFindingsProvider, "get_scan_result", wrapper)
    return calls


def test_version_command():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "SentinelAI" in result.output


def test_scan_default_path_runs():
    result = runner.invoke(app, ["scan", "."])
    assert result.exit_code == 0
    assert "SentinelAI Findings" in result.output


def test_scan_default_mode_is_standard(monkeypatch):
    calls = _spy_get_scan_result(monkeypatch)
    result = runner.invoke(app, ["scan", "."])
    assert result.exit_code == 0
    assert calls[0][1] == ScanMode.STANDARD


def test_scan_quick_selects_quick_mode(monkeypatch):
    calls = _spy_get_scan_result(monkeypatch)
    result = runner.invoke(app, ["scan", ".", "--quick"])
    assert result.exit_code == 0
    assert calls[0][1] == ScanMode.QUICK


def test_scan_full_selects_full_mode(monkeypatch):
    calls = _spy_get_scan_result(monkeypatch)
    result = runner.invoke(app, ["scan", ".", "--full"])
    assert result.exit_code == 0
    assert calls[0][1] == ScanMode.FULL


def test_scan_quick_and_full_together_errors():
    result = runner.invoke(app, ["scan", ".", "--quick", "--full"])
    assert result.exit_code == ExitCode.INVALID_INPUT


def test_scan_severity_filtering():
    result = runner.invoke(app, ["scan", ".", "--severity", "critical", "--format", "json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    findings = data["findings"]["scanner"]
    assert len(findings) > 0
    assert all(f["severity"] == "critical" for f in findings)
    # Statistics must reflect the same filtered result, not the full scan.
    assert data["statistics"]["total_findings"] == len(findings)


def test_scan_no_severity_filter_returns_all_mock_findings():
    result = runner.invoke(app, ["scan", ".", "--format", "json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert len(data["findings"]["scanner"]) == 10


def test_scan_invalid_severity_errors():
    result = runner.invoke(app, ["scan", ".", "--severity", "extreme"])
    assert result.exit_code == ExitCode.INVALID_INPUT
    assert "severity" in result.output.lower()


def test_scan_invalid_format_errors():
    result = runner.invoke(app, ["scan", ".", "--format", "xml"])
    assert result.exit_code == ExitCode.INVALID_INPUT
    assert "format" in result.output.lower()


def test_scan_nonexistent_path_errors():
    result = runner.invoke(app, ["scan", "/definitely/does/not/exist/xyz"])
    assert result.exit_code == ExitCode.INVALID_INPUT


def test_scan_uses_the_findings_provider(monkeypatch):
    calls = _spy_get_scan_result(monkeypatch)
    result = runner.invoke(app, ["scan", "."])
    assert result.exit_code == 0
    assert len(calls) == 1
    assert calls[0][0] == "."


def test_scan_json_output_is_valid_json():
    result = runner.invoke(app, ["scan", ".", "--format", "json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert "schema_version" in data
    assert "repository" in data
    assert "scan" in data
    assert "statistics" in data
    assert "scanner" in data["findings"]
    assert "ai_enriched" in data["findings"]


def test_scan_markdown_format_runs():
    result = runner.invoke(app, ["scan", ".", "--format", "markdown"])
    assert result.exit_code == 0
    assert "# SentinelAI Security Report" in result.output


def test_scan_html_format_runs():
    result = runner.invoke(app, ["scan", ".", "--format", "html"])
    assert result.exit_code == 0
    assert "<html" in result.output.lower()


def test_scan_output_file_option_writes_file(tmp_path):
    out_file = tmp_path / "report.json"
    result = runner.invoke(app, ["scan", ".", "--format", "json", "-o", str(out_file)])
    assert result.exit_code == 0
    assert out_file.exists()
    data = json.loads(out_file.read_text(encoding="utf-8"))
    assert len(data["findings"]["scanner"]) == 10


# --- Milestone 3: terminal presentation -----------------------------------------------------------------


def test_scan_explicit_terminal_format_shows_presentation():
    result = runner.invoke(app, ["scan", ".", "--format", "terminal"])
    assert result.exit_code == 0
    assert "SentinelAI Security Scan" in result.output
    assert "Security Summary" in result.output
    assert "SentinelAI Findings" in result.output


def test_scan_default_format_shows_header_progress_and_summary():
    result = runner.invoke(app, ["scan", "."])
    assert result.exit_code == 0
    assert "SentinelAI Security Scan" in result.output
    assert "Retrieving scan results" in result.output
    assert "Security Summary" in result.output


def test_scan_does_not_fake_scanner_or_ai_stages():
    result = runner.invoke(app, ["scan", "."])
    assert result.exit_code == 0
    lowered = result.output.lower()
    for fake_stage in ("semgrep completed", "bandit completed", "ai reasoning completed", "verification completed"):
        assert fake_stage not in lowered


def test_scan_details_shows_single_finding():
    result = runner.invoke(app, ["scan", ".", "--details", "SENT-002"])
    assert result.exit_code == 0
    assert "SENT-002" in result.output
    assert "sql-injection" in result.output
    # The detail view replaces the full table, not adds to it.
    assert "SentinelAI Findings" not in result.output


def test_scan_details_invalid_id_errors():
    result = runner.invoke(app, ["scan", ".", "--details", "SENT-999"])
    assert result.exit_code == ExitCode.INVALID_INPUT
    assert "SENT-999" in result.output


def test_scan_details_ignored_for_json_format():
    result = runner.invoke(app, ["scan", ".", "--details", "SENT-002", "--format", "json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert len(data["findings"]["scanner"]) == 10  # not filtered down to one finding


# --- Milestone 3: structured output must stay uncontaminated -----------------------------------------------------------------


def test_scan_json_output_not_contaminated_by_terminal_presentation():
    result = runner.invoke(app, ["scan", ".", "--format", "json"])
    assert result.exit_code == 0
    assert "SentinelAI Security Scan" not in result.output
    assert "Security Summary" not in result.output
    assert "\x1b[" not in result.output
    json.loads(result.output)  # still parses cleanly


def test_scan_markdown_output_not_contaminated_by_terminal_presentation():
    result = runner.invoke(app, ["scan", ".", "--format", "markdown"])
    assert result.exit_code == 0
    assert "SentinelAI Security Scan" not in result.output
    assert "Security Summary" not in result.output
    assert "\x1b[" not in result.output


def test_scan_html_output_not_contaminated_by_terminal_presentation():
    result = runner.invoke(app, ["scan", ".", "--format", "html"])
    assert result.exit_code == 0
    assert "SentinelAI Security Scan" not in result.output
    assert "Security Summary" not in result.output
    assert "\x1b[" not in result.output


# --- Milestone 5: JSON report generator -----------------------------------------------------------------


def test_scan_json_stdout_contains_json_only():
    result = runner.invoke(app, ["scan", ".", "--format", "json"])
    assert result.exit_code == 0
    # The entire stdout must parse as one JSON document - nothing before
    # or after it (no banner, no progress line, no trailing note).
    json.loads(result.output)


def test_scan_json_report_includes_statistics_and_repository_metadata():
    result = runner.invoke(app, ["scan", ".", "--format", "json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["statistics"]["total_findings"] == 10
    assert data["repository"]["path"] == "."
    assert data["scan"]["mode"] == "standard"


def test_scan_json_ai_findings_empty_for_mock_provider():
    result = runner.invoke(app, ["scan", ".", "--format", "json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["findings"]["ai_enriched"] == []
    assert data["statistics"]["ai_enrichment_status"] == "unavailable"


def test_scan_json_output_file_is_utf8(tmp_path):
    out_file = tmp_path / "report.json"
    result = runner.invoke(app, ["scan", ".", "--format", "json", "-o", str(out_file)])
    assert result.exit_code == 0
    raw = out_file.read_bytes()
    decoded = raw.decode("utf-8")  # raises if not valid UTF-8
    json.loads(decoded)


def test_scan_json_output_to_nonexistent_directory_errors_cleanly():
    result = runner.invoke(app, ["scan", ".", "--format", "json", "-o", "no_such_dir/report.json"])
    assert result.exit_code == ExitCode.INVALID_INPUT
    assert "Error" in result.output
    assert "Traceback" not in result.output


def test_scan_json_deterministic_across_invocations():
    first = runner.invoke(app, ["scan", ".", "--format", "json"])
    second = runner.invoke(app, ["scan", ".", "--format", "json"])
    assert first.exit_code == 0
    assert second.exit_code == 0
    first_data = json.loads(first.output)
    second_data = json.loads(second.output)
    # Timestamps/duration legitimately differ per run; finding order and
    # content must not.
    assert [f["finding_id"] for f in first_data["findings"]["scanner"]] == [
        f["finding_id"] for f in second_data["findings"]["scanner"]
    ]


# --- Milestone 6: Markdown report generator -----------------------------------------------------------------


def test_scan_markdown_stdout_contains_markdown_only():
    result = runner.invoke(app, ["scan", ".", "--format", "markdown"])
    assert result.exit_code == 0
    assert result.output.startswith("# SentinelAI Security Report")
    assert "\x1b[" not in result.output


def test_scan_markdown_report_includes_statistics_and_repository():
    result = runner.invoke(app, ["scan", ".", "--format", "markdown"])
    assert result.exit_code == 0
    assert "Total Findings:** 10" in result.output
    assert "AI Enrichment:** unavailable" in result.output


def test_scan_markdown_severity_filter_reflected_in_report():
    result = runner.invoke(app, ["scan", ".", "--severity", "critical", "--format", "markdown"])
    assert result.exit_code == 0
    assert "Total Findings:** 4" in result.output
    assert "| Critical | 4 |" in result.output
    assert "| High | 0 |" in result.output


def test_scan_markdown_output_file_writes_utf8(tmp_path):
    out_file = tmp_path / "report.md"
    result = runner.invoke(app, ["scan", ".", "--format", "markdown", "-o", str(out_file)])
    assert result.exit_code == 0
    assert out_file.exists()
    raw = out_file.read_bytes()
    text = raw.decode("utf-8")  # raises if not valid UTF-8
    assert text.startswith("# SentinelAI Security Report")


def _strip_volatile_lines(text: str) -> str:
    # Timestamp/duration legitimately vary per run; everything else must not.
    return "\n".join(line for line in text.splitlines() if "Timestamp" not in line and "Duration" not in line)


def test_scan_markdown_deterministic_across_invocations():
    first = runner.invoke(app, ["scan", ".", "--format", "markdown"])
    second = runner.invoke(app, ["scan", ".", "--format", "markdown"])
    assert first.exit_code == 0
    assert second.exit_code == 0
    assert _strip_volatile_lines(first.output) == _strip_volatile_lines(second.output)


# --- Milestone 7: HTML report generator -----------------------------------------------------------------


def test_scan_html_stdout_contains_html_only():
    result = runner.invoke(app, ["scan", ".", "--format", "html"])
    assert result.exit_code == 0
    assert result.output.strip().startswith("<!DOCTYPE html>")
    assert result.output.rstrip().endswith("</html>")
    assert "\x1b[" not in result.output


def test_scan_html_report_includes_statistics_and_repository():
    result = runner.invoke(app, ["scan", ".", "--format", "html"])
    assert result.exit_code == 0
    assert ">10<" in result.output  # total findings stat card
    assert "Unavailable" in result.output  # AI enrichment status, capitalized by the template


def test_scan_html_severity_filter_reflected_in_report():
    result = runner.invoke(app, ["scan", ".", "--severity", "critical", "--format", "html"])
    assert result.exit_code == 0
    assert ">4<" in result.output


def test_scan_html_output_file_writes_utf8(tmp_path):
    out_file = tmp_path / "report.html"
    result = runner.invoke(app, ["scan", ".", "--format", "html", "-o", str(out_file)])
    assert result.exit_code == 0
    assert out_file.exists()
    raw = out_file.read_bytes()
    text = raw.decode("utf-8")  # raises if not valid UTF-8
    assert text.strip().startswith("<!DOCTYPE html>")


def test_scan_html_no_external_resources():
    result = runner.invoke(app, ["scan", ".", "--format", "html"])
    assert result.exit_code == 0
    assert "<script src=" not in result.output
    assert '<link rel="stylesheet"' not in result.output


# --- Milestone 8: SARIF report generator -----------------------------------------------------------------


def test_scan_sarif_stdout_contains_sarif_only():
    result = runner.invoke(app, ["scan", ".", "--format", "sarif"])
    assert result.exit_code == 0
    assert "\x1b[" not in result.output
    data = json.loads(result.output)
    assert data["version"] == "2.1.0"


def test_scan_sarif_report_includes_all_findings():
    result = runner.invoke(app, ["scan", ".", "--format", "sarif"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert len(data["runs"][0]["results"]) == 10


def test_scan_sarif_severity_filter_reflected_in_results():
    result = runner.invoke(app, ["scan", ".", "--severity", "critical", "--format", "sarif"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    results = data["runs"][0]["results"]
    assert len(results) == 4
    assert all(r["level"] == "error" for r in results)


def test_scan_sarif_no_fake_ai_fields_for_mock_provider():
    result = runner.invoke(app, ["scan", ".", "--format", "sarif"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    for r in data["runs"][0]["results"]:
        assert "ai" not in r["properties"]["sentinelai"]


def test_scan_sarif_output_file_writes_utf8(tmp_path):
    out_file = tmp_path / "results.sarif"
    result = runner.invoke(app, ["scan", ".", "--format", "sarif", "-o", str(out_file)])
    assert result.exit_code == 0
    assert out_file.exists()
    raw = out_file.read_bytes()
    text = raw.decode("utf-8")  # raises if not valid UTF-8
    data = json.loads(text)
    assert data["version"] == "2.1.0"


def test_scan_sarif_deterministic_across_invocations():
    first = runner.invoke(app, ["scan", ".", "--format", "sarif"])
    second = runner.invoke(app, ["scan", ".", "--format", "sarif"])
    assert first.exit_code == 0
    assert second.exit_code == 0
    first_ids = [r["properties"]["sentinelai"]["findingId"] for r in json.loads(first.output)["runs"][0]["results"]]
    second_ids = [r["properties"]["sentinelai"]["findingId"] for r in json.loads(second.output)["runs"][0]["results"]]
    assert first_ids == second_ids


def test_scan_invalid_format_message_mentions_sarif():
    result = runner.invoke(app, ["scan", ".", "--format", "xml"])
    assert result.exit_code == ExitCode.INVALID_INPUT
    assert "sarif" in result.output.lower()


# --- Milestone 9: `sentinelai report` -----------------------------------------------------------------


def _scan_to_json_file(tmp_path, filename="scan-result.json"):
    """Run the real `scan` command once and save its JSON report to a file, for `report` tests to load."""
    out_file = tmp_path / filename
    result = runner.invoke(app, ["scan", ".", "--format", "json", "-o", str(out_file)])
    assert result.exit_code == 0
    return out_file


def test_report_json_to_json(tmp_path):
    scan_file = _scan_to_json_file(tmp_path)
    result = runner.invoke(app, ["report", str(scan_file), "--format", "json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert len(data["findings"]["scanner"]) == 10


def test_report_json_to_markdown(tmp_path):
    scan_file = _scan_to_json_file(tmp_path)
    result = runner.invoke(app, ["report", str(scan_file), "--format", "markdown"])
    assert result.exit_code == 0
    assert result.output.startswith("# SentinelAI Security Report")


def test_report_json_to_html(tmp_path):
    scan_file = _scan_to_json_file(tmp_path)
    result = runner.invoke(app, ["report", str(scan_file), "--format", "html"])
    assert result.exit_code == 0
    assert result.output.strip().startswith("<!DOCTYPE html>")


def test_report_json_to_sarif(tmp_path):
    scan_file = _scan_to_json_file(tmp_path)
    result = runner.invoke(app, ["report", str(scan_file), "--format", "sarif"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["version"] == "2.1.0"
    assert len(data["runs"][0]["results"]) == 10


def test_report_default_format_is_json(tmp_path):
    scan_file = _scan_to_json_file(tmp_path)
    result = runner.invoke(app, ["report", str(scan_file)])
    assert result.exit_code == 0
    json.loads(result.output)  # parses as JSON with no --format given


def test_report_statistics_consistent_with_original_scan(tmp_path):
    scan_file = _scan_to_json_file(tmp_path)
    original = json.loads(scan_file.read_text(encoding="utf-8"))
    result = runner.invoke(app, ["report", str(scan_file), "--format", "json"])
    assert result.exit_code == 0
    reported = json.loads(result.output)
    assert reported["statistics"]["total_findings"] == original["statistics"]["total_findings"]
    assert reported["statistics"]["critical_findings"] == original["statistics"]["critical_findings"]


def test_report_output_file_writes_utf8(tmp_path):
    scan_file = _scan_to_json_file(tmp_path)
    out_file = tmp_path / "report.md"
    result = runner.invoke(app, ["report", str(scan_file), "--format", "markdown", "-o", str(out_file)])
    assert result.exit_code == 0
    text = out_file.read_bytes().decode("utf-8")
    assert text.startswith("# SentinelAI Security Report")


def test_report_invalid_output_path_errors_cleanly(tmp_path):
    scan_file = _scan_to_json_file(tmp_path)
    result = runner.invoke(
        app, ["report", str(scan_file), "--format", "json", "-o", str(tmp_path / "no_such_dir" / "report.json")]
    )
    assert result.exit_code == ExitCode.INVALID_INPUT
    assert "Error" in result.output
    assert "Traceback" not in result.output


def test_report_unsupported_format_rejected():
    result = runner.invoke(app, ["report", "anything.json", "--format", "xml"])
    assert result.exit_code == ExitCode.INVALID_INPUT
    assert "format" in result.output.lower()


def test_report_rejects_terminal_format():
    # `report` renders persisted formats only - "terminal" is scan-only.
    result = runner.invoke(app, ["report", "anything.json", "--format", "terminal"])
    assert result.exit_code == ExitCode.INVALID_INPUT


def test_report_nonexistent_input_file_errors_cleanly():
    result = runner.invoke(app, ["report", "does-not-exist.json"])
    assert result.exit_code == ExitCode.INVALID_INPUT
    assert "Error" in result.output
    assert "Traceback" not in result.output


def test_report_malformed_json_input_errors_cleanly(tmp_path):
    bad_file = tmp_path / "bad.json"
    bad_file.write_text("{not valid json", encoding="utf-8")
    result = runner.invoke(app, ["report", str(bad_file)])
    assert result.exit_code == ExitCode.INVALID_INPUT
    assert "Error" in result.output
    assert "Traceback" not in result.output


def test_report_does_not_contaminate_stdout_with_rich_output(tmp_path):
    scan_file = _scan_to_json_file(tmp_path)
    result = runner.invoke(app, ["report", str(scan_file), "--format", "json"])
    assert result.exit_code == 0
    assert "\x1b[" not in result.output
    json.loads(result.output)


def test_report_deterministic_output(tmp_path):
    scan_file = _scan_to_json_file(tmp_path)
    first = runner.invoke(app, ["report", str(scan_file), "--format", "json"])
    second = runner.invoke(app, ["report", str(scan_file), "--format", "json"])
    assert first.output == second.output


def test_report_does_not_call_the_findings_provider(tmp_path, monkeypatch):
    scan_file = _scan_to_json_file(tmp_path)
    calls = _spy_get_scan_result(monkeypatch)
    result = runner.invoke(app, ["report", str(scan_file), "--format", "json"])
    assert result.exit_code == 0
    assert len(calls) == 0  # `report` must never touch the FindingsProvider


def test_scan_still_calls_the_provider_exactly_once(monkeypatch):
    # Companion to the above: confirms the spy itself works and that `scan`
    # (unlike `report`) does call the provider.
    calls = _spy_get_scan_result(monkeypatch)
    result = runner.invoke(app, ["scan", ".", "--format", "json"])
    assert result.exit_code == 0
    assert len(calls) == 1


# --- Milestone 10: exit codes, --fail-on, error routing -----------------------------------------------------------------


def _break_provider(monkeypatch, message: str = "simulated backend outage"):
    """Make MockFindingsProvider.get_scan_result raise, without touching production code."""

    def boom(self, repo_path, mode=ScanMode.STANDARD, tier=ScannerTier.CORE):
        raise RuntimeError(message)

    monkeypatch.setattr(MockFindingsProvider, "get_scan_result", boom)


class TestExitCodes:
    """
    CLI-level --fail-on tests, exercised against the real mock dataset
    (4 critical, 4 high, 2 medium, 0 low). Because --severity only ever
    *removes* lower severities (never the highest ones present), the
    critical/high findings already in the mock data can't be filtered
    away - so a case like "--fail-on critical succeeds because no
    critical findings are visible" can't be constructed against this
    fixed dataset via the CLI. That exact boundary (fail-on vs. each
    severity tier, including the "nothing at/above threshold" case) is
    covered precisely in tests/test_fail_on_gating.py against synthetic
    ScanStatistics instead.
    """

    def test_successful_scan_exits_success(self):
        result = runner.invoke(app, ["scan", "."])
        assert result.exit_code == ExitCode.SUCCESS

    def test_findings_present_without_fail_on_still_succeeds(self):
        # Mock data always has findings; default --fail-on is "none".
        result = runner.invoke(app, ["scan", ".", "--format", "json"])
        assert result.exit_code == ExitCode.SUCCESS

    def test_fail_on_high_with_high_findings_fails(self):
        result = runner.invoke(app, ["scan", ".", "--fail-on", "high"])
        assert result.exit_code == ExitCode.SECURITY_FINDINGS

    def test_fail_on_high_with_critical_findings_fails(self):
        result = runner.invoke(app, ["scan", ".", "--severity", "critical", "--fail-on", "high"])
        assert result.exit_code == ExitCode.SECURITY_FINDINGS

    def test_fail_on_medium_fails_when_medium_findings_present(self):
        result = runner.invoke(app, ["scan", ".", "--fail-on", "medium"])
        assert result.exit_code == ExitCode.SECURITY_FINDINGS

    def test_fail_on_low_fails_when_any_finding_present(self):
        result = runner.invoke(app, ["scan", ".", "--fail-on", "low"])
        assert result.exit_code == ExitCode.SECURITY_FINDINGS

    def test_fail_on_critical_fails_since_mock_data_has_critical_findings(self):
        result = runner.invoke(app, ["scan", ".", "--fail-on", "critical"])
        assert result.exit_code == ExitCode.SECURITY_FINDINGS

    def test_fail_on_none_never_fails_regardless_of_findings(self):
        result = runner.invoke(app, ["scan", ".", "--fail-on", "none"])
        assert result.exit_code == ExitCode.SUCCESS

    def test_invalid_fail_on_value_is_invalid_input(self):
        result = runner.invoke(app, ["scan", ".", "--fail-on", "banana"])
        assert result.exit_code == ExitCode.INVALID_INPUT
        assert "fail-on" in result.output.lower()

    def test_fail_on_applies_to_severity_filtered_result(self):
        # --severity low includes everything (low is the lowest tier), so
        # this is equivalent to no filtering - included mainly to confirm
        # --fail-on and --severity compose without interfering with each
        # other's semantics.
        result = runner.invoke(app, ["scan", ".", "--severity", "low", "--fail-on", "critical"])
        assert result.exit_code == ExitCode.SECURITY_FINDINGS


class TestErrorCases:
    def test_nonexistent_repository_is_invalid_input(self):
        result = runner.invoke(app, ["scan", "/definitely/does/not/exist/xyz"])
        assert result.exit_code == ExitCode.INVALID_INPUT

    def test_provider_failure_is_provider_error_not_security_findings(self, monkeypatch):
        _break_provider(monkeypatch)
        result = runner.invoke(app, ["scan", ".", "--fail-on", "critical"])
        assert result.exit_code == ExitCode.PROVIDER_ERROR

    def test_provider_failure_message_is_clean_no_traceback_by_default(self, monkeypatch):
        _break_provider(monkeypatch, "simulated backend outage")
        result = runner.invoke(app, ["scan", "."])
        assert "simulated backend outage" in result.stderr
        assert "Traceback" not in result.output

    def test_malformed_report_input_is_invalid_input(self, tmp_path):
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("{not valid json", encoding="utf-8")
        result = runner.invoke(app, ["report", str(bad_file)])
        assert result.exit_code == ExitCode.INVALID_INPUT

    def test_unsupported_schema_version_is_invalid_input(self, tmp_path):
        scan_file = _scan_to_json_file(tmp_path)
        data = json.loads(scan_file.read_text(encoding="utf-8"))
        data["schema_version"] = "9.0"
        scan_file.write_text(json.dumps(data), encoding="utf-8")
        result = runner.invoke(app, ["report", str(scan_file)])
        assert result.exit_code == ExitCode.INVALID_INPUT

    def test_output_write_failure_is_invalid_input(self, tmp_path):
        result = runner.invoke(
            app, ["scan", ".", "--format", "json", "-o", str(tmp_path / "no_such_dir" / "report.json")]
        )
        assert result.exit_code == ExitCode.INVALID_INPUT

    def test_unexpected_internal_error_during_rendering_is_internal_error(self, monkeypatch):
        import sentinelai.main as main_module

        def boom(fmt, result, stats):
            raise RuntimeError("simulated rendering bug")

        monkeypatch.setattr(main_module, "_render_structured_content", boom)
        result = runner.invoke(app, ["scan", ".", "--format", "json"])
        assert result.exit_code == ExitCode.INTERNAL_ERROR
        assert "simulated rendering bug" in result.stderr


class TestOutputStreams:
    def test_structured_stdout_stays_pure_when_write_fails(self, tmp_path):
        result = runner.invoke(
            app, ["scan", ".", "--format", "json", "-o", str(tmp_path / "no_such_dir" / "report.json")]
        )
        assert result.exit_code == ExitCode.INVALID_INPUT
        assert result.stdout == ""

    def test_errors_go_to_stderr_not_stdout(self):
        result = runner.invoke(app, ["scan", "/definitely/does/not/exist/xyz"])
        assert result.stdout == ""
        assert "Error" in result.stderr

    def test_provider_error_leaves_stdout_empty_for_json_format(self, monkeypatch):
        _break_provider(monkeypatch)
        result = runner.invoke(app, ["scan", ".", "--format", "json"])
        assert result.stdout == ""
        assert "Error" in result.stderr

    def test_terminal_mode_output_readable_and_on_stdout(self):
        result = runner.invoke(app, ["scan", "."])
        assert result.exit_code == 0
        assert "SentinelAI Security Scan" in result.stdout
        assert result.stderr == ""

    def test_debug_flag_shows_traceback_on_stderr(self, tmp_path):
        result = runner.invoke(
            app,
            ["--debug", "scan", ".", "--format", "json", "-o", str(tmp_path / "no_such_dir" / "report.json")],
        )
        assert result.exit_code == ExitCode.INVALID_INPUT
        assert "Traceback" in result.stderr

    def test_without_debug_flag_no_traceback_shown(self, tmp_path):
        result = runner.invoke(
            app, ["scan", ".", "--format", "json", "-o", str(tmp_path / "no_such_dir" / "report.json")]
        )
        assert result.exit_code == ExitCode.INVALID_INPUT
        assert "Traceback" not in result.stderr


class TestBackwardCompatibility:
    def test_severity_still_filters_findings_not_exit_code(self):
        result = runner.invoke(app, ["scan", ".", "--severity", "critical", "--format", "json"])
        assert result.exit_code == ExitCode.SUCCESS  # no --fail-on given
        data = json.loads(result.output)
        assert all(f["severity"] == "critical" for f in data["findings"]["scanner"])

    def test_report_command_success_path_unchanged(self, tmp_path):
        scan_file = _scan_to_json_file(tmp_path)
        result = runner.invoke(app, ["report", str(scan_file), "--format", "json"])
        assert result.exit_code == ExitCode.SUCCESS
        assert len(json.loads(result.output)["findings"]["scanner"]) == 10

    def test_json_still_valid(self):
        result = runner.invoke(app, ["scan", ".", "--format", "json"])
        json.loads(result.output)

    def test_markdown_still_valid(self):
        result = runner.invoke(app, ["scan", ".", "--format", "markdown"])
        assert result.output.startswith("# SentinelAI Security Report")

    def test_html_still_valid(self):
        result = runner.invoke(app, ["scan", ".", "--format", "html"])
        assert result.output.strip().startswith("<!DOCTYPE html>")

    def test_sarif_still_valid(self):
        result = runner.invoke(app, ["scan", ".", "--format", "sarif"])
        data = json.loads(result.output)
        assert data["version"] == "2.1.0"
