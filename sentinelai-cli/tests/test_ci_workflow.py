"""
Static validation of the GitHub Actions CI workflow (.github/workflows/ci.yml).

This is deliberately lightweight: parse the YAML with PyYAML and assert
on its structure and the shell commands it runs, rather than building a
GitHub-Actions-in-a-box test harness. It cannot execute the workflow
(that requires an actual GitHub Actions runner); it can only catch
"this file is malformed" or "this file no longer does what the README
claims" before either reaches CI.

Note: PyYAML's default (YAML 1.1) loader treats a bare `on:` key as the
boolean True, not the string "on" - a well-known GitHub Actions/YAML
gotcha. The workflow file quotes it as "on": to sidestep the ambiguity
for every parser, so these tests can use data["on"] directly.
"""
from pathlib import Path

import yaml

WORKFLOW_PATH = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "ci.yml"


def _load_workflow() -> dict:
    with open(WORKFLOW_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _all_run_commands(workflow: dict) -> str:
    """Concatenate every step's `run:` shell script into one string, for substring checks."""
    commands = []
    for job in workflow["jobs"].values():
        for step in job.get("steps", []):
            if "run" in step:
                commands.append(step["run"])
    return "\n".join(commands)


def _all_step_names(workflow: dict) -> list:
    names = []
    for job in workflow["jobs"].values():
        for step in job.get("steps", []):
            if "name" in step:
                names.append(step["name"])
    return names


# --- existence / parsing -----------------------------------------------------------------


def test_workflow_file_exists():
    assert WORKFLOW_PATH.exists()


def test_workflow_yaml_parses_correctly():
    workflow = _load_workflow()
    assert isinstance(workflow, dict)


def test_workflow_triggers_on_push_and_pull_request():
    workflow = _load_workflow()
    assert "push" in workflow["on"]
    assert "pull_request" in workflow["on"]


def test_workflow_has_exactly_one_job():
    workflow = _load_workflow()
    assert len(workflow["jobs"]) == 1


def test_workflow_declares_minimal_permissions():
    workflow = _load_workflow()
    assert workflow.get("permissions") == {"contents": "read"}


# --- stages -----------------------------------------------------------------


def test_workflow_has_checkout_step():
    workflow = _load_workflow()
    uses = [s.get("uses", "") for job in workflow["jobs"].values() for s in job["steps"]]
    assert any(u.startswith("actions/checkout@") for u in uses)


def test_workflow_has_python_setup_step():
    workflow = _load_workflow()
    uses = [s.get("uses", "") for job in workflow["jobs"].values() for s in job["steps"]]
    assert any(u.startswith("actions/setup-python@") for u in uses)


def test_workflow_installs_from_pyproject_not_a_duplicated_list():
    commands = _all_run_commands(_load_workflow())
    assert "pip install -e" in commands
    assert ".[dev]" in commands
    # No hardcoded package list (e.g. "pip install typer rich pydantic") -
    # dependencies come from pyproject.toml, not duplicated here.
    assert "pip install typer" not in commands


def test_workflow_runs_pytest_before_the_scan():
    workflow = _load_workflow()
    names = _all_step_names(workflow)
    commands = _all_run_commands(workflow)
    assert "pytest -v" in commands
    pytest_index = next(i for i, n in enumerate(names) if "test" in n.lower())
    scan_index = next(i for i, n in enumerate(names) if "scan" in n.lower() and "sentinelai" in n.lower())
    assert pytest_index < scan_index


def test_workflow_working_directory_is_the_cli_package():
    workflow = _load_workflow()
    job = next(iter(workflow["jobs"].values()))
    assert job["defaults"]["run"]["working-directory"] == "sentinelai-cli"


# --- SentinelAI commands -----------------------------------------------------------------


def test_workflow_scan_command_present_with_fail_on_and_json_output():
    commands = _all_run_commands(_load_workflow())
    assert "sentinelai scan ." in commands
    assert "--format json" in commands
    assert "--output scan-result.json" in commands
    assert "--fail-on high" in commands


def test_workflow_generates_sarif_from_saved_result_not_a_second_scan():
    commands = _all_run_commands(_load_workflow())
    assert "sentinelai report scan-result.json --format sarif" in commands
    assert "--output sentinelai-results.sarif" in commands


def test_workflow_generates_html_from_saved_result_not_a_second_scan():
    commands = _all_run_commands(_load_workflow())
    assert "sentinelai report scan-result.json --format html" in commands
    assert "--output sentinelai-report.html" in commands


def test_workflow_does_not_scan_twice():
    commands = _all_run_commands(_load_workflow())
    assert commands.count("sentinelai scan ") == 1


# --- exit-code handling -----------------------------------------------------------------


def test_workflow_captures_scan_exit_code_explicitly():
    commands = _all_run_commands(_load_workflow())
    assert "GITHUB_OUTPUT" in commands
    assert "exit_code=$?" in commands


def test_workflow_treats_exit_code_1_as_security_failure_not_tool_error():
    commands = _all_run_commands(_load_workflow())
    assert '"$code" = "1"' in commands
    assert "SECURITY GATE FAILED" in commands


def test_workflow_treats_exit_codes_2_3_4_as_tool_failures():
    commands = _all_run_commands(_load_workflow())
    for code, label in (("2", "INVALID INPUT"), ("3", "PROVIDER ERROR"), ("4", "INTERNAL ERROR")):
        assert f'"$code" = "{code}"' in commands or "internal error" in commands.lower()
        assert label in commands or "INTERNAL ERROR" in commands


def test_workflow_does_not_silently_ignore_tool_failures():
    # Every non-zero, non-1 branch of the final evaluation step must exit 1
    # (fail the job) - a tool error must not be swallowed as a pass.
    commands = _all_run_commands(_load_workflow())
    # crude but effective: the evaluation step's run block should contain
    # multiple "exit 1" occurrences (one per failing branch), not zero.
    assert commands.count("exit 1") >= 3


def test_report_generation_is_skipped_on_tool_failure():
    workflow = _load_workflow()
    job = next(iter(workflow["jobs"].values()))
    sarif_step = next(s for s in job["steps"] if "SARIF" in s.get("name", ""))
    html_step = next(s for s in job["steps"] if "HTML" in s.get("name", ""))
    assert "exit_code" in sarif_step.get("if", "")
    assert "exit_code" in html_step.get("if", "")


# --- artifacts -----------------------------------------------------------------


def test_workflow_uploads_artifacts_step_exists():
    workflow = _load_workflow()
    uses = [s.get("uses", "") for job in workflow["jobs"].values() for s in job["steps"]]
    assert any(u.startswith("actions/upload-artifact@") for u in uses)


def test_workflow_artifact_upload_runs_always():
    workflow = _load_workflow()
    job = next(iter(workflow["jobs"].values()))
    upload_step = next(s for s in job["steps"] if s.get("uses", "").startswith("actions/upload-artifact@"))
    assert upload_step.get("if") == "always()"


def test_workflow_artifact_includes_scan_result_json():
    workflow = _load_workflow()
    job = next(iter(workflow["jobs"].values()))
    upload_step = next(s for s in job["steps"] if s.get("uses", "").startswith("actions/upload-artifact@"))
    assert "scan-result.json" in upload_step["with"]["path"]


def test_workflow_artifact_includes_sarif():
    workflow = _load_workflow()
    job = next(iter(workflow["jobs"].values()))
    upload_step = next(s for s in job["steps"] if s.get("uses", "").startswith("actions/upload-artifact@"))
    assert "sentinelai-results.sarif" in upload_step["with"]["path"]


def test_workflow_artifact_includes_html():
    workflow = _load_workflow()
    job = next(iter(workflow["jobs"].values()))
    upload_step = next(s for s in job["steps"] if s.get("uses", "").startswith("actions/upload-artifact@"))
    assert "sentinelai-report.html" in upload_step["with"]["path"]


def test_workflow_artifact_has_a_clear_name():
    workflow = _load_workflow()
    job = next(iter(workflow["jobs"].values()))
    upload_step = next(s for s in job["steps"] if s.get("uses", "").startswith("actions/upload-artifact@"))
    assert upload_step["with"]["name"] == "sentinelai-security-reports"


# --- no API keys / secrets -----------------------------------------------------------------


def test_workflow_requires_no_secrets_or_api_keys():
    raw_text = WORKFLOW_PATH.read_text(encoding="utf-8")
    assert "secrets." not in raw_text
    forbidden_terms = [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "HUGGINGFACE",
        "QDRANT",
    ]
    upper_text = raw_text.upper()
    for term in forbidden_terms:
        assert term not in upper_text


def test_workflow_env_block_is_absent_or_empty():
    workflow = _load_workflow()
    # No global or job-level `env:` smuggling in credentials.
    assert "env" not in workflow
    for job in workflow["jobs"].values():
        assert "env" not in job
