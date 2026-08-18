"""Where OSV-Scanner looks for dependency manifests, proven against the real tool.

The wrapper's unit tests (tests/test_scanners_osv.py) patch `subprocess.run`,
so they can prove the command *contains* `--recursive` but not that the flag
does what the fix depends on. These tests invoke the installed `osv-scanner`
and the real CLI, because the defect being guarded against was a discovery
behaviour, not a parsing one:

    sentinelai scan sentinelai-manual-test --extended   ->  worked
    sentinelai scan . --extended                        ->  exit 3

`osv-scanner scan source <dir>` inspects only the top level of <dir>. Every
manifest in this repository is nested one level down, so scanning the
repository root - the documented Docker invocation, and the natural thing to
run - exited 128 "No package sources found", which the orchestrator correctly
treats as a scanner failure and surfaces as PROVIDER_ERROR.

Marked `benchmark` for the same reason as tests/test_performance_basic.py:
these shell out to real tools and reach the network (OSV resolves package
metadata through deps.dev), so they are excluded from the default run and
from CI, which deliberately installs only the core three scanners. The
`skipif` guards are independent of the marker so the file is also safe to run
directly on a machine without the dependency scanners.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_DEPENDENCY_SCANNERS = ("trivy", "osv-scanner")
_missing = [tool for tool in _DEPENDENCY_SCANNERS if shutil.which(tool) is None]

requires_osv = pytest.mark.skipif(
    shutil.which("osv-scanner") is None, reason="osv-scanner not on PATH"
)
requires_extended = pytest.mark.skipif(
    _missing, reason=f"dependency scanner(s) not on PATH: {', '.join(_missing)}"
)

# A manifest with known advisories, so a successful scan is distinguishable
# from a scan that merely ran. Pinned deliberately old; these are the same
# packages the project's benchmark fixture declares.
_MANIFEST = "flask==0.12.2\nrequests==2.19.1\npyyaml==5.1\n"


def _write_manifest(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / "requirements.txt"
    manifest.write_text(_MANIFEST, encoding="utf-8")
    return manifest


def _osv(target: Path) -> subprocess.CompletedProcess:
    """Invoke OSV exactly as the wrapper does - the flags under test are the point."""
    return subprocess.run(
        ["osv-scanner", "scan", "source", "--format", "json", "--recursive", str(target)],
        capture_output=True,
        text=True,
        timeout=300,
    )


def _groups(completed: subprocess.CompletedProcess) -> int:
    data = json.loads(completed.stdout)
    return sum(
        len(package.get("groups") or [])
        for result in (data.get("results") or [])
        for package in (result.get("packages") or [])
    )


def _sources(completed: subprocess.CompletedProcess) -> set:
    data = json.loads(completed.stdout)
    return {(result.get("source") or {}).get("path", "") for result in (data.get("results") or [])}


# --- discovery ------------------------------------------------------------------------------------------------------


@pytest.mark.benchmark
@requires_osv
def test_top_level_manifest_is_found(tmp_path):
    """The case that always worked. Guards against a fix that trades one shape for the other."""
    _write_manifest(tmp_path)

    completed = _osv(tmp_path)

    assert completed.returncode == 1, completed.stderr
    assert _groups(completed) > 0


@pytest.mark.benchmark
@requires_osv
def test_nested_manifest_is_found_when_scanning_the_root(tmp_path):
    """The defect. Without --recursive this exits 128 and the whole scan fails closed."""
    _write_manifest(tmp_path / "packages" / "service")

    completed = _osv(tmp_path)

    assert completed.returncode == 1, f"nested manifest not discovered: {completed.stderr}"
    assert _groups(completed) > 0
    assert any(source.endswith("requirements.txt") for source in _sources(completed))


@pytest.mark.benchmark
@requires_osv
def test_every_nested_manifest_is_found_not_merely_the_first(tmp_path):
    """This repository has two (sentinelai-cli/ and sentinelai-manual-test/)."""
    _write_manifest(tmp_path / "alpha")
    _write_manifest(tmp_path / "beta" / "deeper")

    sources = _sources(_osv(tmp_path))

    assert sum(1 for source in sources if source.endswith("requirements.txt")) == 2, sources


@pytest.mark.benchmark
@requires_osv
def test_no_manifest_anywhere_still_fails_closed(tmp_path):
    """--recursive widens the search; it must not make an empty search succeed.

    Exit 128 with empty stdout is what the wrapper converts into
    ScannerExecutionError, and the orchestrator into PROVIDER_ERROR. If this
    ever starts returning 0, a repository with no dependency manifest would be
    silently reported as having no dependency vulnerabilities.
    """
    (tmp_path / "src" / "deep").mkdir(parents=True)
    (tmp_path / "src" / "deep" / "app.py").write_text("x = 1\n", encoding="utf-8")

    completed = _osv(tmp_path)

    assert completed.returncode == 128
    assert "No package sources found" in completed.stderr
    assert completed.stdout.strip() == ""


# --- the fix, through the CLI ----------------------------------------------------------------------------------------


@pytest.mark.benchmark
@requires_extended
def test_extended_scan_of_a_root_with_a_nested_manifest_succeeds(tmp_path):
    """End to end: the exact shape that produced `exit 3` before the fix.

    Asserts the dependency findings are present *and* correctly represented in
    the normal scan output - right tier, right scanner name, manifest-scoped
    path, and no synthesized line numbers.
    """
    _write_manifest(tmp_path / "service")
    (tmp_path / "service" / "app.py").write_text("import flask\n", encoding="utf-8")
    report = tmp_path / "report.json"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "sentinelai.main",
            "scan",
            str(tmp_path),
            "--extended",
            "--format",
            "json",
            "--output",
            str(report),
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )

    assert completed.returncode == 0, f"exit {completed.returncode}: {completed.stdout}{completed.stderr}"

    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["scan"]["scanner_tier"] == "extended"

    dependency = [f for f in data["findings"]["scanner"] if f["scanner"] == "osv-scanner"]
    assert dependency, "extended scan produced no OSV findings for a nested manifest"

    for finding in dependency:
        assert finding["file"].endswith("requirements.txt")
        assert finding["line_start"] is None, "dependency findings must not carry invented line numbers"
        assert finding["category"] == "vulnerable-dependency"

    counts = data["statistics"]["scanner_counts"]
    assert counts.get("osv-scanner", 0) == len(dependency)
