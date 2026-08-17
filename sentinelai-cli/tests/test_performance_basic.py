"""
Basic performance sanity checks - regression guards, not benchmarks.

These exist to catch a gross regression (a hang, an accidental O(n^2), a
scanner invoked twice) before it reaches a release, not to produce the
numbers reported in the paper. The paper's numbers come from
sentinelai-manual-test/README.md's own repeated, hand-timed runs; the
thresholds here are set well above the machine those numbers were
measured on specifically so this file stays robust on a slower or
busier CI runner instead of flaking. Do not tighten these thresholds to
match the paper's reported figures.

Two tests here (test_scanner_only_scan_completes_within_budget and
test_ai_enriched_scan_completes_within_budget) invoke real
Semgrep/Bandit/GitLeaks binaries - every other scanner test in this
project (test_scanners_semgrep.py etc.) deliberately mocks
subprocess.run, per this project's established "never invoke a real
scanner in the unit suite" convention. These two break that convention
on purpose, because a performance sanity check that doesn't run the
real tools isn't measuring anything real.

That means they must never run as part of the default `pytest`/`pytest
-q` invocation - the project's whole test suite is fast and
tool-free precisely because nothing else does this. They are marked
`@pytest.mark.benchmark`, and pyproject.toml's `[tool.pytest.ini_options]`
sets `addopts = "-m 'not benchmark'"`, so plain `pytest`/`pytest -q`
(including CI's `pytest -v` step) excludes them automatically - no
per-command opt-out flag needed anywhere. Run them explicitly with:

    pytest -m benchmark -v

The `@pytest.mark.skipif` guards on both are a second, independent line
of defense for whoever runs that command directly: skipped (not failed)
if the three scanner executables aren't all on PATH - exactly
SentinelAI's own CI's situation if the marker filter were ever bypassed,
since its `pytest -v` step runs *before* the "Install scanner tools"
step later in the same job (see ../../.github/workflows/ci.yml) - and
skipped if the benchmark repo itself isn't present.

sentinelai-manual-test/ is a permanent fixture living at the repository
root (a sibling of sentinelai-cli/, not inside it) rather than under
tests/fixtures/: it needs to be reachable by a plain `sentinelai scan
sentinelai-manual-test` from the repo root for the paper/demo commands
in its own README to work verbatim, and it must stay outside
sentinelai-cli/ specifically so SentinelAI's CI self-scan of
`sentinelai-cli/.` (see ci.yml) never scans it - it is deliberately full
of high-severity findings, and self-scanning it would fail that job's
--fail-on gate for a reason that has nothing to do with SentinelAI's own
code quality.

Invoked via `python -m sentinelai.main`, not the installed `sentinelai`
console script: this works whether or not the package was installed
with its entry point registered on PATH, and guarantees the same Python
interpreter (sys.executable) that's running pytest is the one actually
running the scan - both already-documented, equally-valid ways to run
SentinelAI (see sentinelai-cli/README.md's "Install" section).
"""
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from sentinelai.providers import MockFindingsProvider
from sentinelai.reporting import to_json
from sentinelai.statistics import calculate_statistics

BENCHMARK_REPO = Path(__file__).resolve().parents[2] / "sentinelai-manual-test"

# Generous on purpose - see module docstring. Semgrep's own process-startup
# and rule-load overhead alone was observed to vary 3.65s-7.93s across
# back-to-back identical runs on a fast local machine; this budget leaves
# comfortable headroom for a slower or colder CI runner without being
# meaningless as a regression guard (it would still catch a hang or a
# multi-scanner-invoked-twice bug).
SCANNER_ONLY_BUDGET_SECONDS = 45.0

# Reporting is pure in-process serialization of already-computed data - no
# subprocess, no I/O beyond stdout. Milliseconds in practice; generous here
# only to stay robust on a slow/loaded machine, not because it's expected
# to need it.
REPORT_GENERATION_BUDGET_SECONDS = 2.0

# AI enrichment makes one sequential LLM call per finding (see
# sentinelai/ai/pipeline.py) with no concurrency, so wall time scales
# roughly linearly with finding count and with whichever local model is
# configured - there is no single number that's safe for every machine.
# This budget is a coarse upper bound for local, opt-in use only; it is
# not a claim about expected latency (that limitation is documented in
# the paper directly).
AI_ENRICHED_BUDGET_SECONDS = 600.0

_REQUIRED_SCANNERS = ("semgrep", "bandit", "gitleaks")
_missing_scanners = [tool for tool in _REQUIRED_SCANNERS if shutil.which(tool) is None]


def _run_scan(*extra_args: str) -> "subprocess.CompletedProcess[str]":
    return subprocess.run(
        [sys.executable, "-m", "sentinelai.main", "scan", str(BENCHMARK_REPO), *extra_args],
        capture_output=True,
        text=True,
        timeout=SCANNER_ONLY_BUDGET_SECONDS + 30,  # hard ceiling well above the soft budget asserted below
    )


@pytest.mark.benchmark
@pytest.mark.skipif(not BENCHMARK_REPO.is_dir(), reason=f"benchmark repo not found at {BENCHMARK_REPO}")
@pytest.mark.skipif(
    _missing_scanners, reason=f"required scanner(s) not on PATH: {', '.join(_missing_scanners)}"
)
def test_scanner_only_scan_completes_within_budget(tmp_path):
    """A full scanner-only scan of the benchmark repo finishes well inside a generous ceiling.

    Also asserts the finding count is exactly the number verified by hand
    in sentinelai-manual-test/README.md (17: 10 Bandit + 6 Semgrep + 1
    GitLeaks) - static analysis is deterministic, so any drift here means
    a scanner wrapper, not the environment, changed behavior.
    """
    output_path = tmp_path / "scan-result.json"

    started = time.perf_counter()
    result = _run_scan("--format", "json", "--output", str(output_path))
    elapsed = time.perf_counter() - started

    assert result.returncode == 0, result.stderr
    assert elapsed < SCANNER_ONLY_BUDGET_SECONDS, (
        f"scanner-only scan took {elapsed:.1f}s, budget is {SCANNER_ONLY_BUDGET_SECONDS}s"
    )

    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert len(data["findings"]["scanner"]) == 17
    assert data["findings"]["ai_enriched"] == []


def test_json_report_generation_is_fast():
    """Serializing an already-scanned result to JSON is cheap, independent of scanner cost.

    Uses MockFindingsProvider's bundled fixture (10 findings) rather than
    a real scan - report generation cost depends only on finding count and
    the reporting layer's own code, never on which provider produced the
    findings, so there is nothing this test would learn from a real scan
    that it can't learn here in milliseconds instead of seconds.
    """
    result = MockFindingsProvider().get_scan_result("/tmp/some-repo")

    started = time.perf_counter()
    stats = calculate_statistics(result)
    content = to_json(result, stats)
    elapsed = time.perf_counter() - started

    assert elapsed < REPORT_GENERATION_BUDGET_SECONDS
    assert json.loads(content)["findings"]["scanner"]  # non-empty, genuinely serialized


@pytest.mark.benchmark
@pytest.mark.skipif(not BENCHMARK_REPO.is_dir(), reason=f"benchmark repo not found at {BENCHMARK_REPO}")
@pytest.mark.skipif(_missing_scanners, reason=f"required scanner(s) not on PATH: {', '.join(_missing_scanners)}")
@pytest.mark.skipif(
    True,
    reason=(
        "AI-enriched performance is opt-in and unmeasured by default: enable locally by removing "
        "this skipif once SENTINELAI_AI_LLM_MODEL and SENTINELAI_AI_EMBEDDING_MODEL are exported "
        "to real, working Ollama models (see sentinelai-cli/README.md). Left permanently skipped "
        "here, rather than gated on the environment variables alone, so this test can never "
        "silently start running - and taking minutes - even under `pytest -m benchmark -v` - just "
        "because someone happens to have AI configured for other work."
    ),
)
def test_ai_enriched_scan_completes_within_budget(tmp_path):
    """Smoke check only, for local opt-in use - see the skip reason for why this never runs by default."""
    output_path = tmp_path / "scan-result-ai.json"

    started = time.perf_counter()
    result = subprocess.run(
        [sys.executable, "-m", "sentinelai.main", "scan", str(BENCHMARK_REPO), "--format", "json", "--output", str(output_path)],
        capture_output=True,
        text=True,
        timeout=AI_ENRICHED_BUDGET_SECONDS + 30,
    )
    elapsed = time.perf_counter() - started

    assert result.returncode == 0, result.stderr
    assert elapsed < AI_ENRICHED_BUDGET_SECONDS

    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert len(data["findings"]["ai_enriched"]) == len(data["findings"]["scanner"])
