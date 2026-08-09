# SentinelAI CLI

Viraj's piece of SentinelAI: the Typer-based CLI that drives a scan and
renders the results. It depends only on the `FindingsProvider` interface
(see `docs/CONTRACTS.md`), not on any specific data source — today that's
`MockFindingsProvider`, backed by a bundled sample dataset, so the CLI,
its output formats, and the shared contracts can all be built and
demoed without waiting on Nithanth's scanner backend or Tanaya's AI
pipeline.

Two commands, two distinct jobs:

- **`sentinelai scan`** — retrieves a scan result (from a
  `FindingsProvider`) and either shows it live or writes a report.
- **`sentinelai report`** — renders an *existing, already-saved* scan
  result as a report. It never runs a scan again.

## Install

```bash
cd sentinelai-cli
python -m venv .venv
.venv/Scripts/pip install -e ".[dev]"   # macOS/Linux: .venv/bin/pip
```

This registers a `sentinelai` command on your PATH (editable install, so
code changes take effect immediately without reinstalling). The `[dev]`
extra pulls in `pytest` for running the test suite.

If you'd rather not install it as a package, `pip install -r requirements.txt`
and run it as `python -m sentinelai.main scan .` instead.

## Usage

```bash
# Scan the current directory, print a compact terminal table
sentinelai scan .

# Scan a specific path
sentinelai scan /path/to/repo

# Scan modes
sentinelai scan . --quick
sentinelai scan . --full

# Only show high and critical findings
sentinelai scan . --severity high

# Export a full report (terminal view is intentionally summary-only)
sentinelai scan . --format markdown -o report.md
sentinelai scan . --format html --output sentinelai-report.html
sentinelai scan . --format json -o report.json
sentinelai scan . --format sarif --output results.sarif

# Check version
sentinelai version
```

### Flags

| Flag | Values | Notes |
|---|---|---|
| `--quick` | flag | Requests a quick scan (`ScanMode.QUICK`). Mutually exclusive with `--full`. |
| `--full` | flag | Requests a full scan (`ScanMode.FULL`). Mutually exclusive with `--quick`. Default (neither flag) is `ScanMode.STANDARD`. |
| `--severity` / `-s` | `low`, `medium`, `high`, `critical` | Shows findings at **or above** this level. Does **not** affect the exit code - see `--fail-on` |
| `--fail-on` | `low`, `medium`, `high`, `critical`, `none` (default) | Exit `1` (`SECURITY_FINDINGS`) if findings at or above this level are present. Independent of `--severity` - see "Exit codes and CI/CD gating" below |
| `--format` / `-f` | `terminal` (default), `json`, `markdown`, `html`, `sarif` | `terminal` prints a compact table; the others produce a full report. Structured formats keep stdout limited to the report itself (no banner), so they stay pipeable |
| `--output` / `-o` | file path | Writes the report to a file instead of stdout. Ignored (with a warning) for `terminal` format |
| `--debug` | flag (app-level, before the subcommand) | Print full tracebacks on unexpected errors instead of a one-line message |

`sentinelai report <input>` takes the same `--format`/`-f` (`json`,
`markdown`, `html`, `sarif` — no `terminal`, since `report` renders
persisted formats only) and `--output`/`-o` as `scan`, plus a required
`<input>` argument: the path to a JSON report previously written by
`scan --format json`.

## Why the table looks compact

The terminal table intentionally shows only ID / location / severity /
category / scanner / confidence — one line per finding. Full AI
explanations, exploit paths, and remediation text are paragraph-length
and don't belong in a table; they're in the markdown/html/json output
instead. This mirrors how tools like Bandit or Semgrep behave: a fast
compact overview in the terminal, full detail in a report.

Also note: the table is designed for terminals ~100+ columns wide (the
default on basically every modern terminal app). In a genuinely narrow
terminal it may wrap.

## Project structure

```
sentinelai-cli/
├── sentinelai/
│   ├── __init__.py
│   ├── core/                # exit codes, error hierarchy, centralized severity ranking
│   ├── contracts/           # shared Pydantic schema — see docs/CONTRACTS.md
│   ├── providers/           # FindingsProvider interface + MockFindingsProvider
│   ├── statistics/          # calculate_statistics(ScanResult) -> ScanStatistics
│   ├── reporting/           # JSON, Markdown, HTML, SARIF report generators + loader.py + Jinja2 templates/
│   ├── presentation/        # Rich terminal rendering — used only for --format terminal
│   └── main.py              # Typer app — wires provider -> statistics -> presentation/reporting
├── tests/
├── docs/
│   └── CONTRACTS.md         # full contract + provider documentation
├── pyproject.toml
├── requirements.txt
└── README.md
```

**Integration seam:** the CLI only ever talks to the abstract
`FindingsProvider` interface (`_get_provider()` in `main.py`). When
Nithanth's backend is ready, a new provider implementation (e.g. calling
his `/scan` endpoint) replaces `MockFindingsProvider()` there — nothing
in `presentation/`, `reporting/`, or the rest of `main.py` needs to
change. Full details, including how Tanaya's
AI-enriched findings join onto scanner findings, are in
`docs/CONTRACTS.md`.

## JSON report

`--format json` produces the canonical, versioned JSON report — the
format Markdown/HTML/SARIF and CI tooling will eventually build on:

```json
{
  "schema_version": "1.0",
  "repository": { "name": "demo-app", "path": "/path/to/demo-app", "commit_hash": null, "branch": null, "languages": [] },
  "scan": { "timestamp": "2026-08-09T12:00:00Z", "mode": "standard", "duration_seconds": 0.01 },
  "statistics": {
    "total_findings": 10,
    "critical_findings": 4,
    "high_findings": 4,
    "medium_findings": 2,
    "low_findings": 0,
    "scanner_counts": { "semgrep": 6, "bandit": 2, "gitleaks": 1, "osv-scanner": 1 },
    "category_counts": { "sql-injection": 1, "secret-exposure": 1, "...": "..." },
    "ai_enrichment_status": "unavailable",
    "matched_ai_findings": 0,
    "confidence": { "enriched_count": 0, "high_count": 0, "medium_count": 0, "low_count": 0, "average_score": null, "min_score": null, "max_score": null },
    "verification_counts": {}
  },
  "findings": {
    "scanner": [ { "finding_id": "SENT-001", "scanner": "gitleaks", "category": "secret-exposure", "severity": "critical", "file": "config/settings.py", "line_start": 14, "rule_id": "gitleaks.aws-access-key", "message": "...", "cwe": "CWE-798" } ],
    "ai_enriched": []
  }
}
```

Findings are sorted by `finding_id` for deterministic output regardless
of provider iteration order. `findings.ai_enriched` correlates to
`findings.scanner` by `finding_id` and is populated once Tanaya's AI
enrichment layer runs — the mock provider always returns it empty, and
`statistics.ai_enrichment_status` says so explicitly (`"unavailable"`)
rather than reporting fabricated confidence numbers. The `statistics`
block is computed once by `sentinelai/statistics/` and reused as-is —
never recalculated inside the report generator, so the terminal summary
and the JSON report can never disagree.

## Markdown report

`--format markdown` produces a report suitable for GitHub PRs, code
review, and developer handoff: an Executive Summary (repository, scan
mode/timestamp/duration, severity table, scanner/category distribution,
AI enrichment status), Scan Information, a compact Findings Overview
table, and a Detailed Findings section with one subsection per finding.

Like JSON, it's built from the same `ScanResult` + `ScanStatistics` and
never recalculates anything — the numbers always match the terminal
summary and the JSON report. Findings are sorted by `finding_id`, the
same ordering JSON uses. A finding with no matching `AIEnrichedFinding`
(correlated by `finding_id`, not list position) shows scanner-level
detail only, plus "_AI enrichment not yet available for this finding._"
— never fabricated AI content.

Raw scanner evidence and AI patch suggestions are rendered inside a
fenced code block whose fence length is chosen dynamically (longer than
any backtick run already in the content), so evidence containing
backticks, headings, or HTML can't corrupt the document structure or be
misinterpreted as Markdown/HTML.

## HTML report

```bash
sentinelai scan . --format html --output report.html
```

`--format html` produces SentinelAI's flagship human-readable report: a
**static**, **self-contained**, **offline-capable** HTML file — open it
directly in any browser, no server, no build step, no network access
required. There is no repository-upload UI, no JavaScript framework, and
no client-side JavaScript at all; every visualization (severity bars,
scanner/category distribution, confidence meters) is plain CSS driven by
values already computed in `ScanStatistics`.

This makes it suitable as a CI artifact, something to attach to a PR or
email a reviewer, or to archive for offline security analysis — the file
is the whole report.

Like JSON and Markdown, it's built from the same `ScanResult` +
`ScanStatistics` (via `sentinelai/reporting/html_report.py` and the
Jinja2 template in `reporting/templates/security_report.html.j2`) and
never recalculates anything. Findings are sorted by `finding_id`, with
stable `id="finding-<id>"` anchors so the Findings Overview table links
directly to each Detailed Findings card. A finding with no matching
`AIEnrichedFinding` shows "AI enrichment unavailable for this finding"
rather than fabricated confidence/exploit/remediation content.

**Security:** all finding/AI content (scanner messages, raw evidence, AI
explanations, patch suggestions) is rendered through Jinja2 with
autoescaping permanently on — nothing is ever marked `|safe`. Content
containing `<script>`, HTML tags, or quotes renders as inert escaped
text, never as executable markup, since scanner output and (eventually)
LLM-generated text must be treated as untrusted.

## SARIF report

```bash
sentinelai scan . --format sarif --output results.sarif
```

`--format sarif` produces a [SARIF](https://sarifweb.azurewebsites.net/)
2.1.0 document (`$schema`/`version: "2.1.0"`, one `run` with `tool.driver`
metadata, deduplicated `rules`, and one `result` per scanner finding) —
the standard interchange format security tooling, CI/CD systems, and
code-scanning integrations (e.g. GitHub code scanning) already know how
to consume. This milestone only adds the generator and the CLI flag;
wiring an actual GitHub Actions/CI upload step is not implemented here.

Severity maps to SARIF's `level` deterministically: `critical`/`high` →
`error`, `medium` → `warning`, `low` → `note`. Scanner findings without a
matching `AIEnrichedFinding` (correlated by `finding_id`) get no `ai`
properties block at all — never fabricated confidence or verification
data. AI confidence never influences `level`, which always reflects the
scanner's own severity. Each result also carries a `partialFingerprints`
entry keyed on `finding_id`, so consumers can track the same finding
across repeated scans.

Like the other formats, it's built from the same `ScanResult` +
`ScanStatistics` and never recalculates anything — the already-computed
`ScanStatistics` is attached as-is under `runs[0].properties.sentinelai.statistics`.

**Validation:** output is checked against SARIF 2.1.0's documented
structure via targeted structural tests (`tests/test_sarif_report.py`),
not a full JSON Schema validator — deliberately, to avoid a large
dependency; see `sarif_report.py`'s module docstring for the reasoning.

## Generating reports from a saved scan (`sentinelai report`)

Scanning and reporting are separate steps. Run the scan once, save its
JSON report, then generate as many other report formats from that saved
file as you like — without rerunning any scanner or AI reasoning:

```bash
# Scan once, save the canonical JSON result
sentinelai scan . --format json --output scan-result.json

# Generate any other format from that saved result - as many times as you like
sentinelai report scan-result.json --format markdown --output report.md
sentinelai report scan-result.json --format html --output report.html
sentinelai report scan-result.json --format sarif --output results.sarif
```

`report` never touches a `FindingsProvider` — it only reads the JSON
file, reconstructs the internal `ScanResult`, and hands it to the exact
same report generators `scan` uses (`sentinelai/reporting/`), so a
report produced this way is indistinguishable from one produced directly
by `scan`.

**Round-trip:** the existing JSON report schema already carries
everything needed to reconstruct a `ScanResult` — no second persistence
format was introduced. `repository`, `scan`, and `findings.scanner` /
`findings.ai_enriched` map field-for-field onto `ScanResult`'s own
fields (see `sentinelai/reporting/loader.py`).

**Statistics:** never trusted from the input file. `report` always
recomputes `ScanStatistics` from the reconstructed findings via the same
`calculate_statistics()` used everywhere else, so a hand-edited
`statistics` block in the input can't produce a report with inconsistent
numbers — and because that function is deterministic, recomputing
reproduces the original scan's numbers exactly whenever the findings
themselves are untampered.

**Input validation:** malformed JSON, a missing/wrong-shaped field, an
invalid enum value, or an unsupported `schema_version` all produce a
clean one-line error and exit code 2 (`INVALID_INPUT`) — never a Python
traceback, and nothing in the input file is ever executed or interpreted
as code.

## Exit codes and CI/CD gating

Every exit code SentinelAI returns is one of five named values
(`sentinelai/core/exit_codes.py`) — a CI pipeline should branch on these,
not on "zero vs. nonzero":

| Code | Name | Meaning |
|---|---|---|
| `0` | `SUCCESS` | Completed. For `scan`, this includes the case where findings exist but `--fail-on` wasn't crossed (or wasn't given) — SentinelAI is informational by default. |
| `1` | `SECURITY_FINDINGS` | Completed successfully, but findings at or above the `--fail-on` threshold were present. This is a security **policy** outcome, not a tool failure. |
| `2` | `INVALID_INPUT` | The request can't be fulfilled as given: a bad flag value, a repository path that doesn't exist, a malformed/unsupported saved report passed to `report`, or an output path that can't be written. Fixable by the user. |
| `3` | `PROVIDER_ERROR` | The `FindingsProvider` (scanner/backend integration) raised while producing a scan result. SentinelAI is working; the thing it depends on is not. |
| `4` | `INTERNAL_ERROR` | An unexpected, unhandled failure inside SentinelAI itself — a bug, not a policy or input problem. Should be rare; run with `--debug` to see the full traceback. |

### `--severity` vs. `--fail-on` — not the same thing

```bash
# --severity: controls what is INCLUDED/DISPLAYED
sentinelai scan . --severity high        # only show high & critical findings

# --fail-on: controls the EXIT CODE, independent of what's displayed
sentinelai scan . --fail-on high         # exit 1 if high/critical findings are present
```

`--fail-on` accepts `low`, `medium`, `high`, `critical`, or `none`
(default). `--fail-on high` fails on high **or** critical findings, but
not medium/low; `--fail-on critical` fails only on critical findings.
They compose: `--fail-on` evaluates against whatever `--severity` left
in the result, so a CI pipeline gates on exactly what it can also see in
the report.

**Default behavior is unchanged and backward compatible:** with no
`--fail-on`, `scan` always exits `0` on a successful run, even with
findings present — exactly as before this milestone. A CI pipeline opts
into gating explicitly:

```bash
sentinelai scan . --fail-on high --format sarif --output results.sarif
```

### Provider/system failures

If the `FindingsProvider` itself raises (a backend outage, a scanner
crash, once a real provider replaces the mock), SentinelAI reports
`PROVIDER_ERROR` (`3`) — never `SECURITY_FINDINGS`. A pipeline should
treat these very differently: `SECURITY_FINDINGS` means "the scan ran,
findings failed the policy"; `PROVIDER_ERROR` means "the scan didn't
complete at all."

### `--debug`

An app-level flag (goes before the subcommand):

```bash
sentinelai --debug scan . --fail-on high
```

Without it, every error is a single actionable line on **stderr**;
`--debug` additionally prints the full traceback (also on stderr).
Structured output (`--format json/markdown/html/sarif`) always goes to
**stdout** and is never mixed with error/diagnostic text, so
`sentinelai scan . --format json 2>errors.log | jq .` behaves correctly
even when something goes wrong.

## CI/CD Integration

`.github/workflows/ci.yml` (repo root) runs on every `push` and
`pull_request`:

1. checks out the repository and sets up Python 3.12
2. installs SentinelAI from `pyproject.toml` (`pip install -e ".[dev]"`) — no dependency list duplicated into the workflow
3. runs the full test suite (`pytest -v`) — a test failure stops the workflow before SentinelAI ever runs, and is never reported as a security finding
4. runs SentinelAI **once**: `sentinelai scan . --format json --output scan-result.json --fail-on high`
5. generates SARIF and HTML from that *saved* result (`sentinelai report scan-result.json --format sarif|html ...`) — the repository is never re-scanned
6. uploads `scan-result.json`, `sentinelai-results.sarif`, and `sentinelai-report.html` as a single `sentinelai-security-reports` artifact, unconditionally (`if: always()`), so reports are preserved even when the gate or the tool itself fails
7. evaluates the captured scan exit code and fails the job with a clearly labeled message distinguishing a security-policy failure (exit `1`) from a tool failure (exit `2`/`3`/`4`)

The workflow uses only `MockFindingsProvider` — the bundled sample
dataset — so it requires **no API keys or secrets** (no OpenAI,
Anthropic, Gemini, Hugging Face, or Qdrant credentials) and produces the
same deterministic result on every run. `permissions: contents: read` is
the only permission granted.

**What this does *not* do yet:** upload the SARIF to GitHub's native
Code Scanning UI (`github/codeql-action/upload-sarif`) — reports are
preserved as a downloadable artifact only, not wired into GitHub's
Security tab. That's a reasonable next step, not implemented here.

### Reproducing the CI run locally

```bash
cd sentinelai-cli
pytest -v
sentinelai scan . --format json --output scan-result.json --fail-on high
echo "exit code: $?"   # 0 = pass, 1 = security gate failed, 2/3/4 = tool failure
sentinelai report scan-result.json --format sarif --output sentinelai-results.sarif
sentinelai report scan-result.json --format html --output sentinelai-report.html
```

## What's next

- Swap `MockFindingsProvider` for a real provider once Nithanth's
  scanner backend is ready.
- Native GitHub Code Scanning SARIF upload.
- Git hooks / pre-commit integration, Docker, and release automation.
