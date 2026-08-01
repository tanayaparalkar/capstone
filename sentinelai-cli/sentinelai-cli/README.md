# SentinelAI CLI — Phase 1

Viraj's piece of SentinelAI: the Typer-based CLI that will eventually
trigger real scans against Nitaanth's backend. Right now it runs entirely
against a bundled mock dataset so the CLI, its output formats, and the
shared `Finding` schema can all be built and demoed without waiting on
the rest of the pipeline.

## Install

```bash
cd sentinelai-cli
pip install -e .
```

This registers a `sentinelai` command on your PATH (editable install, so
code changes take effect immediately without reinstalling).

If you'd rather not install it as a package, `pip install -r requirements.txt`
and run it as `python -m sentinelai.main scan .` instead.

## Usage

```bash
# Scan the current directory, print a compact table
sentinelai scan .

# Scan a specific path
sentinelai scan /path/to/repo

# Only show high and critical findings
sentinelai scan . --severity high

# Export a full report (table view is intentionally summary-only)
sentinelai scan . --format markdown -o report.md
sentinelai scan . --format html -o report.html
sentinelai scan . --format json -o report.json

# Check version
sentinelai version
```

### Flags

| Flag | Values | Notes |
|---|---|---|
| `--severity` / `-s` | `low`, `medium`, `high`, `critical` | Shows findings at **or above** this level |
| `--format` / `-f` | `table` (default), `json`, `markdown`, `html` | `table` prints a compact terminal view; the others produce a full report |
| `--output` / `-o` | file path | Writes the report to a file instead of stdout. Ignored (with a warning) for `table` format |

## Why the table looks compact

The terminal table intentionally shows only ID / location / severity /
type / confidence — one line per finding. Full AI descriptions, exploit
paths, and remediation text are paragraph-length and don't belong in a
table; they're in the markdown/html/json output instead. This mirrors
how tools like Bandit or Semgrep behave: a fast compact overview in the
terminal, full detail in a report.

Also note: the table is designed for terminals ~100+ columns wide (the
default on basically every modern terminal app). In a genuinely narrow
(~80-column) terminal it may wrap.

## Project structure

```
sentinelai-cli/
├── sentinelai/
│   ├── __init__.py
│   ├── models.py          # Finding / Severity / Confidence — the shared schema
│   ├── mock_findings.json # 10 realistic sample findings covering every severity
│   ├── data.py             # load_findings() — swap this for the real API call in Phase 4
│   ├── display.py          # Rich terminal table
│   ├── formatters.py       # to_json / to_markdown / to_html
│   └── main.py             # Typer app — wires everything together
├── pyproject.toml
├── requirements.txt
└── README.md
```

**Design note for later phases:** `data.load_findings(repo_path)` is the
only function that knows findings currently come from a local JSON file.
Everything else (`display.py`, `formatters.py`, `main.py`) just consumes
a `list[Finding]` and doesn't care where it came from. When Nitaanth's
`/scan` endpoint is ready (Phase 4), the whole swap is: replace the body
of that one function with an HTTP call that returns the same shape. Send
Nitaanth and Tanaya this file — `models.py` — so the backend's Pydantic
models and Tanaya's agent output match it field-for-field; that's what
prevents an integration-week scramble.

## The `Finding` schema

```json
{
  "id": "SENT-001",
  "file": "config/settings.py",
  "line": 14,
  "rule": "gitleaks.aws-access-key",
  "severity": "critical",
  "type": "secret-exposure",
  "ai_description": "LLM-generated explanation of the vulnerability",
  "exploit_path": "LLM-generated exploit narrative, or null if not applicable",
  "remediation": "Suggested fix",
  "confidence": "high"
}
```

`severity` is one of `low | medium | high | critical`.
`confidence` is one of `low | medium | high`.
`exploit_path` is nullable — not every finding (e.g. a vulnerable
dependency with no clear exploit narrative) will have one.

## What's next (Phase 2+)

- **Phase 2:** web frontend (scan form + results view) consuming this
  same mock data via a mocked API call, built in parallel with this CLI.
- **Phase 3:** polish the markdown/html templates in `formatters.py` into
  fuller report designs.
- **Phase 4:** swap `data.load_findings()` for a real call to Nitaanth's
  `/scan` endpoint.
