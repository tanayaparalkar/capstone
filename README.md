# SentinelAI

A command-line security scanner that runs several established analysis tools
over a repository, correlates their findings into single issues, and renders
the result as JSON, Markdown, HTML, or SARIF. AI enrichment is available as an
opt-in layer backed by a local model — the scanner works fully without it.

SentinelAI is a capstone project. The CLI is the deliverable; the rest of this
repository is the benchmark fixture, container packaging, CI pipeline, and the
measurements behind the accompanying paper.

---

## What it does

Most scanners answer "what did *this* tool find?" SentinelAI runs five of them
and answers "what is actually wrong with this repository?" — deduplicating the
same defect reported by different tools under different rule IDs, and grading
severity consistently across all of them.

- **Runs real tools, not reimplementations.** Semgrep, Bandit, GitLeaks, and
  optionally Trivy and OSV-Scanner, each invoked as a subprocess and parsed
  from its documented JSON output.
- **Fails closed.** If a scanner is missing or errors, the scan fails with a
  distinct exit code rather than silently reporting fewer findings.
- **Deterministic output.** Finding IDs are derived from content, not list
  position, so two runs of the same commit produce the same report.

## Core capabilities

| Capability | Detail |
|---|---|
| **Scanner tiers** | `core` — Semgrep, Bandit, GitLeaks (default, no extra setup). `extended` — adds Trivy and OSV-Scanner for dependency scanning, opt-in via `--extended`. |
| **Correlation** | Findings from different scanners describing the same defect are grouped into one issue, matched on normalized CWE, file, and line proximity. |
| **AI enrichment** | Optional. A multi-agent pipeline (evidence → exploit/remediation → critic → synthesizer) runs against a **local** Ollama model, with a deterministic verifier over its output. Off unless configured. |
| **Report formats** | `json`, `markdown`, `html`, `sarif`. `sentinelai report` re-renders a saved JSON scan into any other format without re-scanning. |
| **CI/CD** | `--fail-on <severity>` gates a pipeline via exit code. SARIF output uploads to GitHub code scanning. See [`.github/workflows/ci.yml`](.github/workflows/ci.yml). |
| **Docker** | Reproducible image with every scanner pinned by version and verified by checksum. Runs non-root; scans a read-only mount. |

**Exit codes:** `0` success · `1` findings crossed `--fail-on` · `2` invalid
input · `3` scanner/provider failure · `4` internal error.

## Project structure

```
capstone/
├── sentinelai-cli/            The CLI application
│   ├── sentinelai/            Source: scanners, correlation, AI, reporting
│   ├── tests/                 Test suite (1,100+ tests)
│   ├── docs/CONTRACTS.md      Data contracts between layers
│   └── README.md              Full technical documentation
├── sentinelai-manual-test/    Intentionally vulnerable benchmark fixture
├── Dockerfile                 Reproducible, version-pinned container image
├── .github/workflows/ci.yml   Test suite + self-scan security gate
├── PAPER_RESULTS.md           Measurements reported in the paper
└── POLISH_REPORT.md           Engineering hardening log
```

## Quick start (host)

Requires Python 3.9+ and the three core scanners on your `PATH`.

```bash
# 1. Install the core scanners
pip install semgrep bandit          # or: brew install semgrep bandit
brew install gitleaks               # Linux: see github.com/gitleaks/gitleaks

# 2. Install the CLI
cd sentinelai-cli
python3 -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -e .
cd ..

# 3. Scan the bundled vulnerable fixture
sentinelai scan sentinelai-manual-test
```

That fixture is 4 Python files and 57 lines, and a core scan reports
**17 raw findings correlated into 13 issues**, 4 of which were caught by more
than one scanner. It is the fastest way to confirm the install works.

Other useful invocations:

```bash
sentinelai scan . --format json --output scan-result.json   # save a scan
sentinelai report scan-result.json --format html -o r.html  # re-render it
sentinelai scan . --fail-on high                            # CI gate: exit 1
sentinelai scan . --severity high                           # display filter
sentinelai --help                                           # all flags
```

## Quick start (Docker)

No local Python or scanner installs — the image carries all five scanners,
pinned and checksum-verified.

```bash
docker build -t sentinelai:local .

mkdir -p sentinelai-output
docker run --rm \
  -v "$(pwd):/workspace:ro" \
  -v "$(pwd)/sentinelai-output:/output" \
  sentinelai:local \
  scan /workspace --format json --output /output/scan-result.json
```

The scanned repository is mounted **read-only**; reports go to a separate
writable mount. The container runs as an unprivileged user (uid 1000) — if
your host uid differs, add `--user "$(id -u):$(id -g)"`.

## Extended tier: dependency scanning

`--extended` adds Trivy and OSV-Scanner. Both must be installed — extended is
a superset of core, never a replacement for it.

```bash
brew install trivy osv-scanner      # macOS; see each project for Linux
sentinelai scan . --extended
```

SentinelAI invokes OSV-Scanner recursively, so supported manifests are
discovered anywhere in the tree — you can point `--extended` at a repository
root and it will find nested manifests in subdirectories, not only one sitting
at the top level. Paths excluded by `.gitignore` are not scanned (see
[Limitations](#limitations)).

If Trivy or OSV-Scanner is missing, or the tree contains no manifest OSV
recognizes at any depth, `--extended` **fails with exit 3** rather than
reporting an empty dependency result. To scan such a repository, drop
`--extended`.

## Optional: local AI enrichment

Entirely optional, entirely local. No API keys, no data leaves your machine,
and nothing here is required to use SentinelAI.

```bash
# One-time: install Ollama (ollama.com), then pull both models
ollama pull llama3.1:8b
ollama pull nomic-embed-text

export SENTINELAI_AI_LLM_MODEL=llama3.1:8b
export SENTINELAI_AI_EMBEDDING_MODEL=nomic-embed-text

sentinelai scan sentinelai-manual-test --ai
```

Enrichment activates only when **both** variables are set. `--ai` requires it
and fails fast if it is not configured; `--no-ai` forces a scanner-only run.
Expect this to take minutes rather than seconds — it makes several model calls
per correlated issue. Model choice, host, and retry behaviour are configurable;
see the [CLI documentation](sentinelai-cli/README.md).

## Tests

```bash
cd sentinelai-cli
pip install -e ".[dev]"
pytest -q            # default suite
pytest -q -m ""      # also runs tests that invoke real scanners
```

The default run excludes benchmark-marked tests, which shell out to real
scanner binaries and reach the network. CI runs the default suite.

## Documentation

| Document | Contents |
|---|---|
| [CLI documentation](sentinelai-cli/README.md) | Full manual: every flag, all four report formats, scanner tiers, AI configuration and troubleshooting, CI integration |
| [Data contracts](sentinelai-cli/docs/CONTRACTS.md) | The types passed between scanning, correlation, AI, and reporting |
| [Benchmark fixture](sentinelai-manual-test/README.md) | What the fixture contains and which scanner catches what |
| [Paper results](PAPER_RESULTS.md) | Measurements reported in the paper, with their scope and caveats |
| [Polish report](POLISH_REPORT.md) | Engineering hardening log |

## Limitations

Worth knowing before you rely on the output:

- **Dependency scanning inherits OSV-Scanner's behaviour**, including that it
  honours `.gitignore`. A manifest inside an ignored path is not discovered,
  and a repository whose only manifest is ignored fails as though it had none.
- **Trivy and OSV findings are not merged with each other.** Both may report
  the same advisory under different identifiers, so an extended-tier count is
  raw findings, not distinct vulnerabilities.
- **Dependency findings carry no line numbers.** They belong to a package
  rather than a line, so they never participate in correlation and always
  remain single-scanner issues.
- **AI enrichment is optional and unverified by default.** It requires a
  locally configured model, adds substantial runtime, and its output is LLM
  reasoning — reviewed by a deterministic verifier, but not a guarantee.
- **The extended tier is not exercised in CI**, which installs only the core
  three scanners by design. Dependency scanning is verified locally.
- **This is a capstone project, not a production security product.** It finds
  what the underlying tools find; a clean scan is not evidence that a
  repository is secure.

## License

Released under the [MIT License](LICENSE).
