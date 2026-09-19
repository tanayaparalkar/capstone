# SentinelAI CLI

Viraj's piece of SentinelAI: the Typer-based CLI that drives a scan and
renders the results. It depends only on the `FindingsProvider` interface
(see `docs/CONTRACTS.md`), not on any specific data source. By default
`sentinelai scan` uses `LiveFindingsProvider` — Nithanth's real backend
and scanner framework (Semgrep, Bandit, GitLeaks; plus Trivy and
OSV-Scanner with `--extended`) — via `_get_provider()` in `main.py`. `MockFindingsProvider`, backed by a bundled sample dataset,
remains available and is what the CLI's own test suite pins itself to,
for deterministic, tool-free assertions. AI enrichment (Tanaya's layer)
is opt-in on top of either provider — see "AI Enrichment Configuration"
below.

Three primary commands:

- **`sentinelai scan`** — retrieves a scan result (from a
  `FindingsProvider`), shows it live, writes a report, or optionally interactively fixes issues (`--fix`).
- **`sentinelai fix`** — runs an end-to-end interactive remediation workflow: scans, narrates vulnerabilities with Rich threat cards and unified visual diffs, prompts for approval, applies atomic patches with AST syntax validation, and executes closed-loop verification.
- **`sentinelai undo`** — atomically rolls back the most recent remediation session from `.sentinelai_backups/`.
- **`sentinelai report`** — renders an *existing, already-saved* scan
  result as a report. It never runs a scan again.

## Install

```bash
cd sentinelai-cli
python3 -m venv .venv
.venv/Scripts/pip install -e ".[dev]"   # macOS/Linux: .venv/bin/pip
```

This registers a `sentinelai` command on your PATH (editable install, so
code changes take effect immediately without reinstalling). The `[dev]`
extra pulls in `pytest` for running the test suite.

If you'd rather not install it as a package, `pip install -r requirements.txt`
and run it as `python3 -m sentinelai.main scan .` instead.

## Run in Docker

The `Dockerfile` at the repository root packages the CLI with all five
scanners pinned, so a scan is reproducible without installing Semgrep,
Bandit, GitLeaks, Trivy, or OSV-Scanner on the host.

Docker here is **packaging only**. The CLI inside behaves exactly as it does
on a host — same scanners, same core/extended tiers, same correlation rules,
same report contracts. Nothing in the image listens on a port, and there is no
server, database, or daemon.

### Build

```bash
docker build -t sentinelai:local .
```

Build from the repository root (the `Dockerfile` needs `sentinelai-cli/` in
its context). BuildKit supplies `TARGETARCH`, so the same command produces a
working image on both x86-64 and Apple Silicon.

### Core scan

```bash
mkdir -p sentinelai-output
docker run --rm \
  -v "$(pwd):/workspace:ro" \
  -v "$(pwd)/sentinelai-output:/output" \
  sentinelai:local \
  scan /workspace --format json --output /output/scan-result.json
```

### Extended scan

Adds Trivy and OSV-Scanner. Needs network access at runtime — Trivy downloads
its vulnerability database on first use (~108 MB) and OSV-Scanner queries
osv.dev:

```bash
mkdir -p sentinelai-output
docker run --rm \
  -v "$(pwd):/workspace:ro" \
  -v "$(pwd)/sentinelai-output:/output" \
  sentinelai:local \
  scan /workspace --extended --format json --output /output/scan-result.json
```

Every other flag works unchanged — `--format markdown|html|sarif`,
`--fail-on`, `--severity`. Exit codes are the same as on a host, so
`--fail-on high` gates a pipeline from inside a container exactly as it does
outside one.

### Filesystem contract

**The scanned repository is mounted read-only (`:ro`) and reports are written
to a separate `/output` mount.** SentinelAI never writes into the repository
it scans. `mkdir -p sentinelai-output` first: Docker creates a missing bind-mount
directory as `root`, which the container's unprivileged user could not then
write to.

The container runs as a non-root user (`sentinelai`, uid 1000). If your host
uid is not 1000, either `chmod 777 sentinelai-output` or add `--user "$(id -u):$(id -g)"`
to the run command.

### What is pinned, and how it is verified

| Tool | Version | Source |
|---|---|---|
| Semgrep | 1.172.0 | PyPI |
| Bandit | 1.9.4 | PyPI |
| GitLeaks | 8.30.1 | GitHub release |
| Trivy | 0.74.0 | GitHub release |
| OSV-Scanner | 2.5.1 | GitHub release |

All five are pinned as build `ARG`s — never `latest`. The three GitHub
downloads are fetched to disk and checked with `sha256sum -c` against each
project's own published checksum file before being installed; nothing is
piped into a shell, and no digest is hardcoded in the `Dockerfile`.

**Limitation, stated plainly:** those checksum files ship from the *same*
GitHub release as the binaries they describe. Verification therefore proves the
download arrived intact — catching truncation, corruption, and mirror
problems — but it is **not an independent trust root** and cannot detect a
compromised or re-tagged upstream release. Pinning digests directly would be
stronger, at the cost of six per-architecture values needing a manual update on
every version bump.

### Git ownership

The image sets `git config --global --add safe.directory '*'`. A bind-mounted
host repository is owned by a uid that does not exist inside the container, and
git refuses to operate on such a checkout — which would fail every scan of a
git repository with `PROVIDER_ERROR`. This relaxes a protection aimed at a
threat this container does not face: it is throwaway, unprivileged, and the
mount is read-only.

### AI enrichment is optional and not bundled

**The standard, reproducible container path is scanner-only.** The image
contains no model weights and runs no Ollama server, so AI enrichment does
**not** work out of the box and no part of this section claims otherwise.

If you want it, you must run Ollama yourself on the host, pull both models
(see "AI Enrichment Configuration" below), and point the container at it —
adding roughly `--add-host=host.docker.internal:host-gateway` plus
`-e SENTINELAI_AI_LLM_MODEL`, `-e SENTINELAI_AI_EMBEDDING_MODEL`, and an
Ollama host variable. That path is unsupported here, deliberately: it depends
on your host networking and on models this image does not ship, which is
exactly what makes it non-reproducible and therefore not the container's
documented use.

## Two ways to run SentinelAI

SentinelAI has one scan pipeline and two modes. They are suited to different
jobs, and the difference is mostly latency:

| | **Scanner-only** (default) | **AI-enriched** (opt-in) |
|---|---|---|
| Setup | none | local Ollama + two models |
| Typical runtime | ~4 s on a small repository | ~285 s on the same repository (approximate local measurement — see below) |
| Cost model | fixed scanner startup | **three** local model requests per **correlated issue** |
| Output | findings, severities, all report formats, `--fail-on` gating | the same, plus a per-finding explanation, impact, and remediation |
| Best for | **CI gating** — fast, deterministic, no secrets | **local triage** — reading and understanding findings |

Both modes run the core scanner set (Semgrep, Bandit, GitLeaks) and produce
every report format. Scanner-only is not a degraded mode: it is what this
project's own CI runs on every push. Dependency scanning is a separate,
opt-in axis — see "Scanner tiers" below.

The AI-enriched pipeline is multi-agent and runs sequentially, so its runtime
scales with finding count — which is why it is documented as a local
developer-triage workflow rather than a CI gate. Each correlated issue costs
**three generation calls**: evidence analysis, exploit and remediation analysis,
then a grounding critique. The benchmark repository has 13 correlated issues, so
a full enrichment run makes 39 generation calls.

**The ~285 s figure is an approximate local measurement, not a latency
promise.** It is the median of three controlled consecutive runs — 279.90 s,
285.31 s, and 304.80 s, with a warm-up run discarded — using `llama3.1:8b` for
generation and `nomic-embed-text` for embeddings on an idle 10-core Apple
Silicon machine. Results vary with hardware, model, quantisation, and machine
load; runtime can increase substantially under competing CPU/GPU load. Treat it
as an order-of-magnitude guide for your own setup rather than a number to plan
against.

`../PAPER_RESULTS.md` records a lower figure for AI enrichment. It is a
historical record of the earlier single-call-per-finding pipeline that preceded
this multi-agent workflow, kept as measured, and is not a claim about current
behaviour.

## Scanner tiers

Which scanners run is a separate axis from `--quick`/`--full` (a depth
control) and from AI enrichment. There are two tiers:

| Tier | Scanners | How to run |
|---|---|---|
| **core** (default) | Semgrep, Bandit, GitLeaks | `sentinelai scan .` |
| **extended** (opt-in) | the core three **plus** Trivy and OSV-Scanner | `sentinelai scan . --extended` |

Extended is a superset: `--extended` adds dependency scanning, it never
takes code scanning away. `--full` does **not** imply `--extended` — an
existing command keeps running exactly the scanners it always ran.

The tier is recorded in every report as `scan.scanner_tier`, so a finding
count can always be read against the scanner set that produced it.

### Prerequisites

**Core scans need nothing new.** `sentinelai scan` runs Semgrep, Bandit, and
GitLeaks only; Trivy and OSV-Scanner do not have to be installed, and a core
scan succeeds on a machine that has neither.

**Extended scans require all three of:**

1. **Trivy** on `PATH` — verified against **0.74.0**.
2. **OSV-Scanner** on `PATH` — verified against **2.5.1**.
3. **A dependency source OSV recognizes** somewhere in the scanned tree (a
   `requirements.txt`, lockfile, SBOM, and so on). Nesting is fine — the whole
   tree is searched, not just the directory you point at.

```bash
brew install trivy osv-scanner        # macOS; see each project for Linux
```

Miss any one of the three and `--extended` fails closed with `PROVIDER_ERROR`
(**exit 3**) rather than silently reporting fewer findings. In particular, a
repository with **no recognized OSV package source** returns exit 3: OSV-Scanner
exits 128 with `No package sources found`, and SentinelAI treats that as a
scanner failure, not an empty result. If you want to scan such a repository,
drop `--extended` — the core tier covers it.

One thing to know about "somewhere in the tree": OSV-Scanner honours
`.gitignore`, so a manifest inside an ignored directory (a vendored or
build-output path, say) is not discovered, and a repository whose *only*
manifest is ignored still returns exit 3.

### The dependency scanners

Both are invoked as subprocesses, exactly like the core three.

- **Trivy** runs as `trivy fs --scanners vuln --format json`. Vulnerability
  scanning only — Trivy's secret scanner is deliberately *not* enabled, since
  GitLeaks is this project's dedicated secret scanner and enabling both would
  report every secret twice. It needs a local vulnerability database, which it
  downloads on first use (~108 MB, then cached).
- **OSV-Scanner** runs as `osv-scanner scan source --format json --recursive`.
  It queries osv.dev at scan time, so it needs network access, and it also
  resolves *transitive* dependencies that never appear in your manifest.
  `--recursive` is not optional: without it OSV inspects only the top level of
  the directory it is given, so scanning a repository root whose manifests sit
  in subdirectories would report `No package sources found` and fail the scan.
  It widens where OSV looks, and nothing else — a tree with no manifest at any
  depth still fails closed exactly as described above.

**OSV group-level deduplication.** OSV reports the same underlying issue once
per advisory database that carries it, then states the equivalence itself in
`groups[].ids`. On the benchmark repository that is 50 raw `vulnerabilities`
entries covering 25 distinct issues, every one a PYSEC/GHSA pair. SentinelAI
emits **one finding per group, not per advisory record**, so those 25 issues
produce 25 findings rather than 50. Nothing is discarded: every group id and
alias (including the CVE) is preserved in the finding's message and evidence.

### How dependency findings differ

Dependency findings are **manifest-scoped and line-less**. A dependency
vulnerability belongs to the package, not to the line of the manifest that
happens to declare it, so these findings carry a `file` (the manifest path,
relative to the repository root) but **no `line_start` or `line_end`**. Because
the correlation heuristic requires a source location, they are never merged —
each stays its own singleton issue.

That is a deliberate accuracy decision. Trivy does expose package-level line
numbers, and using them was measured to be actively wrong on this project's own
benchmark: `CVE-2020-14343` and `CVE-2020-1747` both map to pyyaml at line 2
under CWE-20, so line-based correlation would have collapsed two distinct
advisories into one issue.

### Known limitation: Trivy and OSV findings are not merged

Trivy and OSV-Scanner **may describe the same vulnerable dependency through
different advisory identifiers** — Trivy reports CVEs, OSV reports GHSA/PYSEC
groups that list the CVE as an alias. On the benchmark repository all 12 of
Trivy's direct-dependency CVEs are also reported by OSV-Scanner, so those
issues appear twice, once per tool.

They are deliberately left separate for now. SentinelAI's correlation heuristic
is **source-location and CWE based**, and it does not merge manifest-scoped
dependency advisories — extending it to do so on package name, installed
version, and advisory alias sets would be a different kind of rule, not a
tuning of the existing one. **Dependency-aware advisory/alias correlation is
future work.** Until then, read a `--extended` dependency count as raw findings
across two tools, not as a count of distinct vulnerabilities.

## AI Enrichment Configuration

`sentinelai scan` runs Tanaya's AI layer (RAG retrieval, LLM reasoning,
and verification over each finding) only when it's explicitly
configured via environment variables. If it isn't, `scan` still works
exactly as described in this README — `ai_findings` stays empty and the
command still succeeds. AI is opt-in, never required.

### Required to enable AI enrichment

AI enrichment uses **two different Ollama models**, because Ollama serves
generation and embeddings from different model families. Both must be set
together — if either is missing, AI enrichment is skipped entirely and the
scan still succeeds (see "When AI is not configured" below):

| Variable | Purpose | Ollama endpoint | Example |
|---|---|---|---|
| `SENTINELAI_AI_LLM_MODEL` | Generates the explanation, exploit path, impact, and remediation for each finding | `/api/generate` | `llama3.1:8b` |
| `SENTINELAI_AI_EMBEDDING_MODEL` | Embeds the security knowledge base so findings can be matched to it | `/api/embed` | `nomic-embed-text` |

> **⚠️ These must be two different models.** A text-generation model
> (`llama3`, `llama3.1:8b`, `mistral`, …) **cannot** produce embeddings —
> Ollama rejects the request with `HTTP 501: Not Implemented`. Setting
> `SENTINELAI_AI_EMBEDDING_MODEL` to the same value as
> `SENTINELAI_AI_LLM_MODEL` is the most common way to hit this. See
> [Troubleshooting](#troubleshooting-ai-enrichment) below.

### Quick start

```bash
# 1. Install Ollama (see https://ollama.com), then start the server
ollama serve

# 2. Pull BOTH models - one for generation, one for embeddings
ollama pull llama3.1:8b        # text generation
ollama pull nomic-embed-text   # embeddings (dedicated embedding model)

# 3. Point SentinelAI at them - two different models
export SENTINELAI_AI_LLM_MODEL=llama3.1:8b
export SENTINELAI_AI_EMBEDDING_MODEL=nomic-embed-text

# 4. Scan with AI enrichment
sentinelai scan . --ai
```

No API key is involved at any point. Ollama runs locally, and SentinelAI
never contacts a cloud service — there is no credential to configure and none
of these variables holds a secret.

### Controlling AI enrichment: `--ai` and `--no-ai`

| Invocation | Behaviour |
|---|---|
| `sentinelai scan .` | Enriches **only if** both model variables are set; otherwise scanner-only. Unchanged from before these flags existed. |
| `sentinelai scan . --ai` | Enrichment is **required**. If either variable is unset, exits `INVALID_INPUT` (**2**) *before scanning*, naming the missing variable(s) and the setup commands. |
| `sentinelai scan . --no-ai` | Skips enrichment **even when configured** — a fast, deterministic scanner-only run. |

`--ai` and `--no-ai` cannot be combined.

Use `--ai` in any script or pipeline where a scanner-only report would be
mistaken for an AI-enriched one: without it, a mistyped variable name silently
produces a report with an empty `ai_enriched` list and exit code 0. Use
`--no-ai` when the two variables live in your shell profile but you want a
quick scan without waiting on the model.

### A note on model tags

`llama3.1` is a **model family shorthand**, not a local model. Ollama resolves
it to a default tag when pulling, but treats every tag as a distinct local
name afterwards — so if you pulled `llama3.1:8b`, then
`SENTINELAI_AI_LLM_MODEL=llama3.1` fails with HTTP 404 even though the weights
are on disk.

**Always use the exact tag you pulled.** `ollama list` shows what you have.
Every runnable command in this README uses `llama3.1:8b` and
`nomic-embed-text`, which is also the configuration the measurements in
`../PAPER_RESULTS.md` were taken under.

To go back to scanner-only mode, pass `--no-ai`, or unset either variable:

```bash
unset SENTINELAI_AI_LLM_MODEL SENTINELAI_AI_EMBEDDING_MODEL
```

### Setting up Ollama

Setting the two variables above is not enough by itself — Ollama is
separate software SentinelAI does not install or bundle. Before
`SENTINELAI_AI_LLM_MODEL`/`SENTINELAI_AI_EMBEDDING_MODEL` can do
anything, Ollama must be installed (see https://ollama.com), running
(`ollama serve`), and already have **both** models pulled locally — see
[Quick start](#quick-start) for the exact commands.

Neither variable is restricted to the example names: any model you've
pulled works, as long as the value matches exactly what `ollama pull`
was given, **and as long as the embedding model is genuinely an
embedding model.** SentinelAI cannot check that from the name alone —
nothing in a model's name reliably indicates whether it supports
embeddings — so the check happens when Ollama is actually called, and
surfaces as the `HTTP 501` error described in
[Troubleshooting](#troubleshooting-ai-enrichment).

If Ollama isn't running, or a named model hasn't been pulled, `scan`
reports a clear error and exits `3` rather than silently skipping AI
enrichment. AI is only silently skipped when the environment variables
themselves are unset (see "When AI is not configured" below).

### Optional AI configuration

| Variable | Default | Meaning |
|---|---|---|
| `SENTINELAI_AI_LLM_HOST` | `http://localhost:11434` | Base URL of the Ollama server (shared by generation and embeddings). |
| `SENTINELAI_AI_RETRIEVAL_TOP_K` | `5` | Number of knowledge-base chunks retrieved per finding. |
| `SENTINELAI_AI_CONFIDENCE_HIGH_THRESHOLD` | `0.75` | Minimum confidence score (inclusive) mapped to `high`. |
| `SENTINELAI_AI_CONFIDENCE_MEDIUM_THRESHOLD` | `0.4` | Minimum confidence score (inclusive) mapped to `medium`; below this is `low`. |
| `SENTINELAI_AI_ENABLE_VERIFICATION` | `true` | Whether the verification step runs before a finding is emitted. |
| `SENTINELAI_AI_LOG_LEVEL` | `INFO` | Logging level for the AI layer. |
| `SENTINELAI_AI_REQUEST_TIMEOUT_SECONDS` | `120` | Per-attempt timeout for every Ollama request (generation and embeddings). Applies to each attempt independently, not divided across retries. Raise it for a slow machine or a larger model. |
| `SENTINELAI_AI_MAX_ATTEMPTS` | `2` | Total attempts per Ollama request, including the first — `1` disables retrying. |

Only **transient** failures are ever retried — connection errors and HTTP
500/502/503/504. Deterministic ones are not: a 404 for a model that was never
pulled, or a 501 from an embedding model that cannot embed, will not resolve
themselves between two attempts, so retrying them would only add delay before
showing the same message. Raising `SENTINELAI_AI_MAX_ATTEMPTS` does not change
that classification. The 2-second backoff between attempts is fixed and not
configurable; keep the attempt count low, because the AI pipeline handles
failures per finding, so the worst case is that delay multiplied across every
finding in a scan.

The defaults reproduce the behaviour these values previously had when they were
hardcoded, so an unconfigured run is unchanged — including the latency figures
in `../PAPER_RESULTS.md`, which were measured under them.

`SENTINELAI_AI_LLM_PROVIDER` and `SENTINELAI_AI_EMBEDDING_PROVIDER` also
exist (see `sentinelai/ai/config.py`) but are currently unused — they're
reserved so a future second LLM/embedding implementation has somewhere
to be selected from, without changing `AISettings`' shape. Setting them
today has no effect.

### Which providers are supported today

**Ollama is the only LLM/embedding provider implemented right now**
(`sentinelai/ai/llm_ollama.py`, `sentinelai/ai/embeddings_ollama.py`).
OpenAI, Gemini, Claude, Groq, and LM Studio are not wired up yet — the
`llm_provider`/`embedding_provider` settings above exist to let a future
provider be selected without a breaking change, but no second
implementation exists in the code today. Don't set them expecting a
cloud provider to activate; nothing currently reads them.

Ollama runs locally and requires **no API key** — `SENTINELAI_AI_LLM_HOST`
names where to reach it, not a credential. More generally, **SentinelAI
never bundles or ships an API key or credential for any provider.**
Whenever a cloud-hosted provider is supported (OpenAI, Gemini, Claude,
Groq, or any future LM-Studio-style local server that does need a key),
its credential would always be supplied by the person deploying
SentinelAI via an environment variable of their own — never checked
into this repository or embedded in the tool itself.

### When AI is not configured (scanner-only mode — the default)

If `SENTINELAI_AI_LLM_MODEL` or `SENTINELAI_AI_EMBEDDING_MODEL` (or
both) is unset, `scan` behaves exactly as if the AI layer didn't exist:
scanner findings are still produced and rendered normally, `ai_findings`
stays an empty list, and the command still exits `0` on success. Missing
AI configuration is never treated as a failure — this is the default,
most common way to run SentinelAI today.

**Scanner-only mode is a fully supported first-class mode, not a
degraded one.** Semgrep, Bandit, and GitLeaks all run, every report
format is produced, and `--fail-on` gating works exactly the same. It is
also what CI runs (see `.github/workflows/ci.yml`), so no CI job needs
Ollama, a model, or any API key.

### Troubleshooting AI enrichment

| Symptom | Cause | Fix |
|---|---|---|
| `AI enrichment failed: the embedding model 'llama3.1:8b' does not support embeddings (HTTP 501 …)` | `SENTINELAI_AI_EMBEDDING_MODEL` points at a text-generation model | `ollama pull nomic-embed-text` then `export SENTINELAI_AI_EMBEDDING_MODEL=nomic-embed-text` |
| `AI enrichment failed: … has no model named 'x' (HTTP 404)` | The model isn't pulled | `ollama pull <model>` |
| `AI enrichment failed: could not reach the Ollama server at http://localhost:11434 …` | Ollama isn't running, or listens elsewhere | `ollama serve`, or set `SENTINELAI_AI_LLM_HOST` |
| `AI enrichment failed: … has no model named 'llama3.1' (HTTP 404)` … `Ollama reported: model 'llama3.1' not found` | The tag doesn't match what you pulled — `llama3.1` is a family shorthand, not a local model | `ollama list`, then use the exact tag (e.g. `llama3.1:8b`) |
| `findings.ai_enriched` is empty and no error is shown | AI isn't configured — one of the two variables is unset | Set both (see [Quick start](#quick-start)), or pass `--ai` to make this an error instead of a silent skip |
| `--ai requires AI enrichment to be configured, but … is not set` (exit `2`) | `--ai` was passed without both variables set | Set the named variable(s), or drop `--ai` for a scanner-only scan |

A failure that prevents AI enrichment entirely — an unreachable server, an
unusable embedding model, or every finding failing — exits with code `3`
(`PROVIDER_ERROR`) and prints a single actionable line to stderr. The scan
is never silently degraded into a success. Add `--debug` for the full
traceback. Failures affecting only *some* findings behave differently; see
[Partial vs. total AI failure](#partial-vs-total-ai-failure) below.

Confirming which mode you're in:

```bash
# scanner-only: "unavailable"; AI enrichment: "available"; some failed: "partial"
sentinelai scan . --format json | python3 -c 'import json,sys; print(json.load(sys.stdin)["statistics"]["ai_enrichment_status"])'
```

### Partial vs. total AI failure

AI enrichment makes one model request per finding, so a failure can affect one
finding or all of them. SentinelAI treats those two cases differently on
purpose:

| Situation | `ai_enrichment_status` | Exit code | What you get |
|---|---|---|---|
| Every finding enriched | `available` | 0 | Full AI output for all findings |
| **Some** findings failed | `partial` | 0 | AI output for the findings that succeeded; the rest keep their scanner results and render as "AI enrichment not yet available for this finding" |
| **Every** finding failed | — | **3** | One actionable error; no report written |
| AI not configured | `unavailable` | 0 | Scanner-only mode (the default) |

A finding that fails enrichment is **never dropped** — its scanner result is
still reported in full. Only the AI commentary is missing, and each skipped
finding is logged as a warning naming it and the reason. The number that failed
is `correlated_findings - matched_ai_findings` in the statistics block whenever
the status is `partial`, because AI enrichment runs once per correlated issue.

This is why a single malformed model response no longer discards a whole scan:
with a small local model, one finding occasionally producing unparseable output
is a normal event, not a reason to throw away sixteen good explanations.

### Retry behaviour

Requests to Ollama — both generation and embeddings — are retried **once** after
a fixed 2-second delay, and only for failures a retry could plausibly fix:
connection refused, DNS and socket timeouts, and transient HTTP 500/502/503/504.

Deliberately **not** retried, because they are deterministic and a second attempt
would fail identically while costing the delay:

- malformed or non-JSON model output, and responses failing schema validation;
- configuration errors (for example, an empty model name);
- all 4xx statuses, such as a model that has not been pulled (404);
- **HTTP 501** — Ollama's response when the configured embedding model cannot
  produce embeddings. This is the most common misconfiguration here, so it is
  reported immediately rather than after a pointless delay.

Each attempt keeps the full 120-second request timeout. A healthy request makes
exactly one HTTP call, so retries add nothing to the normal path.

### End-to-end example

With Ollama already running and both models pulled (see above):

```bash
export SENTINELAI_AI_LLM_MODEL=llama3.1:8b        # generation
export SENTINELAI_AI_EMBEDDING_MODEL=nomic-embed-text   # embeddings

sentinelai scan . --format json --output scan-result.json
```

With both variables correctly set and Ollama reachable, `scan` runs
Semgrep/Bandit/GitLeaks as usual and then runs every scanner finding
through retrieval, LLM reasoning, and verification before writing the
report — `scan-result.json`'s `findings.ai_enriched` list is populated
(one entry per scanner finding, correlated by `finding_id`) instead of
staying empty, and `statistics.ai_enrichment_status` no longer reads
`"unavailable"`.

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

## Interactive Remediation Workflow (`sentinelai fix`)

SentinelAI features a comprehensive, safe, closed-loop interactive remediation engine that converts abstract scanner findings and AI reasoning into validated source code patches.

```bash
# Interactively scan and fix issues in the current repository
sentinelai fix .

# Scan with fix flag (equivalent workflow)
sentinelai scan . --fix

# Non-interactive CI mode (auto-apply all verified fixes)
sentinelai fix . -y

# Dry-run simulation (preview diffs and threat cards without writing to disk)
sentinelai fix . --dry-run

# Revert previous session changes atomically
sentinelai undo .
```

### Interactive Decision Prompt

During the remediation session, each finding is presented sequentially with a rich Threat Narrative card and color-coded unified diff. The prompt provides full developer agency:

- `[y]es` — Apply the displayed patch, validate syntax, and verify fix.
- `[n]o` / `[s]kip` — Reject this patch and proceed to next finding.
- `[a]ll` — Automatically accept and apply all remaining patches in the session.
- `[d]etails` — Expand in-depth AI exploit path, blast radius, and remediation guidance.
- `[q]uit` — Terminate remediation session immediately.

### Safety Architecture & Closed-Loop Verification

1. **Deterministic & AI-Generated Patches:** Leverages structured LLM diffs or deterministic AST-based templates across 13 vulnerability categories.
2. **Pre-flight AST Syntax Validation:** Patches are validated with `ast.parse` before writing to disk. Malformed or invalid syntax is rejected immediately.
3. **Multi-File Atomic Snapshotting:** Before any modifications are applied, repository state is preserved in `.sentinelai_backups/<session_id>/`.
4. **Closed-Loop Rescan Verification:** Upon applying a patch, SentinelAI triggers an internal re-scan to confirm the original finding is resolved and verify no new vulnerabilities or regressions were introduced. If a regression occurs, the patch is automatically rolled back.
5. **Session Undo:** Run `sentinelai undo` to restore the exact state prior to the last remediation session.

### Framework-Aware Rate Limiting Advisory

When API endpoints, DoS risks, or resource exhaustion vulnerabilities (CWE-400) are identified, SentinelAI automatically detects the repository's web framework (`Flask`, `FastAPI`, `Django`, `Express.js`, or generic Python/Node) and presents framework-native rate limiting configuration advice (e.g. `Flask-Limiter`, `slowapi`, `django-ratelimit`, `express-rate-limit`) to prevent server downtime and resource starvation.

### Supported Vulnerability Classes

- **SQL Injection (CWE-89):** Parameterizes queries with placeholders and tuple argument binding.
- **Command Injection (CWE-78):** Replaces `os.system()` with `subprocess.run()` list arguments and disables `shell=True`.
- **Insecure Deserialization (CWE-502):** Replaces unsafe `pickle.loads` with `json.loads` and `yaml.load` with `yaml.safe_load`.
- **Arbitrary Code Execution (CWE-94):** Replaces dangerous `eval()` with `ast.literal_eval()`.
- **Weak Cryptography (CWE-327):** Upgrades broken `hashlib.md5()` to `hashlib.sha256()`.
- **Hardcoded Secrets (CWE-798):** Replaces plaintext credentials and API keys with `os.environ.get()`.
- **Security Misconfiguration (CWE-16):** Disables hardcoded `DEBUG = True` with environment guards.
- **Reflected XSS (CWE-79):** Sanitizes untrusted user inputs with `html.escape()`.
- **Broken Access Control / IDOR (CWE-639 / CWE-284):** Scopes database queries to authenticated user context (`current_user.id`).
- **Server-Side Request Forgery (SSRF) (CWE-918):** Adds URL scheme and domain whitelist validation guards.
- **Cross-Site Request Forgery (CSRF) (CWE-352):** Applies `@csrf_protect` decorators to POST endpoints.


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
│   ├── providers/           # FindingsProvider interface + LiveFindingsProvider (default) + MockFindingsProvider
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
`FindingsProvider` interface (`_get_provider()` in `main.py`), currently
returning `LiveFindingsProvider()` — the same seam that was previously
`MockFindingsProvider()` before Nithanth's backend was ready. Swapping
the concrete implementation there again in the future (e.g. a remote/API
provider) would require no change in `presentation/`, `reporting/`, or
the rest of `main.py`. Full details, including how Tanaya's
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
4. installs Semgrep, Bandit, and GitLeaks — the core-tier scanners the scan in step 6 needs on `PATH` to run at all
5. runs SentinelAI **once**, against this repository, through the real `LiveFindingsProvider`: `sentinelai scan . --format json --output scan-result.json --fail-on high`
6. generates SARIF and HTML from that *saved* result (`sentinelai report scan-result.json --format sarif|html ...`) — the repository is never re-scanned
7. uploads `scan-result.json`, `sentinelai-results.sarif`, and `sentinelai-report.html` as a single `sentinelai-security-reports` artifact, unconditionally (`if: always()`), so reports are preserved even when the gate or the tool itself fails
8. evaluates the captured scan exit code and fails the job with a clearly labeled message distinguishing a security-policy failure (exit `1`) from a tool failure (exit `2`/`3`/`4`)

The gate scan runs the **core** tier, and the workflow deliberately does not
install Trivy or OSV-Scanner. Nothing in it runs an extended scan — the gate is
core-tier because dependency advisories are published and amended continuously,
so gating on them would fail the job on days when nothing in this repository
changed, and `pytest -v` excludes the benchmark-marked tests that use those
binaries. Installing tools no step consumes would only add two more ways for CI
to fail. `.github/workflows/ci.yml` carries the pinned install recipe as a
comment for whoever adds an extended job later.

`sentinelai scan` always uses `LiveFindingsProvider` — there is no flag
or environment variable to select `MockFindingsProvider` instead, so
this step performs a real scan of the checked-out repository and its
output can vary between runs as the repository itself changes. The
workflow still requires **no API keys or secrets**: AI enrichment only
runs when `SENTINELAI_AI_LLM_MODEL` and `SENTINELAI_AI_EMBEDDING_MODEL`
are both set, and this workflow never sets them, so no LLM, embedding,
or database credentials of any kind (Ollama or otherwise) are needed
here. `permissions: contents: read` is the only permission granted.

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

- Native GitHub Code Scanning SARIF upload.
- Git hooks / pre-commit integration, Docker, and release automation.
