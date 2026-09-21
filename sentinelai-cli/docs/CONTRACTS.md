# SentinelAI — Shared Contracts

This document describes the Pydantic contracts in `sentinelai/contracts/`
that form the integration boundary between the three layers of the
pipeline:

```
Nithanth                       Tanaya                      Viraj
scanner findings    ─────►     AI enrichment    ─────►      CLI + reports
+ repository context           + verification                + statistics
```

These models are the single source of truth for what data crosses each
boundary. Changing a field here is a cross-team decision, not a local
refactor.

---

## `ScannerFinding`

**Purpose:** deterministic output from a single security scanner
(Semgrep, Bandit, GitLeaks, Trivy, OSV Scanner), normalized into one
common shape. Carries no AI-generated content.

**Owner:** Nithanth (Repository Analysis, Static Security Engine and
Backend Foundation).

**Location:** `sentinelai/contracts/scanner_finding.py`

| Field | Type | Required | Notes |
|---|---|---|---|
| `finding_id` | `str` | Yes | Stable identifier, e.g. `"SENT-001"`. Correlation key used by `AIEnrichedFinding.finding_id`. |
| `scanner` | `str` | Yes | Tool that produced the finding, e.g. `"semgrep"`, `"bandit"`, `"gitleaks"`, `"trivy"`, `"osv-scanner"`. Free-form, not a closed enum — Nithanth may add scanners without a contract change. |
| `category` | `str` | Yes | Vulnerability category, e.g. `"sql-injection"`, `"secret-exposure"`. Free-form, Nithanth's taxonomy. |
| `severity` | `Severity` enum | Yes | `low \| medium \| high \| critical`. |
| `file` | `str` | No | Path relative to repo root. Absent is valid (e.g. whole-dependency findings). |
| `line_start` | `int` | No | First affected line. Absent is valid. |
| `line_end` | `int` | No | Last affected line. Must not be set without `line_start`, and must not be less than `line_start` (enforced by a model validator). |
| `rule_id` | `str` | Yes | Scanner-specific rule identifier, e.g. `"semgrep.python.sql-injection.string-concat"`. |
| `message` | `str` | Yes | Scanner-generated description. |
| `raw_evidence` | `str` | No | Raw code snippet or scanner output backing the finding. |
| `cwe` | `str` | No | CWE identifier, e.g. `"CWE-89"`, where the rule maps to one. |

`file` / `line_start` / `line_end` are optional together because not every
finding maps to a specific source line — a vulnerable dependency version
in `requirements.txt` is a valid finding with only a `file`, or none at all.

**Dependency findings are manifest-scoped and line-less.** Trivy and OSV
Scanner set `file` to the manifest path (relative to the repository root) and
leave `line_start`/`line_end` unset. A dependency vulnerability is a property
of the package, not of the line that happens to declare it, and both tools'
line data was measured to be actively misleading: `CVE-2020-14343` and
`CVE-2020-1747` share pyyaml's line 2 under CWE-20, so a line-based rule would
merge two distinct advisories. Because correlation requires a source location,
these findings are always singleton groups — see "Scanner tiers" under
`ScanResult` below.

**OSV Scanner emits one finding per advisory *group*, not per record.** OSV
reports the same issue once per advisory database carrying it and states the
equivalence in `groups[].ids` (on the benchmark repository: 50 records, 25
groups, each a PYSEC/GHSA pair). The wrapper emits one `ScannerFinding` per
group and preserves every group id and alias — including the CVE — in
`message` and `raw_evidence`, so the deduplication loses no identifier.

---

## `AIEnrichedFinding`

**Purpose:** Tanaya's RAG/agentic-reasoning layer's structured output for
one finding — explanation, exploit narrative, remediation, and a
confidence assessment. Viraj's layer only ever consumes this shape; it
does not calculate confidence, run verification, or assume anything about
the underlying agent graph.

**Owner:** Tanaya (AI Reasoning, RAG, Security Knowledge and Intelligence
Layer).

**Location:** `sentinelai/contracts/ai_finding.py`

| Field | Type | Required | Notes |
|---|---|---|---|
| `finding_id` | `str` | Yes | **Correlation key** — matches the `finding_id` of the primary `ScannerFinding` this enrichment is about. |
| `title` | `str` | Yes | Short human-readable title. |
| `severity` | `Severity` enum | Yes | May restate or refine the scanner's severity. |
| `scanner_sources` | `list[str]` | No (default `[]`) | Scanners that contributed evidence, e.g. `["semgrep", "bandit"]`. |
| `evidence` | `str` | No | Evidence excerpt supporting the explanation. |
| `repository_context` | `str` | No | The relevant slice of repository context used during reasoning for *this* finding — not the full repository-context object Nithanth builds, just what Tanaya chose to attach here. |
| `explanation` | `str` | Yes | LLM-generated explanation of the vulnerability. |
| `exploit_path` | `str` | No | Narrative of how the finding could be exploited, if applicable. |
| `impact` | `str` | No | Description of potential impact if exploited. |
| `remediation` | `str` | Yes | Suggested fix. |
| `patch_suggestion` | `str` | No | Concrete patch/diff suggestion, if generated. |
| `confidence_score` | `float` | Yes | `0.0`–`1.0`. Numeric confidence — Viraj does not compute this, only displays/sorts on it. |
| `confidence_label` | `ConfidenceLabel` enum | Yes | `low \| medium \| high`. |
| `verification_status` | `VerificationStatus` enum | Yes | `unverified \| verified \| rejected \| insufficient_evidence`. |
| `related_findings` | `list[str]` | No (default `[]`) | `finding_id`s of other findings participating in the same multi-step exploit chain. |
| `references` | `list[str]` | No (default `[]`) | External references, e.g. CWE/OWASP links. |

**Note on the knowledge-base fields:** Tanaya's security knowledge base
(`security_kb/`) separately tracks per-vulnerability-*type* fields like
"vulnerable pattern" and "exploit condition" for RAG retrieval. Those are
inputs to her reasoning, not outputs on a per-finding object, so they are
intentionally not modeled here.

---

## `ScanResult`

**Purpose:** the scan-level envelope the reporting layer consumes.
Connects repository/scan metadata with both finding layers.

**Location:** `sentinelai/contracts/scan_result.py`

```python
class ScanResult(BaseModel):
    repository: RepositoryInfo
    metadata: ScanMetadata
    scanner_findings: list[ScannerFinding] = []
    ai_findings: list[AIEnrichedFinding] = []
```

- `RepositoryInfo` — `name`, `path` (required), `commit_hash`, `branch`,
  `languages` (all optional).
- `ScanMetadata` — `timestamp`, `mode` (`quick \| standard \| full`,
  required), `duration_seconds` (optional, filled in once the scan
  finishes), `scanner_tier` (`core \| extended`, defaults to `core`).
- `ScannerTier` — which scanner *set* ran, an axis orthogonal to
  `ScanMode`'s depth control. It defaults to `core`, so a report written
  before the tier existed still loads and reads correctly.

### Scanner tiers

| Tier | Membership | Selected by |
|---|---|---|
| `core` (default) | Semgrep, Bandit, GitLeaks | `sentinelai scan <path>` |
| `extended` | Semgrep, Bandit, GitLeaks, **Trivy, OSV Scanner** | `sentinelai scan <path> --extended` |

`extended` is a strict superset of `core`, never a replacement. `ScanMode`
(`quick`/`standard`/`full`) does not affect membership — `--full` does not
imply `--extended`, so an existing invocation runs exactly the scanners it
always ran. Membership lives on each `ScannerRegistry` registration, and
`ScanMetadata.scanner_tier` records which set produced a given result, so a
finding count can always be read against the scanner set behind it.

**Extended-scan prerequisites.** Trivy and OSV Scanner must both be on `PATH`,
and the scanned tree must contain a dependency source OSV recognizes. A
repository with no recognized OSV package source produces `PROVIDER_ERROR`
(exit 3) under `--extended`: OSV exits 128 with `No package sources found`, and
that is treated as a scanner failure, not an empty result. Core scans require
neither binary and succeed on a machine that has neither.

**Known limitation — dependency findings are not cross-correlated.** Trivy and
OSV Scanner may describe the same vulnerable dependency through different
advisory identifiers (Trivy reports CVEs; OSV reports GHSA/PYSEC groups that
carry the CVE as an alias), so the same issue can appear as two raw findings.
The correlation heuristic is source-location and CWE based and deliberately
does not merge manifest-scoped dependency advisories. Dependency-aware
advisory/alias correlation is future work.

**How the two finding layers connect:** `scanner_findings` and
`ai_findings` are two parallel lists, not a nested structure. A finding
is fully valid with only a `ScannerFinding` and no matching
`AIEnrichedFinding` yet (e.g. before AI enrichment has run, or if it's
disabled) — the reporting layer joins them by `finding_id` at render
time. This keeps the two layers independently evolvable: Nithanth's
scanner output and Tanaya's AI output can each ship on their own
schedule without either blocking the other.

`ScanResult` is deliberately flat about future extension: new optional
fields can be added to any of these three models later (e.g. a
`scan_id`, tool version) without breaking existing reporting code, since
Pydantic models fall back to defaults for anything not explicitly set.

---

## Provider interface

**Location:** `sentinelai/providers/base.py`, `sentinelai/providers/live_provider.py`, `sentinelai/providers/mock_provider.py`

```python
class FindingsProvider(ABC):
    @abstractmethod
    def get_scan_result(
        self,
        repo_path: str,
        mode: ScanMode = ScanMode.STANDARD,
        tier: ScannerTier = ScannerTier.CORE,
    ) -> ScanResult:
        ...
```

Viraj's CLI depends on this abstract interface, **not** on Nithanth's
backend directly — the contract is what makes the following true rather
than aspirational. Two implementations exist today:

- `LiveFindingsProvider` — the default (`main.py`'s `_get_provider()`
  returns this unconditionally). Loads the target repository through
  Nithanth's backend and runs the scanners for the requested `tier`
  against it via `ScannerOrchestrator` — Semgrep, Bandit, and GitLeaks
  for `core` (the default), plus Trivy and OSV Scanner for `extended`
  (`sentinelai scan --extended`) — returning a real `ScanResult` with
  `ai_findings` empty (AI enrichment, when configured, is applied
  separately by the CLI — see the AI-layer sections above).
- `MockFindingsProvider` — reads a bundled JSON fixture instead of
  running real scanners. Still present and still useful: it's what the
  CLI's own test suite pins itself to for deterministic, tool-free
  assertions, independent of whether Semgrep/Bandit/GitLeaks are
  installed in the environment running the tests.

Both return the identical `ScanResult` shape, and nothing in the CLI,
reporting, or statistics code differs based on which one is active. A
future third implementation (e.g. a remote/API-backed provider) would be
a drop-in replacement the same way `LiveFindingsProvider` was.

---

## Integration example

A single finding's path through the full pipeline:

```
ScannerFinding                          AIEnrichedFinding
  finding_id: "SENT-002"                  finding_id: "SENT-002"   ◄── correlation key
  scanner: "semgrep"                      title: "SQL injection via
  category: "sql-injection"                       unparameterized login query"
  severity: critical                      severity: critical
  file: "app/db/queries.py"               scanner_sources: ["semgrep"]
  line_start: 47                          explanation: "The login query is
  rule_id: "semgrep.python.sql-                        built via string
           injection.string-concat"                    concatenation..."
  message: "User-supplied input                exploit_path: "Submitting
           concatenated directly                    \"' OR '1'='1' --\" as
           into a SQL query string."             the username bypasses..."
  raw_evidence: "query = f\"SELECT ...\""  impact: "Full authentication
                                                     bypass..."
        │                                  remediation: "Use parameterized
        │  produced by Nithanth's                       queries..."
        │  scanner layer                   confidence_score: 0.95
        │                                  confidence_label: high
        ▼                                  verification_status: verified
   (passed to Tanaya's                            │
    reasoning pipeline                            │  produced by Tanaya's
    as input evidence)                            │  RAG + reasoning pipeline
                                                    ▼
                              ┌─────────────────────────────────────┐
                              │            ScanResult                │
                              │  scanner_findings: [SENT-002, ...]   │
                              │  ai_findings:      [SENT-002, ...]   │
                              └─────────────────────────────────────┘
                                                    │
                                                    ▼
                                          Reporting layer (Viraj)
                                  joins scanner_findings + ai_findings
                                  on finding_id to render one row/section
                                  per finding, across table/JSON/Markdown/
                                  HTML/SARIF output.
```

**What Viraj's layer does and does not do:**

- Viraj **does not** calculate AI confidence — `confidence_score` and
  `confidence_label` arrive already computed on `AIEnrichedFinding`.
- Viraj **does not** run AI reasoning — no LLM calls, no RAG retrieval,
  no agent orchestration live in this layer.
- Viraj **does not** duplicate scanner logic — no re-implementing
  Semgrep/Bandit/GitLeaks/Trivy/OSV rules or re-deriving severity.
- Viraj **only consumes** the structured contracts above: it reads a
  `ScanResult` from a `FindingsProvider`, joins the two finding lists,
  and renders/exports/summarizes what it's given.

---

## Safe Patch Remediation Engine Contracts

**Location:** `sentinelai/patcher/models.py`

The remediation engine operates over structured, machine-applicable patch specifications, validation states, and closed-loop verification results.

### `Patch`

**Purpose:** Machine-applicable representation of a proposed remediation for a finding.

| Field | Type | Required | Notes |
|---|---|---|---|
| `finding_id` | `str` | Yes | Foreign key to `ScannerFinding.finding_id` / `AIEnrichedFinding.finding_id`. |
| `file_path` | `Path` | Yes | Absolute path to the file to modify. |
| `line_start` | `Optional[int]` | No | 1-indexed first line of code targeted for replacement. |
| `line_end` | `Optional[int]` | No | 1-indexed last line of code targeted for replacement. |
| `original_snippet` | `str` | Yes | Exact code snippet before the patch. |
| `replacement_snippet` | `str` | Yes | Code snippet to replace the original. |
| `diff` | `str` | Yes | Unified diff format (`--- a/... +++ b/...`) for user presentation. |
| `explanation` | `str` | Yes | Human-readable explanation of why this fix is safe. |
| `cwe` | `Optional[str]` | No | CWE ID associated with the vulnerability (e.g., `"CWE-89"`). |
| `category` | `Optional[str]` | No | Vulnerability category (e.g., `"sql-injection"`). |
| `rule_id` | `Optional[str]` | No | Rule ID from the originating scanner. |
| `scanner` | `Optional[str]` | No | Scanner identifier (e.g., `"bandit"`). |
| `severity` | `Optional[Severity]` | No | Severity level of the finding being fixed. |

### `PatchStatus` Enum

- `APPLIED`: Patch applied to target file and passed pre-flight syntax check.
- `REJECTED`: Patch was rejected by the user or pre-flight validation.
- `SKIPPED`: User chose to skip the patch.
- `FAILED`: Patch application failed (file not found, syntax error, or disk write error).
- `ROLLED_BACK`: Patch was applied but reverted due to test regression or user undo.

### `VerificationStatus` Enum (Patcher)

- `CONFIRMED_FIXED`: Re-scan confirmed target finding is completely resolved with no new regressions.
- `UNRESOLVED`: Re-scan shows target finding still present in code.
- `REGRESSED`: Patch introduced one or more new scanner findings.
- `SYNTAX_ERROR`: Patch caused an AST syntax error in the target file.
- `FALLBACK_VERIFIED`: Static verification confirmed replacement snippet present and original removed.

### `MitigatedVulnerability`

**Purpose:** Metrics record representing a successfully verified and mitigated vulnerability.

| Field | Type | Required | Notes |
|---|---|---|---|
| `finding_id` | `str` | Yes | Finding identifier. |
| `cwe` | `Optional[str]` | No | CWE identifier (e.g. `"CWE-78"`). |
| `category` | `Optional[str]` | No | Vulnerability category (e.g. `"command-injection"`). |
| `file` | `str` | Yes | Target file path. |
| `rule_id` | `Optional[str]` | No | Original scanner rule ID. |
| `severity` | `Optional[Severity]` | No | Finding severity. |
| `verification_status` | `VerificationStatus` | Yes | Result of the post-patch verification. |

### `RemediationSessionSummary`

**Purpose:** Aggregate metrics and audit report produced at the conclusion of an interactive remediation session.

| Field | Type | Default | Notes |
|---|---|---|---|
| `total_findings` | `int` | `0` | Number of findings reviewed in session. |
| `patches_applied` | `int` | `0` | Count of patches applied to disk. |
| `patches_rejected` | `int` | `0` | Count of patches skipped or rejected. |
| `vulnerabilities_avoided` | `int` | `0` | Confirmed resolved vulnerabilities. |
| `regressions_detected` | `int` | `0` | Count of patches rolled back due to regressions. |
| `unresolved_count` | `int` | `0` | Findings still present after patch attempt. |
| `modified_files` | `List[str]` | `[]` | List of file paths modified. |
| `avoided_cwes` | `List[str]` | `[]` | Deduplicated list of resolved CWE IDs. |
| `mitigated_findings` | `List[MitigatedVulnerability]` | `[]` | Detailed list of mitigated vulnerabilities. |
| `static_analysis_clean` | `bool` | `True` | True if rescan detected no remaining issues. |

