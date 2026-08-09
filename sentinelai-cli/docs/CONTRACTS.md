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
  finishes).

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

**Location:** `sentinelai/providers/base.py`, `sentinelai/providers/mock_provider.py`

```python
class FindingsProvider(ABC):
    @abstractmethod
    def get_scan_result(self, repo_path: str, mode: ScanMode = ScanMode.STANDARD) -> ScanResult:
        ...
```

Viraj's CLI depends on this abstract interface, **not** on Nithanth's
backend directly. Today `MockFindingsProvider` is the only
implementation — it reads a bundled JSON fixture and returns a
`ScanResult` with `ai_findings` empty (the mock has no AI layer).

Once Nithanth's backend is ready, a second implementation (e.g. an
`ApiFindingsProvider` calling his `/scan` endpoint, or a call into his
orchestration function directly) is a drop-in replacement: it returns the
same `ScanResult` shape, and nothing in the CLI, reporting, or statistics
code needs to change.

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
