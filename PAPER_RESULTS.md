# SentinelAI — Results (paper draft)

Measurement environment, applying to every figure below unless stated otherwise:
Apple M5 Mac (24 GB RAM, macOS 26.5.2), Python 3.11.15, Semgrep 1.172.0,
Bandit 1.9.4, GitLeaks 8.30.1. AI enrichment used `llama3.1:8b` for generation
and `nomic-embed-text` for embeddings, both served locally by Ollama, with no
cloud API or credential involved. Measurements dated 2026-08-17.

Two independent measurements are reported and must not be conflated: a
**scanner-only n=5** result (§5.1) and an **AI-enriched n=3** result (§5.2.1).
They describe different modes with different cost profiles.

---

## 5.1 Scanner-only performance (n=5)

The benchmark repository, `sentinelai-manual-test`, comprises five scanned files
(four Python modules totalling 57 lines, plus a dependency manifest) containing
seven intentionally introduced vulnerability classes: SQL injection, weak
cryptography, insecure deserialization, command injection (in two forms),
insecure YAML loading, arbitrary code execution, and hardcoded credentials.

SentinelAI reports **17 findings** on this repository — 10 from Bandit, 6 from
Semgrep, 1 from GitLeaks. Detection is fully deterministic: all five runs
produced the identical 17 findings, and only wall-clock time varied.

| Metric | Value |
|---|---|
| Findings | 17 (Bandit 10, Semgrep 6, GitLeaks 1) |
| Severity | 0 Critical / 7 High / 7 Medium / 3 Low |
| Runtime (n=5) | min 3.59 s, median 4.47 s, mean 4.93 s, max 6.55 s |

Isolating each scanner attributes essentially all runtime and all variance to
Semgrep: across three back-to-back invocations on identical input, Semgrep ranged
3.14–5.53 s, while Bandit completed in ≈0.08 s and GitLeaks in ≈0.03 s, each with
negligible spread. The dominant cost is therefore Semgrep's per-invocation
startup and rule-loading overhead rather than any property of the target
repository, and total scan time is effectively independent of repository size at
this scale.

---

## 5.2 AI enrichment

With both models configured, all **17 of 17** findings were enriched successfully
(`ai_enrichment_status: available`), each receiving a generated title,
explanation, impact, and remediation validated against a strict six-field schema.

### 5.2.1 Latency (n=3)

On the 17-finding benchmark, local AI enrichment completed successfully in all
three measured runs. End-to-end latency ranged from 128.7 s to 158.4 s (median
146.1 s; mean 144.4 s), or approximately 8.5 s per finding. The current pipeline
is sequential and issues one local generation request per finding, with no
batching or concurrency. These results characterize a local developer-triage
workflow rather than a low-latency CI gate.

| Run | Exit | Elapsed | Scanner | AI-enriched | Status | Verified / Rejected | Confidence mean |
|---|---|---|---|---|---|---|---|
| 1 | 0 | 158.4 s | 17 | 17 | available | 6 / 11 | 0.8542 |
| 2 | 0 | 146.1 s | 17 | 17 | available | 7 / 10 | 0.8542 |
| 3 | 0 | 128.7 s | 17 | 17 | available | 7 / 10 | 0.8542 |

| Statistic | AI-enriched, n=3 |
|---|---:|
| Minimum | 128.7 s |
| Median | 146.1 s |
| Mean | 144.4 s |
| Maximum | 158.4 s |
| Mean per finding | ≈8.5 s |
| Completion reliability | 3/3 runs, 17/17 enrichments each |

Timings are end-to-end wall clock and include local embedding of the knowledge
base, retrieval, and sequential generation.

### 5.2.2 Run-to-run variance

Verification status varied for one of 17 findings across the three runs, despite
temperature 0, because a generic Semgrep category (`security`) is verified only
when the generated text contains that literal token. Confidence remained
identical because it depends on deterministic retrieval relevance and evidence
presence rather than LLM output.

Sixteen of seventeen verdicts were stable across all three runs. This variance is
the reason the verifier improvement in §5.3 was measured by a controlled
substitution over fixed text rather than by comparing two live runs, which would
have drifted by ±1 finding on this benchmark.

---

## 5.3 Verifier improvement (controlled)

Category normalization originally replaced only hyphens, leaving Bandit's
underscore-joined test names (for example
`subprocess_popen_with_shell_equals_true`) as single unsplittable tokens that no
generated explanation could contain. Extending normalization to split on
underscores and drop generic security nouns addressed this.

Because each benchmark run re-invokes the LLM, a naive before/after comparison
cannot separate the effect of the change from run-to-run model variance. We
therefore reimplemented the original normalization verbatim and re-ran it against
the **already-generated explanations** from a completed run, so that only the
verifier logic differs:

| Verifier logic | Verified (identical LLM text) |
|---|---|
| Original (hyphen-only) | 1 / 17 |
| Revised (hyphen + underscore + stop-word removal) | 6 / 17 |
| **Improvement** | **+5 (verifier change alone)** |

Holding the generated explanations fixed and swapping only the verifier logic,
verification rose from 1/17 to 6/17 — a +5 improvement attributable to the
normalization change alone, independent of run-to-run model variance. The
recomputation reproduces the recorded verdicts exactly, confirming the method.
The five findings that changed verdict are precisely the five with splittable
underscored categories: `hardcoded_sql_expressions`, `start_process_with_a_shell`,
`subprocess_popen_with_shell_equals_true`, `yaml_load`, and
`hardcoded_password_string`.

An incidental finding supports a separate correctness fix: a pre-fix enrichment
run contained 17 findings under only **12 unique `finding_id`s**, because
anonymous Semgrep invocations return the literal placeholder `"requires login"`
in the fingerprint field of every result. Since `finding_id` is the key joining
scanner findings to their enrichment across the statistics engine and all four
report renderers, this collision would silently collapse entries rather than
fail. After the fix, all 17 findings carry distinct identifiers.

---

## 5.4 Remaining rejections: root cause

The 11 remaining rejections are not attributable to verifier logic. In each case
the finding's `category` field carries a tool-internal identifier rather than a
vulnerability descriptor, so no correct explanation could contain it:

| Category | Count | Origin |
|---|---|---|
| `blacklist` | 4 | Bandit's `test_name` for B403/B404/B301/B307 (import `pickle`, import `subprocess`, `pickle.loads`, `eval`) |
| `hashlib` | 1 | Bandit's `test_name` for B324 — a module name; explanations say "MD5" |
| `security` | 6 | Semgrep's generic `metadata.category`, shared by most rules |

No correct explanation could contain these identifiers, so these 11 findings
cannot verify under the current category mapping. Raising the figure requires
improving category derivation in the scanner wrappers — for instance mapping
Bandit's `test_id` to a descriptive name instead of using `test_name` — and not
further changes to the verification heuristic.

---

## 5.5 Robustness

SentinelAI was hardened for local-model reliability through two mechanisms.
First, enrichment failures are handled per finding: successful enrichments are
retained, partial completion is reported through the existing enrichment-status
contract, and an all-failure condition remains a provider error. Second, Ollama
generation and embedding requests retry once after a fixed two-second delay only
for transport-level or transient server failures; malformed model output, schema
failures, configuration errors, and deterministic HTTP failures are not retried.
The final system passed 631 tests, with two benchmark-marked tests deselected by
default.

A healthy request continues to make exactly one HTTP call per generation and per
embedding batch, so the retry mechanism adds no cost to the normal path. When
Ollama is unavailable the CLI reports a single actionable error and exits with
code 3, distinct from the exit code used for security-gate failures.

---

## 5.6 Self-scan

SentinelAI passes its own security gate. Scanning the CLI's source tree with
`--fail-on high` produces **1135 findings** (Bandit 1132, Semgrep 3) with **0
Critical and 0 High**, exiting **0**. Of these, 1125 originate in the test suite,
and 1096 of the 1110 Low-severity findings are Bandit's `B101` (`assert_used`)
flagging pytest assertions — expected noise for a Python test suite rather than a
code-quality signal. A single prior High-severity finding was a false positive on
a masked credential placeholder in a bundled test fixture; shortening the
placeholder so it no longer matches Semgrep's AWS access-key pattern resolved it
without affecting any test.

The absolute count is a snapshot that tracks test-suite size: because most
findings are `B101` on pytest assertions, adding tests raises it (it moved from
1089 to 1135 when this project's own test count went from 581 to 631). The
figures that carry meaning, and that are stable, are **0 Critical and 0 High**
and the resulting **exit code 0**.

---

## Reproducibility

Every figure above is reproducible from the committed repository. See
`sentinelai-manual-test/README.md` for the exact commands, and `POLISH_REPORT.md`
for the per-change record behind §5.3–§5.5.
