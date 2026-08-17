# sentinelai-manual-test

A small, intentionally vulnerable Python codebase used as SentinelAI's
benchmark/demo repository. It exists purely to be scanned — every
vulnerability in it is deliberate, and none of this code is meant to run.

It lives outside `sentinelai-cli/` on purpose: SentinelAI's own CI scans
`sentinelai-cli/.` as a live demonstration of the security-gate workflow,
and this repo is full of intentional high-severity findings by design —
nesting it inside the scanned tree would fail that gate for the wrong
reason.

## Structure

```
sentinelai-manual-test/
├── app/
│   ├── db.py            SQL injection (string-concatenated query)
│   ├── crypto_utils.py   Weak cryptography (MD5 for password hashing)
│   └── tasks.py          Insecure deserialization (pickle.loads)
│                         Command injection (os.system)
│                         Command injection (subprocess, shell=True)
│                         Insecure YAML loading (yaml.load, no Loader)
│                         Arbitrary code execution (eval)
├── config/
│   └── settings.py       Hardcoded credentials (API key, password)
└── requirements.txt      Pinned dependencies (not scanned for CVEs today -
                           SentinelAI has no dependency-vulnerability
                           scanner wired up; see main project README)
```

5 files that SentinelAI scans (4 Python source files, 57 LOC, plus one
`requirements.txt`), 6 files total in the directory once this README is
counted too. `README.md` is not itself scanned meaningfully - GitLeaks
walks it but finds nothing, and it has no `.py` extension so Bandit and
Semgrep's Python-only rules never touch it.

## Expected detections (verified against a real scan)

| Vulnerability | Bandit | Semgrep | GitLeaks |
|---|:---:|:---:|:---:|
| SQL injection (`db.py`) | ✅ B608 | ✅ sqlalchemy-execute-raw-query | — |
| Weak crypto / MD5 (`crypto_utils.py`) | ✅ B324 | ✅ md5-used-as-password | — |
| Insecure deserialization (`tasks.py`) | ✅ B301 | ✅ avoid-pickle | — |
| Command injection, `os.system` (`tasks.py`) | ✅ B605 | — | — |
| Command injection, `shell=True` (`tasks.py`) | ✅ B602 | ✅ subprocess-shell-true | — |
| Insecure YAML load (`tasks.py`) | ✅ B506 | — | — |
| `eval()` (`tasks.py`) | ✅ B307 | ✅ eval-detected | — |
| Hardcoded API key (`settings.py:14`) | — | ✅ detected-generic-api-key | ✅ generic-api-key |
| Hardcoded password (`settings.py:15`) | ✅ B105 | — | — |

**17 raw findings total** (10 Bandit + 6 Semgrep + 1 GitLeaks). SentinelAI
does not deduplicate across scanners, so this is also the exact count
`sentinelai scan` reports. Note the two credentials in `settings.py` are
each caught by a *different* scanner and missed by the others — a small,
honest illustration of why running three scanners finds more than
running any one of them.

Both credential values in `settings.py` are unmistakably fake, not just
randomly generated: the "API key" is a literal `0-9a-f` counting sequence
repeated three times, and the "password" says outright that it isn't
real. Neither resembles a real cloud/SaaS vendor's key format (no
Amazon-, GitHub-, Slack-, or Stripe-style prefix). Verified this doesn't weaken
detection before settling on it — several more realistic-looking
placeholder strings (e.g. anything containing the word "EXAMPLE" or
"FAKE" inside the value itself) were silently dropped by GitLeaks' own
allowlist and produced *zero* findings; the counting-sequence pattern is
the one that stayed both obviously synthetic and still detected by all
three scanners' relevant rules.

Not caught by anything: `os.system` isn't flagged by Semgrep's default
(no `--config`) ruleset, and `yaml.load` isn't flagged by Semgrep's
default ruleset either. Both are real gaps in Semgrep's out-of-the-box
free rules, not a SentinelAI bug — a more complete Semgrep ruleset
(`--config=auto` or `p/security-audit`) would likely catch both, but
SentinelAI does not currently pass a `--config` (see
`sentinelai/scanners/semgrep.py`).

## Running the benchmark

From the repository root:

```bash
# Scanner-only (the default - no AI configuration needed)
sentinelai scan sentinelai-manual-test --format json --output /tmp/bench-result.json

# Time it
time sentinelai scan sentinelai-manual-test --format json --output /tmp/bench-result.json

# Inspect what was found
python3 -c "import json; d=json.load(open('/tmp/bench-result.json')); print(len(d['findings']['scanner']), 'findings')"

# Human-readable terminal view
sentinelai scan sentinelai-manual-test

# Optional: AI-enriched (requires Ollama + two models - see sentinelai-cli/README.md)
export SENTINELAI_AI_LLM_MODEL=llama3.1:8b
export SENTINELAI_AI_EMBEDDING_MODEL=nomic-embed-text
sentinelai scan sentinelai-manual-test --format json --output /tmp/bench-result-ai.json
```

`sentinelai-cli/tests/test_performance_basic.py` runs a version of the
first command automatically as a regression guard.
