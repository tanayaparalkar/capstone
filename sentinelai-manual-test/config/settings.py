"""Application configuration - hardcoded credentials are intentional for benchmarking.

Both values below are unmistakably fake, not merely random: API_KEY is a
literal 0-9/a-f counting sequence repeated three times (no real credential
generator would ever emit this), and DB_PASSWORD says outright that it is
not real. Neither resembles any real cloud/SaaS vendor's key format
(no Amazon-, GitHub-, Slack-, or Stripe-style prefix) and neither is a
plausible live secret.
They still trigger the same scanner rules a realistic-looking hardcoded
credential would (verified against real Bandit/Semgrep/GitLeaks runs -
see ../README.md), which is all this fixture needs from them.
"""
DEBUG = True
API_KEY = "0123456789abcdef0123456789abcdef0123456789abcdef"
DB_PASSWORD = "NotARealPassword-DoNotUse-Benchmark123"
