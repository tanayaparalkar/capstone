"""Rate limiting detection and repository-tailored advisory generator.

Phase 5: If a user has a broken API endpoint or DoS vulnerability that allows
malicious actors or traffic surges to take the server down (CWE-400, CWE-770,
OWASP API4:2023), SentinelAI detects the repository's web framework (FastAPI,
Flask, Django, Express, or generic Python) and provides a customized, drop-in
rate-limiting configuration.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from ..contracts import ScannerFinding


def detect_repository_framework(repo_root: Path) -> str:
    """Detect web framework used in the repository (fastapi, flask, django, express, python)."""
    # 1. Inspect dependency manifests
    req_file = repo_root / "requirements.txt"
    if req_file.exists():
        try:
            text = req_file.read_text(encoding="utf-8", errors="ignore").lower()
            if "fastapi" in text:
                return "fastapi"
            if "flask" in text:
                return "flask"
            if "django" in text:
                return "django"
        except Exception:
            pass

    pyproject = repo_root / "pyproject.toml"
    if pyproject.exists():
        try:
            text = pyproject.read_text(encoding="utf-8", errors="ignore").lower()
            if "fastapi" in text:
                return "fastapi"
            if "flask" in text:
                return "flask"
            if "django" in text:
                return "django"
        except Exception:
            pass

    package_json = repo_root / "package.json"
    if package_json.exists():
        try:
            text = package_json.read_text(encoding="utf-8", errors="ignore").lower()
            if "express" in text:
                return "express"
        except Exception:
            pass

    # 2. Inspect source code files for framework imports
    for py_file in repo_root.glob("**/*.py"):
        if any(p in py_file.parts for p in (".git", ".sentinelai", ".venv", "venv", "__pycache__")):
            continue
        try:
            content = py_file.read_text(encoding="utf-8", errors="ignore")[:3000]
            if "from fastapi" in content or "import fastapi" in content:
                return "fastapi"
            if "from flask" in content or "import flask" in content:
                return "flask"
            if "from django" in content or "import django" in content:
                return "django"
        except Exception:
            pass

    return "python"


def is_rate_limiting_relevant(finding: ScannerFinding) -> bool:
    """Check if finding describes an API endpoint, DoS risk, or lack of rate limiting."""
    cat = (finding.category or "").lower()
    msg = (finding.message or "").lower()
    rule = (finding.rule_id or "").lower()
    cwe = (finding.cwe or "").upper()

    dos_cwes = {"CWE-400", "CWE-770", "CWE-307", "CWE-799", "CWE-20"}
    if cwe in dos_cwes:
        return True

    keywords = (
        "rate-limit",
        "ratelimit",
        "denial of service",
        "dos",
        "resource exhaustion",
        "brute-force",
        "brute force",
        "unrestricted access",
        "take server down",
        "api flood",
        "broken api",
    )
    if any(k in cat or k in msg or k in rule for k in keywords):
        return True

    return False


def get_rate_limiting_advisory(repo_root: Path, finding: Optional[ScannerFinding] = None) -> Dict[str, str]:
    """Generate framework-tailored rate limiting configuration for the user's repository."""
    framework = detect_repository_framework(repo_root)

    if framework == "fastapi":
        return {
            "framework": "FastAPI",
            "package": "slowapi",
            "install": "pip install slowapi",
            "code_example": (
                "from slowapi import Limiter, _rate_limit_exceeded_handler\n"
                "from slowapi.util import get_remote_address\n"
                "from slowapi.errors import RateLimitExceeded\n\n"
                "limiter = Limiter(key_func=get_remote_address)\n"
                "app.state.limiter = limiter\n"
                "app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)\n\n"
                "@app.get('/api/resource')\n"
                "@limiter.limit('60/minute')\n"
                "async def protected_endpoint(request: Request):\n"
                "    return {'status': 'ok'}"
            ),
            "summary": "Attaches slowapi Limiter to prevent API flooding and denial-of-service (CWE-400).",
        }
    elif framework == "flask":
        return {
            "framework": "Flask",
            "package": "Flask-Limiter",
            "install": "pip install Flask-Limiter",
            "code_example": (
                "from flask import Flask\n"
                "from flask_limiter import Limiter\n"
                "from flask_limiter.util import get_remote_address\n\n"
                "app = Flask(__name__)\n"
                "limiter = Limiter(get_remote_address, app=app, default_limits=['200 per day', '50 per hour'])\n\n"
                "@app.route('/api/resource', methods=['POST'])\n"
                "@limiter.limit('10 per minute')\n"
                "def protected_route():\n"
                "    return {'status': 'ok'}"
            ),
            "summary": "Enforces IP-based request throttling with Flask-Limiter to guard against API exhaustion.",
        }
    elif framework == "django":
        return {
            "framework": "Django / DRF",
            "package": "django-ratelimit",
            "install": "pip install django-ratelimit",
            "code_example": (
                "from django_ratelimit.decorators import ratelimit\n\n"
                "@ratelimit(key='ip', rate='30/m', block=True)\n"
                "def my_api_view(request):\n"
                "    return JsonResponse({'status': 'ok'})\n\n"
                "# In Django REST Framework settings.py:\n"
                "REST_FRAMEWORK = {\n"
                "    'DEFAULT_THROTTLE_CLASSES': [\n"
                "        'rest_framework.throttling.AnonRateThrottle',\n"
                "        'rest_framework.throttling.UserRateThrottle',\n"
                "    ],\n"
                "    'DEFAULT_THROTTLE_RATES': {\n"
                "        'anon': '100/day',\n"
                "        'user': '1000/day',\n"
                "    }\n"
                "}"
            ),
            "summary": "Applies django-ratelimit or DRF Throttling to stop brute-force and resource exhaustion.",
        }
    elif framework == "express":
        return {
            "framework": "Express.js",
            "package": "express-rate-limit",
            "install": "npm install express-rate-limit",
            "code_example": (
                "const rateLimit = require('express-rate-limit');\n\n"
                "const apiLimiter = rateLimit({\n"
                "  windowMs: 15 * 60 * 1000, // 15 minutes\n"
                "  max: 100, // Limit each IP to 100 requests per windowMs\n"
                "  message: 'Too many requests from this IP, please try again later.'\n"
                "});\n\n"
                "app.use('/api/', apiLimiter);"
            ),
            "summary": "Uses express-rate-limit middleware to restrict request volume per client IP.",
        }
    else:
        return {
            "framework": "Python",
            "package": "limits / redis",
            "install": "pip install limits",
            "code_example": (
                "from limits import storage, strategies, parse\n\n"
                "backend = storage.MemoryStorage()\n"
                "strategy = strategies.MovingWindowRateLimiter(backend)\n"
                "limit = parse('60/minute')\n\n"
                "def handle_request(client_ip):\n"
                "    if not strategy.hit(limit, client_ip):\n"
                "        raise RuntimeError('Rate limit exceeded (429)')\n"
                "    # Process request"
            ),
            "summary": "Configures token-bucket or memory/Redis rate limiting to prevent denial-of-service (CWE-400).",
        }
