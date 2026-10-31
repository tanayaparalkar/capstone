"""
Gemini on Vertex AI: the generateContent endpoint.

Vertex AI, not AI Studio. They are different services with different
data handling, and this provider only ever calls
{location}-aiplatform.googleapis.com.

Credentials come from GOOGLE_APPLICATION_CREDENTIALS, the standard
Application Default Credentials variable, which names a service-account
JSON file. Turning that into a short-lived OAuth access token needs an
RS256 signature, so google-auth is imported lazily on first use: it is
only required when Gemini actually runs, and a missing install is a
CONFIGURATION ProviderError rather than an import failure for everyone
else. The access token is fetched per call (google-auth caches and
refreshes it), sent only in the Authorization header, and never
logged, repr'd or put in an error. `token_provider` exists so tests and
other credential sources can supply a token without google-auth.

Project and location come from the constructor, else GOOGLE_CLOUD_PROJECT
/ GOOGLE_CLOUD_LOCATION (location defaults to us-central1; "global" uses
the un-prefixed host).

retains_data is None, meaning "not verified", deliberately.
TODO(retains_data): read Google Cloud's "Vertex AI generative AI and
zero data retention" page
(https://cloud.google.com/vertex-ai/generative-ai/docs/data-governance),
the Service Specific Terms for Vertex AI, and the Cloud Data Processing
Addendum, and record which project settings the answer depends on
(abuse-monitoring prompt logging and in-memory caching are covered
there, and may need an exception or opt-out on the project in use).
These terms do not apply to Gemini via AI Studio; do not copy a value
from AI Studio documentation.
"""
import logging
import os
import urllib.parse
from typing import Callable, ClassVar, Optional

from ._http import optional_count, post_json, validate_timeout
from .base import LLMProvider, LLMResponse
from .errors import ProviderError, ProviderErrorKind

logger = logging.getLogger("sentinelai")

CREDENTIALS_ENV = "GOOGLE_APPLICATION_CREDENTIALS"
PROJECT_ENV = "GOOGLE_CLOUD_PROJECT"
LOCATION_ENV = "GOOGLE_CLOUD_LOCATION"
DEFAULT_LOCATION = "us-central1"
DEFAULT_TIMEOUT_S = 60.0
_SCOPES = ("https://www.googleapis.com/auth/cloud-platform",)

TokenProvider = Callable[[], str]


class GeminiProvider(LLMProvider):
    name: ClassVar[str] = "gemini"
    is_local: ClassVar[bool] = False
    retains_data: ClassVar[Optional[bool]] = None
    retention_basis: ClassVar[str] = "unverified_project_data_governance"
    retention_policy_note: ClassVar[str] = (
        "Google Cloud Vertex AI service-specific data-handling terms distinguish model-training "
        "restrictions from service and abuse-monitoring storage depending on project settings; "
        "actual project configuration cannot be verified at runtime."
    )

    def __init__(
        self,
        model: str,
        *,
        project: Optional[str] = None,
        location: Optional[str] = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        token_provider: Optional[TokenProvider] = None,
    ) -> None:
        if not model:
            raise ProviderError(self.name, ProviderErrorKind.CONFIGURATION, "no model configured")
        if token_provider is None:
            credentials_path = os.environ.get(CREDENTIALS_ENV, "").strip()
            if not credentials_path:
                raise ProviderError(self.name, ProviderErrorKind.CONFIGURATION, f"{CREDENTIALS_ENV} is not set")
            token_provider = _ServiceAccountTokens(self.name, credentials_path)
        project = project or os.environ.get(PROJECT_ENV, "").strip()
        if not project:
            raise ProviderError(self.name, ProviderErrorKind.CONFIGURATION, f"{PROJECT_ENV} is not set")
        location = location or os.environ.get(LOCATION_ENV, "").strip() or DEFAULT_LOCATION

        self.model = model
        self.project = project
        self.location = location
        self._tokens = token_provider
        self._timeout_s = validate_timeout(timeout_s)
        host = "aiplatform.googleapis.com" if location == "global" else f"{location}-aiplatform.googleapis.com"
        quote = lambda part: urllib.parse.quote(part, safe="")  # noqa: E731
        self._url = (
            f"https://{host}/v1/projects/{quote(project)}/locations/{quote(location)}"
            f"/publishers/google/models/{quote(model)}:generateContent"
        )

    def __repr__(self) -> str:
        return f"GeminiProvider(model={self.model!r}, project={self.project!r}, location={self.location!r})"

    def complete(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.1,
    ) -> LLMResponse:
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"maxOutputTokens": max_tokens, "temperature": temperature},
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        headers = {"Authorization": f"Bearer {self._access_token()}"}
        body, latency_ms = post_json(self.name, self._url, payload, headers, self._timeout_s)
        text = _candidate_text(body)
        if text is None:
            raise ProviderError(self.name, ProviderErrorKind.INVALID_RESPONSE, _no_text_reason(body))
        usage = body.get("usageMetadata") if isinstance(body.get("usageMetadata"), dict) else {}
        logger.debug("gemini completion: model=%s latency_ms=%.0f", self.model, latency_ms)
        return LLMResponse(
            text=text,
            model=body.get("modelVersion") if isinstance(body.get("modelVersion"), str) else self.model,
            prompt_tokens=optional_count(usage.get("promptTokenCount")),
            completion_tokens=optional_count(usage.get("candidatesTokenCount")),
            latency_ms=latency_ms,
        )

    def _access_token(self) -> str:
        try:
            token = self._tokens()
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(
                self.name, ProviderErrorKind.AUTH, f"could not obtain an access token ({type(exc).__name__})"
            ) from None
        if not isinstance(token, str) or not token:
            raise ProviderError(self.name, ProviderErrorKind.AUTH, "credential source returned no access token")
        return token


def _candidate_text(body: object) -> Optional[str]:
    """Concatenated text parts of the first candidate, excluding thought parts; None if there are none."""
    try:
        parts = body["candidates"][0]["content"]["parts"]  # type: ignore[index]
    except (KeyError, IndexError, TypeError):
        return None
    if not isinstance(parts, list):
        return None
    texts = [p["text"] for p in parts if isinstance(p, dict) and isinstance(p.get("text"), str) and not p.get("thought")]
    return "".join(texts) if texts else None


def _no_text_reason(body: object) -> str:
    # blockReason / finishReason are documented enum values (SAFETY, MAX_TOKENS, ...), never content.
    if isinstance(body, dict):
        feedback = body.get("promptFeedback")
        if isinstance(feedback, dict) and isinstance(feedback.get("blockReason"), str):
            return f"prompt blocked ({feedback['blockReason'][:40]})"
        candidates = body.get("candidates")
        if isinstance(candidates, list) and candidates and isinstance(candidates[0], dict):
            reason = candidates[0].get("finishReason")
            if isinstance(reason, str):
                return f"no text in first candidate (finishReason {reason[:40]})"
    return "no text in first candidate"


class _ServiceAccountTokens:
    """Access tokens from a service-account file via google-auth, which caches and refreshes them."""

    def __init__(self, provider: str, credentials_path: str) -> None:
        self._provider = provider
        self._path = credentials_path
        self._credentials = None
        self._request = None

    def __repr__(self) -> str:
        return "_ServiceAccountTokens()"

    def __call__(self) -> str:
        try:
            from google.auth.transport.requests import Request
            from google.oauth2 import service_account
        except ImportError:
            raise ProviderError(
                self._provider, ProviderErrorKind.CONFIGURATION, "google-auth is not installed (pip install 'google-auth[requests]')"
            ) from None
        if self._credentials is None:
            try:
                self._credentials = service_account.Credentials.from_service_account_file(self._path, scopes=_SCOPES)
            except (OSError, ValueError, KeyError):
                raise ProviderError(
                    self._provider, ProviderErrorKind.CONFIGURATION, f"{CREDENTIALS_ENV} is not a readable service-account file"
                ) from None
            self._request = Request()
        if not self._credentials.valid:
            self._credentials.refresh(self._request)
        return self._credentials.token
