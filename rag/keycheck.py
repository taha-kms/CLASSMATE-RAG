"""
Checking that a stored provider key is accepted (#147).

Saving a key in Settings or with `rag config set` only writes it down. A typo
then shows up on the first question, as a provider error in the middle of an
answer. This asks the provider's model-list endpoint instead, which is read
only and free, so the mistake shows up when the key is entered.

Rules that shape the code:

- The key tested is the one the application would use: the environment
  before the credential store, as in #48. The result says which one it was,
  so "valid" cannot be about a key other than the one just typed.
- Only the three known providers can be asked about, and each has a fixed
  host. OPENAI_BASE_URL is the one exception and is already how that
  provider is reached; it is a local-only setting.
- The key goes in a header, never a URL, so it cannot end up in a proxy log
  or an exception message.
- Redirects are refused. urllib would resend the key to wherever a redirect
  points.
- The result is one of a fixed set of outcomes. The provider's response body
  and any exception text are discarded, never copied into the result.
"""

from __future__ import annotations

import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Literal

from rag.secrets import load_secrets, mask, safe_url

Provider = Literal["anthropic", "openai", "gemini"]
PROVIDERS: tuple[str, ...] = ("anthropic", "openai", "gemini")

Outcome = Literal["valid", "rejected", "rate_limited", "unreachable", "unexpected", "missing"]

TIMEOUT_SECONDS = 8.0

#: Environment variables that can hold each provider's key, first match wins.
_KEY_NAMES: dict[str, tuple[str, ...]] = {
    "anthropic": ("ANTHROPIC_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
}

_ANTHROPIC_URL = "https://api.anthropic.com/v1/models?limit=1"
_OPENAI_DEFAULT_BASE = "https://api.openai.com/v1"
_GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models?pageSize=1"

#: What each outcome means, in words that are safe to show. Never built from
#: anything the provider sent back.
MESSAGES: dict[str, str] = {
    "valid": "The provider accepted the key.",
    "rejected": "The provider rejected the key. Check it was copied whole and has not been revoked.",
    "rate_limited": "The provider is rate limiting this key. The key was recognised.",
    "unreachable": "Could not reach the provider: check the network connection and any proxy.",
    "unexpected": "The provider answered in a way this check does not recognise.",
    "missing": "No key is set for this provider.",
}


@dataclass(frozen=True)
class KeyCheck:
    provider: str
    outcome: str
    #: "environment" or "stored", or "" when there is no key.
    source: str
    #: Last four characters only, from rag.secrets.mask.
    masked: str
    #: Where the request went, with credentials stripped. Matters for OpenAI,
    #: where a local server on OPENAI_BASE_URL may accept any key.
    target: str
    #: HTTP status when the provider answered, else None.
    status: int | None

    @property
    def message(self) -> str:
        return MESSAGES[self.outcome]

    def as_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "outcome": self.outcome,
            "ok": self.outcome == "valid",
            "message": self.message,
            "source": self.source,
            "masked": self.masked,
            "target": self.target,
            "status": self.status,
        }


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
        return None


def _default_opener() -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(_NoRedirect)


def _effective_key(provider: str) -> tuple[str | None, str]:
    """The key the application would use, and where it came from."""
    stored = load_secrets()
    for name in _KEY_NAMES[provider]:
        value = os.getenv(name)
        if value:
            return value, "environment"
    for name in _KEY_NAMES[provider]:
        value = stored.get(name)
        if value:
            return value, "stored"
    return None, ""


def _openai_base() -> str:
    from rag.config import load_config

    return (load_config().openai_base_url or _OPENAI_DEFAULT_BASE).rstrip("/")


def classify(provider: str, status: int) -> str:
    """Map an HTTP status to an outcome. The only place statuses are interpreted."""
    if 200 <= status < 300:
        return "valid"
    if status in (401, 403):
        return "rejected"
    # Google answers 400 "API key not valid" for a bad key rather than 401.
    if provider == "gemini" and status == 400:
        return "rejected"
    if status == 429:
        return "rate_limited"
    return "unexpected"


def check_key(provider: str, *, opener: urllib.request.OpenerDirector | Any | None = None) -> KeyCheck:
    """
    Ask `provider` whether its key is accepted.

    `opener` is a seam for tests, which must make no network calls.
    """
    if provider not in PROVIDERS:
        raise ValueError(f"unknown provider {provider!r}; expected one of {', '.join(PROVIDERS)}")

    key, source = _effective_key(provider)

    if provider == "anthropic":
        url = _ANTHROPIC_URL
        headers = {"x-api-key": key or "", "anthropic-version": "2023-06-01"}
    elif provider == "openai":
        url = f"{_openai_base()}/models"
        headers = {"Authorization": f"Bearer {key or ''}"}
    else:
        url = _GEMINI_URL
        headers = {"x-goog-api-key": key or ""}

    target = safe_url(url.split("?", 1)[0])

    if not key:
        return KeyCheck(provider, "missing", "", "", target, None)

    request = urllib.request.Request(url, headers={**headers, "Accept": "application/json"}, method="GET")
    status: int | None
    try:
        with (opener or _default_opener()).open(request, timeout=TIMEOUT_SECONDS) as response:
            status = int(response.status)
    except urllib.error.HTTPError as e:
        # An answer, even a refusal: the body is closed unread.
        status = int(e.code)
        e.close()
    except Exception:  # noqa: BLE001 - every network failure is "unreachable"; the text is discarded
        return KeyCheck(provider, "unreachable", source, mask(key), target, None)

    return KeyCheck(provider, classify(provider, status), source, mask(key), target, status)
