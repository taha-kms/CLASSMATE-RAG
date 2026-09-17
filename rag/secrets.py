"""
Storing and redacting provider credentials.

The application has no authentication by design and binds to loopback, so
the protection around a stored API key is that nothing outside the machine
can reach it. That only holds if the key never leaves the machine by
accident, which is what this module is for.

Three rules, each of which has been a real incident somewhere:

- a stored secret is never returned in full. Reads give a masked form, so a
  key cannot end up in a browser history, a screenshot, a bug report or a
  log by someone simply asking the settings endpoint what is configured
- secrets live in a file outside the image, so `docker rm` does not lose
  them and no image layer ever contains one
- the environment wins over the file, matching #48, so a key exported for
  one command is not silently overridden by a stale stored one
"""

from __future__ import annotations

import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path

from rag.config import resolve_data_path

#: Settings that hold credentials. Anything listed here is masked on read
#: and scrubbed from log output.
SECRET_KEYS: tuple[str, ...] = (
    "HF_TOKEN",
    "HUGGINGFACE_HUB_TOKEN",
    "CLASSMATE_RAG_HF_TOKEN",
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
)


def is_secret(key: str) -> bool:
    """
    Whether a setting holds a credential.

    Matches the known names plus anything that looks like one, so a
    provider added later is masked before someone remembers to list it.
    """
    upper = key.upper()
    if upper in SECRET_KEYS:
        return True
    return any(marker in upper for marker in ("TOKEN", "API_KEY", "SECRET", "PASSWORD"))


def mask(value: str | None) -> str:
    """
    A form safe to show, log or put in a bug report.

    Keeps the last four characters so someone can tell which key is
    configured without the value being usable.
    """
    if not value:
        return ""
    if len(value) <= 8:
        # Too short to reveal any of: a four-character tail would be most
        # of it.
        return "*" * len(value)
    return f"{'*' * 8}{value[-4:]}"


def secrets_path() -> Path:
    """
    Where credentials are stored.

    Inside the project rather than the image, so a container can mount it
    and `docker rm` does not lose it. Deliberately not .env: that file is
    hand-edited and copied from .env.example, and mixing generated
    credentials into it makes both harder to reason about.
    """
    return resolve_data_path("./.secrets.env")


def _parse(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def load_secrets() -> dict[str, str]:
    """Every stored credential, in full. For applying, never for display."""
    path = secrets_path()
    if not path.is_file():
        return {}
    return _parse(path.read_text(encoding="utf-8"))


def read_secret(key: str) -> str | None:
    """
    One credential, environment first.

    Matches the precedence in #48: a variable exported in the shell beats
    the stored file, so a one-off override behaves as expected.
    """
    from_env = os.getenv(key)
    if from_env:
        return from_env
    return load_secrets().get(key) or None


def store_secret(key: str, value: str) -> None:
    """
    Save a credential, replacing any existing one.

    The file is written 0600. A world-readable file holding an API key is
    the kind of thing nobody notices until it matters.
    """
    path = secrets_path()
    values = load_secrets()

    if value:
        values[key] = value
    else:
        values.pop(key, None)

    body = "\n".join(
        [
            "# Credentials written by `rag config set`.",
            "# Not for hand editing; see .env for ordinary settings.",
            "# Anything exported in your shell takes precedence over this.",
            *(f"{k}={v}" for k, v in sorted(values.items())),
            "",
        ]
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    # Create with restrictive permissions rather than fixing them after,
    # which would leave a window where the file is readable.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, stat.S_IRUSR | stat.S_IWUSR)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(body)
    os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)


def delete_secret(key: str) -> None:
    store_secret(key, "")


@dataclass(frozen=True)
class SecretStatus:
    """What is configured, without saying what it is."""

    key: str
    configured: bool
    source: str  # "environment", "file" or "unset"
    masked: str


def secret_status(key: str) -> SecretStatus:
    from_env = os.getenv(key)
    if from_env:
        return SecretStatus(key, True, "environment", mask(from_env))
    stored = load_secrets().get(key)
    if stored:
        return SecretStatus(key, True, "file", mask(stored))
    return SecretStatus(key, False, "unset", "")


def all_secret_status() -> list[SecretStatus]:
    return [secret_status(k) for k in SECRET_KEYS]


#: Matches the common API-key shapes so a value can be scrubbed from text
#: even when it did not arrive through a setting we know the name of.
_KEY_SHAPES = re.compile(
    r"\b("
    r"sk-ant-[A-Za-z0-9_\-]{8,}"
    r"|sk-[A-Za-z0-9_\-]{16,}"
    r"|hf_[A-Za-z0-9]{8,}"
    r"|AIza[A-Za-z0-9_\-]{8,}"
    r")\b"
)


def redact(text: str) -> str:
    """
    Remove anything key-shaped from text before it is logged or displayed.

    A backstop for the paths that do not go through secret_status: an
    exception message from a provider SDK, a command echoed into a log.
    """
    if not text:
        return text

    redacted = _KEY_SHAPES.sub(lambda m: mask(m.group(0)), text)

    # Also scrub the exact values that are configured, which catches keys
    # whose shape is not known here.
    for value in load_secrets().values():
        if value and len(value) > 8:
            redacted = redacted.replace(value, mask(value))
    return redacted
