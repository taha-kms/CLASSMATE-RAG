"""
Reading and changing configuration without editing ``.env`` by hand (#96).

Three things this module is careful about, each of which a settings page can
get wrong in a way that looks fine:

- **Precedence stays visible.** The shell beats ``.env`` beats the defaults
  (#48). If ``LOG_LEVEL`` is exported in the shell and the file says
  something else, saving a new value to the file changes nothing, and a page
  that reports "saved" without saying so is lying. Every result says where the
  effective value came from, and a change that cannot take effect says why.
- **Values are checked before they are written.** The loaders in
  ``rag.config`` fall back to the default on an unparseable number, so a typo
  would otherwise be accepted and then silently ignored.
- **Credentials never touch ``.env``.** They go to the credential store
  (``rag.secrets``), and are never returned in full.

The registry below is what ``rag config``, ``/settings`` and the web page all
read, so a new setting is added once.
"""

from __future__ import annotations

import os
import re
import stat
import tempfile
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from dotenv import dotenv_values

from rag.bootstrap import project_root
from rag.secrets import delete_secret, secret_status, store_secret

Kind = Literal["str", "int", "float", "bool", "choice", "secret"]
Source = Literal["environment", "file", "default", "stored", "unset"]


class SettingError(ValueError):
    """A change that cannot be made, with a message fit to show the user."""

    def __init__(self, key: str, message: str) -> None:
        super().__init__(f"{key}: {message}")
        self.key = key
        self.message = message


@dataclass(frozen=True)
class Setting:
    key: str
    kind: Kind
    default: str
    group: str
    description: str
    #: True when the running process set something up once from this value, so
    #: a saved change only applies after a restart. False means the next
    #: question or ingest reads the new value.
    restart: bool = False
    choices: tuple[str, ...] = ()
    minimum: float | None = None
    maximum: float | None = None
    #: True when the setting decides where credentials or corpus text are sent.
    #: These can be changed from the command line on this machine but not over
    #: HTTP, because the API has no authentication: anyone who could reach the
    #: port could otherwise point a stored API key at a server of their choosing.
    local_only: bool = False


_MODEL = "Answers and models"
_RETRIEVAL = "Retrieval"
_INGEST = "Ingestion"
_STORAGE = "Storage"
_ROUTING = "Subject routing"
_CREDS = "Credentials"

REGISTRY: tuple[Setting, ...] = (
    # -- answers and models
    Setting(
        "LLM_PROVIDER",
        "choice",
        "llama_cpp",
        _MODEL,
        "Which provider generates answers. llama_cpp runs locally; the others send your question and the "
        "retrieved text to that provider.",
        choices=("llama_cpp", "anthropic", "openai", "gemini"),
    ),
    Setting("ANTHROPIC_MODEL", "str", "", _MODEL, "Model for the anthropic provider. Empty uses the default."),
    Setting("OPENAI_MODEL", "str", "", _MODEL, "Model for the openai provider. Empty uses the default."),
    Setting(
        "OPENAI_BASE_URL",
        "str",
        "",
        _MODEL,
        "Point the openai adapter at another server speaking the same protocol.",
        local_only=True,
    ),
    Setting("GEMINI_MODEL", "str", "", _MODEL, "Model for the gemini provider. Empty uses the default."),
    Setting(
        "MODEL_PROFILE",
        "choice",
        "custom",
        _MODEL,
        "Sized model set: light, balanced, heavy, or custom to use the paths below.",
        restart=True,
        choices=("light", "balanced", "heavy", "custom"),
    ),
    Setting(
        "LLM_MODEL_PATH", "str", "./models/Llama-3.1-8B-Instruct.Q4_K_M.gguf", _MODEL, "Local .gguf file.", restart=True
    ),
    Setting(
        "LLAMA_GPU_LAYERS",
        "int",
        "0",
        _MODEL,
        "Layers to offload to the GPU. 0 is CPU only, -1 is all.",
        restart=True,
        minimum=-1,
        maximum=999,
    ),
    Setting(
        "EMBEDDING_MODEL_NAME",
        "str",
        "intfloat/multilingual-e5-base",
        _MODEL,
        "Embedding model. Changing it makes the existing index unreadable until you run `rag rebuild`.",
        restart=True,
    ),
    # -- retrieval
    Setting("USE_HYBRID", "bool", "true", _RETRIEVAL, "Combine keyword and vector results."),
    Setting("K_VECTOR", "int", "8", _RETRIEVAL, "Candidates fetched from the vector store.", minimum=1, maximum=100),
    Setting("K_BM25", "int", "8", _RETRIEVAL, "Candidates fetched from the keyword index.", minimum=1, maximum=100),
    Setting("ENABLE_NEIGHBOR_EXPANSION", "bool", "true", _RETRIEVAL, "Pull in chunks adjacent to each hit."),
    Setting(
        "NEIGHBOR_RADIUS", "int", "1", _RETRIEVAL, "Neighbouring chunks to pull in either side.", minimum=0, maximum=10
    ),
    Setting(
        "DOC_DIVERSITY_CAP",
        "int",
        "3",
        _RETRIEVAL,
        "Most chunks kept from any single document.",
        minimum=1,
        maximum=50,
    ),
    Setting(
        "DEFAULT_LANGUAGE",
        "choice",
        "auto",
        _RETRIEVAL,
        "Answer language: en, it, or auto to follow the question.",
        choices=("auto", "en", "it"),
    ),
    Setting("STRICT_CITATIONS", "bool", "false", _RETRIEVAL, "Clean up [n] markers after generation."),
    Setting("APPEND_SOURCES_BLOCK", "bool", "false", _RETRIEVAL, "Append a Sources list when strict citations is on."),
    Setting(
        "TRANSLATE_ON_MISS", "bool", "false", _RETRIEVAL, "Translate an answer that came back in the wrong language."
    ),
    Setting(
        "LOG_LEVEL",
        "choice",
        "INFO",
        _RETRIEVAL,
        "Verbosity on stderr.",
        restart=True,
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
    ),
    # -- ingestion
    Setting("CHUNK_SIZE", "int", "1000", _INGEST, "Characters per chunk.", minimum=100, maximum=8000),
    Setting("CHUNK_OVERLAP", "int", "150", _INGEST, "Overlap between neighbouring chunks.", minimum=0, maximum=2000),
    Setting(
        "INGEST_THREADS",
        "int",
        "0",
        _INGEST,
        "Threads used to chunk pages. 0 means half your cores.",
        minimum=0,
        maximum=256,
    ),
    Setting("DEDUP_CHUNKS", "bool", "false", _INGEST, "Drop near-duplicate chunks at ingest time."),
    Setting(
        "DEDUP_THRESHOLD",
        "float",
        "0.92",
        _INGEST,
        "Similarity above which two chunks count as duplicates.",
        minimum=0,
        maximum=1,
    ),
    Setting("ENABLE_OCR", "bool", "false", _INGEST, "Run OCR over PDFs that have no text layer."),
    Setting("ENABLE_LANGUAGE_DETECTION", "bool", "true", _INGEST, "Detect chunk language when metadata says auto."),
    # -- storage
    Setting(
        "CHROMA_PERSIST_DIRECTORY",
        "str",
        "./indexes/chroma",
        _STORAGE,
        "Where the embedded Chroma database lives.",
        restart=True,
    ),
    Setting("CHROMA_COLLECTION_NAME", "str", "classmate_rag", _STORAGE, "Collection name inside Chroma.", restart=True),
    Setting(
        "CHROMA_HTTP_URL",
        "str",
        "",
        _STORAGE,
        "Chroma server endpoint. Empty means embedded mode.",
        restart=True,
        local_only=True,
    ),
    Setting("BM25_DIRECTORY", "str", "./indexes/bm25", _STORAGE, "Where the keyword index lives.", restart=True),
    Setting(
        "EMB_CACHE_DIR", "str", "./indexes/emb_cache", _STORAGE, "Where cached embeddings are stored.", restart=True
    ),
    # -- routing
    Setting("ENABLE_ROUTING", "bool", "false", _ROUTING, "Pick a model per question subject.", restart=True),
    Setting(
        "ROUTE_MATH_MODEL_PATH",
        "str",
        "./models/DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf",
        _ROUTING,
        "GGUF for the maths route.",
        restart=True,
    ),
    Setting(
        "ROUTE_CODE_MODEL_PATH",
        "str",
        "./models/Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf",
        _ROUTING,
        "GGUF for the code route.",
        restart=True,
    ),
    Setting(
        "ROUTE_TRANSLATION_MODEL_PATH",
        "str",
        "./models/salamandraTA-7B-instruct.Q4_K_M.gguf",
        _ROUTING,
        "GGUF for the translation route.",
        restart=True,
    ),
    Setting(
        "ROUTE_DEFAULT_MODEL_PATH",
        "str",
        "./models/Qwen3-8B-Q4_K_M.gguf",
        _ROUTING,
        "GGUF for everything else.",
        restart=True,
    ),
    Setting(
        "ROUTE_N_CTX",
        "int",
        "4096",
        _ROUTING,
        "Context window for routed models.",
        restart=True,
        minimum=256,
        maximum=131072,
    ),
    Setting(
        "ROUTE_N_GPU_LAYERS",
        "int",
        "0",
        _ROUTING,
        "Layers offloaded to the GPU. 0 is CPU, -1 is all.",
        restart=True,
        minimum=-1,
        maximum=999,
    ),
    Setting(
        "ROUTE_MAX_TOKENS", "int", "768", _ROUTING, "Cap on generated tokens.", restart=True, minimum=1, maximum=32768
    ),
    Setting("ROUTE_TEMPERATURE", "float", "0.2", _ROUTING, "Sampling temperature.", restart=True, minimum=0, maximum=2),
    Setting("ROUTE_TOP_P", "float", "0.95", _ROUTING, "Nucleus sampling cutoff.", restart=True, minimum=0, maximum=1),
    Setting(
        "ROUTE_QUERY_MARGIN",
        "float",
        "0.10",
        _ROUTING,
        "Score gap below which a question counts as ambiguous.",
        restart=True,
        minimum=0,
        maximum=1,
    ),
    Setting(
        "ROUTE_METADATA_THRESHOLD",
        "float",
        "0.60",
        _ROUTING,
        "Share of retrieved chunks that must agree on a subject.",
        restart=True,
        minimum=0,
        maximum=1,
    ),
    Setting(
        "ROUTE_TRANSLATION_REQUIRES_INTENT",
        "bool",
        "true",
        _ROUTING,
        "Require an explicit translate keyword for that route.",
        restart=True,
    ),
    # -- credentials (stored outside .env; see rag.secrets)
    Setting("ANTHROPIC_API_KEY", "secret", "", _CREDS, "Key for the anthropic provider."),
    Setting("OPENAI_API_KEY", "secret", "", _CREDS, "Key for the openai provider."),
    Setting("GEMINI_API_KEY", "secret", "", _CREDS, "Key for the gemini provider."),
    Setting("HF_TOKEN", "secret", "", _CREDS, "Token for private or gated Hugging Face repos."),
)

#: Settings that decide what code or data the application loads, or where it is
#: stored. Over an unauthenticated API they would let a caller who can already
#: upload a file (POST /ingest keeps it under data/uploads/) point
#: LLM_MODEL_PATH at it, or move the index somewhere of their choosing. Like
#: the *_URL settings, they are changed from the command line on this machine.
_ALSO_LOCAL_ONLY = frozenset(
    {
        "EMBEDDING_MODEL_NAME",
        "LLM_MODEL_PATH",
        "ROUTE_MATH_MODEL_PATH",
        "ROUTE_CODE_MODEL_PATH",
        "ROUTE_TRANSLATION_MODEL_PATH",
        "ROUTE_DEFAULT_MODEL_PATH",
        "CHROMA_PERSIST_DIRECTORY",
        "BM25_DIRECTORY",
        "EMB_CACHE_DIR",
    }
)
REGISTRY = tuple(replace(s, local_only=True) if s.key in _ALSO_LOCAL_ONLY else s for s in REGISTRY)

_BY_KEY: dict[str, Setting] = {s.key: s for s in REGISTRY}

#: Restart-required changes this process has written to .env but deliberately
#: not applied. Applying them live would be wrong: flipping ENABLE_ROUTING in a
#: running server could load four 7B models. key -> the value now in the file
#: ("" when removed).
_PENDING: dict[str, str] = {}
_TRUE = ("1", "true", "yes", "y", "on")
_FALSE = ("0", "false", "no", "n", "off")


def get_setting(key: str) -> Setting:
    try:
        return _BY_KEY[key.strip().upper()]
    except KeyError:
        raise SettingError(key, "no such setting. `rag config list` shows the ones that exist.") from None


def env_file_path() -> Path:
    """The ``.env`` that ``rag.config`` loads: the one at the project root."""
    return project_root() / ".env"


# --------------------------------------------------------------- validation


def normalise(setting: Setting, raw: str) -> str:
    """The canonical text for ``raw``, or a SettingError saying what is wrong."""
    value = raw.strip()
    key = setting.key

    # Control characters and Unicode line/paragraph separators, not just \n and
    # \r: str.splitlines() treats \x0b, \x0c, \x85, \u2028 and \u2029 as line
    # breaks too, so a value carrying one could be split into a second entry
    # the next time the file is rewritten.
    if any(unicodedata.category(ch) in ("Cc", "Zl", "Zp") for ch in value):
        raise SettingError(key, "must be a single line of printable text.")

    if setting.kind == "secret":
        if not value:
            raise SettingError(key, "is empty. Use `rag config unset` to remove it.")
        return value

    if value == "":
        # Empty means "use the default" for every kind, matching the loaders.
        return ""

    if "${" in value:
        # python-dotenv expands ${NAME} from the environment, in this file and
        # when the application loads it. A setting set to ${OPENAI_API_KEY}
        # would read back as the key itself.
        raise SettingError(key, "cannot contain '${': .env would expand it from the environment.")

    if setting.kind == "bool":
        lowered = value.lower()
        if lowered in _TRUE:
            return "true"
        if lowered in _FALSE:
            return "false"
        raise SettingError(key, f"must be true or false, not {value!r}.")

    if setting.kind == "choice":
        # Case-insensitive, canonical case from the registry (LOG_LEVEL is upper, the rest lower).
        for choice in setting.choices:
            if choice.lower() == value.lower():
                return choice
        raise SettingError(key, f"must be one of {', '.join(setting.choices)}, not {value!r}.")

    if setting.kind in ("int", "float"):
        try:
            number = int(value) if setting.kind == "int" else float(value)
        except ValueError:
            raise SettingError(key, f"must be a number, not {value!r}.") from None
        if setting.minimum is not None and number < setting.minimum:
            raise SettingError(key, f"must be at least {_fmt(setting.minimum)}.")
        if setting.maximum is not None and number > setting.maximum:
            raise SettingError(key, f"must be at most {_fmt(setting.maximum)}.")
        return str(number)

    return value


def _fmt(n: float) -> str:
    return str(int(n)) if float(n).is_integer() else str(n)


# -------------------------------------------------------------- .env access


def _file_values(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    # interpolate=False: report what the file says, never what ${NAME} expands to.
    # The expansion comes from the environment and may be a credential.
    return {k: v for k, v in dotenv_values(path, interpolate=False).items() if v is not None}


def _quote(value: str) -> str:
    """Quote only when the dotenv parser would otherwise misread the value."""
    if value and not re.search(r"[\s#'\"\\$]", value):
        return value
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def write_env(path: Path, updates: Mapping[str, str | None]) -> None:
    """
    Apply ``updates`` to ``path``: a string sets the key, ``None`` removes it.

    Edits in place so comments, ordering and unrelated lines survive, and
    writes atomically so a crash cannot leave a half-written ``.env``.
    """
    # Split on \n only. splitlines() also breaks on \x0c, \u2028 and friends,
    # which the dotenv parser does not, so a value holding one would turn into
    # a second entry when the file is rewritten.
    lines = path.read_text(encoding="utf-8").replace("\r\n", "\n").split("\n") if path.is_file() else []
    if lines and lines[-1] == "":
        lines.pop()
    mode = stat.S_IMODE(path.stat().st_mode) if path.is_file() else 0o600

    remaining = dict(updates)
    out: list[str] = []
    seen: set[str] = set()
    for line in lines:
        m = re.match(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=", line)
        key = m.group(1) if m else None
        if key is None or key not in updates:
            out.append(line)
            continue
        # The dotenv parser lets a later duplicate win, so keep the first
        # position and drop the rest rather than leave a stale override.
        if key in seen:
            continue
        seen.add(key)
        new = updates[key]
        if new is not None:
            out.append(f"{key}={_quote(new)}")
        remaining.pop(key, None)

    for key, new in remaining.items():
        if new is not None:
            out.append(f"{key}={_quote(new)}")

    body = "\n".join(out) + "\n" if out else ""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".env.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(body)
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


# ---------------------------------------------------------------- reading


@dataclass(frozen=True)
class SettingState:
    setting: Setting
    #: Effective value; masked for credentials, "" when unset.
    value: str
    source: Source
    #: True when the shell holds a different value than ``.env``, so saving to
    #: the file will not change what the application uses.
    overridden: bool = False
    #: What ``.env`` says, when it disagrees with the effective value.
    file_value: str | None = None
    #: A saved change to a restart-required setting that the running process has
    #: not picked up. ``value`` is still what is in use.
    pending_value: str | None = None


def _shell_overrides(key: str, file_values: Mapping[str, str]) -> bool:
    """
    Whether ``key`` was set by the shell rather than loaded from ``.env``.

    ``load_dotenv`` copies the file into ``os.environ``, so after loading the
    two look identical. They differ only when the shell set something else.
    """
    if key in _PENDING:
        return False  # the environment holds the value from before our own pending save
    env = os.environ.get(key)
    if env is None or env == "":
        return False
    if key not in file_values:
        return True
    if "${" in file_values[key]:
        # The loader expanded it, so the two can never be compared as text.
        # Treating that as an override would put the expanded value on screen.
        return False
    return env != file_values[key]


def describe(key: str, *, env_path: Path | None = None) -> SettingState:
    setting = get_setting(key)
    if setting.kind == "secret":
        status = secret_status(setting.key)
        source: Source = {"environment": "environment", "file": "stored"}.get(status.source, "unset")
        return SettingState(setting, status.masked, source)

    file_values = _file_values(env_path or env_file_path())
    file_value = file_values.get(setting.key)
    if setting.key in _PENDING:
        # Report what the running process uses, and what will replace it.
        running = os.environ.get(setting.key) or ""
        pending = file_value or ""
        if running == pending:
            _PENDING.pop(setting.key, None)  # changed back, nothing is waiting any more
        else:
            return SettingState(
                setting,
                running or setting.default,
                "file" if running else "default",
                pending_value=pending,
            )
    if _shell_overrides(setting.key, file_values):
        env = os.environ[setting.key]
        overridden = file_value is not None and file_value != env
        return SettingState(setting, env, "environment", overridden, file_value if overridden else None)
    if file_value not in (None, ""):
        return SettingState(setting, file_value, "file")
    return SettingState(setting, setting.default, "default")


def describe_all(*, env_path: Path | None = None) -> list[SettingState]:
    return [describe(s.key, env_path=env_path) for s in REGISTRY]


# ---------------------------------------------------------------- writing


@dataclass(frozen=True)
class ChangeResult:
    key: str
    #: "set" or "unset".
    action: str
    #: True when the running application now uses the new value.
    applied: bool
    restart_required: bool
    #: Why ``applied`` is False, for display.
    note: str = ""


def apply_changes(
    changes: Mapping[str, str | None], *, env_path: Path | None = None, trusted_local: bool = False
) -> list[ChangeResult]:
    """
    Validate every change, then make them. Nothing is written if any is invalid.

    ``None`` removes a setting. Credentials go to the credential store, the
    rest to ``.env``.

    Settings marked ``local_only`` are refused unless ``trusted_local`` is set,
    which only the command line does. The default is the safe one on purpose:
    a caller that forgets the flag, such as a future endpoint, gets the
    restriction rather than silently skipping it.
    """
    path = env_path or env_file_path()
    file_values = _file_values(path)

    cleaned: dict[str, str | None] = {}
    for raw_key, raw_value in changes.items():
        setting = get_setting(raw_key)
        if setting.local_only and not trusted_local:
            raise SettingError(
                setting.key,
                f"can only be changed from the command line on this machine: rag config set {setting.key}=VALUE",
            )
        cleaned[setting.key] = None if raw_value is None else normalise(setting, raw_value)

    _check_together(cleaned, file_values)

    env_updates: dict[str, str | None] = {}
    results: list[ChangeResult] = []

    for key, value in cleaned.items():
        setting = _BY_KEY[key]
        action = "unset" if value is None or value == "" else "set"

        if setting.kind == "secret":
            if value is None:
                delete_secret(key)
            else:
                store_secret(key, value)
            shell = os.environ.get(key)
            blocked = bool(shell) and shell != value
            results.append(
                ChangeResult(
                    key,
                    action,
                    applied=not blocked,
                    restart_required=False,
                    note="Overridden by a variable exported in your shell." if blocked else "",
                )
            )
            continue

        blocked = _shell_overrides(key, file_values) and os.environ.get(key) != value
        env_updates[key] = value if action == "set" else None
        if not blocked:
            if setting.restart:
                # Saved to .env, not applied: see _PENDING.
                _PENDING[key] = value if action == "set" else ""  # type: ignore[assignment]
            elif action == "set":
                os.environ[key] = value  # type: ignore[assignment]
            else:
                os.environ.pop(key, None)
        results.append(
            ChangeResult(
                key,
                action,
                applied=not blocked and not setting.restart,
                restart_required=setting.restart and not blocked,
                note=(
                    "Saved, but a variable exported in your shell overrides it."
                    if blocked
                    else ("Takes effect after a restart." if setting.restart else "")
                ),
            )
        )

    if env_updates:
        write_env(path, env_updates)

    # The cached Config would otherwise keep serving the old values.
    from rag.config import load_config

    load_config(reload=True)
    return results


def _check_together(cleaned: Mapping[str, str | None], file_values: Mapping[str, str]) -> None:
    """Rules that involve more than one setting, judged on the values that would result."""

    def resulting(key: str) -> int:
        if key in cleaned:
            raw = cleaned[key]
            return int(raw) if raw else int(_BY_KEY[key].default)
        current = os.environ.get(key) or file_values.get(key) or _BY_KEY[key].default
        try:
            return int(current)
        except ValueError:
            return int(_BY_KEY[key].default)

    if "CHUNK_SIZE" in cleaned or "CHUNK_OVERLAP" in cleaned:
        size, overlap = resulting("CHUNK_SIZE"), resulting("CHUNK_OVERLAP")
        if overlap >= size:
            raise SettingError("CHUNK_OVERLAP", f"must be smaller than CHUNK_SIZE ({size}), not {overlap}.")
