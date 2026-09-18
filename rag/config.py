"""
CLASSMATE-RAG configuration loader.

- Reads environment variables and .env without failing on import.
- Provides a typed Config object with sensible defaults.
- No hard dependency on any cloud API keys.
- Includes light validators you can call at runtime (not on import).

Usage:
    from classmate.config import load_config
    cfg = load_config()
    # Optionally validate when needed:
    # cfg.validate_for_embeddings()
    # path = cfg.validate_for_llm()  # returns resolved model path
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def _project_root() -> Path:
    """
    Directory that relative data paths resolve against.

    Walks up from this file looking for pyproject.toml, which is what a source
    checkout or an editable install looks like. Falls back to the working
    directory when the package lives in site-packages, where writing indexes
    next to the installed code would be wrong.

    Without this, "./indexes/bm25" was interpreted relative to wherever the
    process happened to start, so running the CLI from another directory
    silently created a second, empty index instead of finding the real one.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return Path.cwd()


def resolve_data_path(value: str | Path) -> Path:
    """Resolve a possibly-relative data path against the project root."""
    p = Path(value).expanduser()
    if p.is_absolute():
        return p
    return (_project_root() / p).resolve()


def _getenv_str(name: str, default: str | None = None) -> str | None:
    val = os.getenv(name)
    if val is None or val == "":
        return default
    return val


def _getenv_int(name: str, default: int) -> int:
    val = os.getenv(name)
    if val is None or val == "":
        return default
    try:
        return int(val)
    except Exception:
        return default


def _getenv_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None or val == "":
        return default
    return str(val).strip().lower() in ("1", "true", "yes", "y", "on")


def _getenv_float(name: str, default: float) -> float:
    val = os.getenv(name)
    if val is None or val == "":
        return default
    try:
        return float(val)
    except Exception:
        return default


@dataclass(frozen=True)
class Config:
    # Embeddings
    embedding_model_name: str = "intfloat/multilingual-e5-base"

    # Local LLM (llama.cpp by default)
    llm_backend: str = "llama_cpp"

    # Which provider generates answers. llama_cpp runs locally; hosted
    # providers register themselves under their own names.
    llm_provider: str = "llama_cpp"

    # Hosted providers. Keys come from the credential store or the
    # environment; models default per provider.
    anthropic_api_key: str | None = None
    anthropic_model: str | None = None
    openai_api_key: str | None = None
    openai_model: str | None = None
    # Points the OpenAI adapter at anything speaking the same protocol.
    openai_base_url: str | None = None
    llm_model_path: Path = Path("./models/Llama-3.1-8B-Instruct.Q4_K_M.gguf")

    # Optional auto-download parameters (used if model file missing)
    hf_token: str | None = None
    llm_repo_id: str | None = None
    llm_filename: str | None = None

    # Chroma
    chroma_persist_directory: Path = Path("./indexes/chroma")
    chroma_collection_name: str = "classmate_rag"

    # Lexical index and embedding cache. Resolved against the project root
    # so the CLI finds the same corpus from any working directory.
    bm25_directory: Path = Path("./indexes/bm25")
    emb_cache_directory: Path = Path("./indexes/emb_cache")

    # Chunking / retrieval
    chunk_size: int = 1000
    chunk_overlap: int = 150
    k_vector: int = 8
    k_bm25: int = 8
    use_hybrid: bool = True

    # Processing toggles
    enable_ocr: bool = False
    enable_language_detection: bool = True

    # Language behavior
    default_language: str = "auto"  # "en" | "it" | "auto"

    # Retrieval ergonomics applied after fusion
    enable_neighbor_expansion: bool = True
    neighbor_radius: int = 1
    doc_diversity_cap: int = 3

    # Ingestion tuning
    ingest_threads: int = 0  # 0 means "derive from the CPU count"
    dedup_chunks: bool = False
    dedup_threshold: float = 0.92

    # Generation parameters used by the routed path
    route_max_tokens: int = 768
    route_temperature: float = 0.2
    route_top_p: float = 0.95

    # Answer post-processing
    strict_citations: bool = False
    append_sources_block: bool = False
    translate_on_miss: bool = False

    # Logging
    log_level: str = "INFO"

    # ----------------------------------------------------------------
    # Subject-aware routing (multi-model)
    # ----------------------------------------------------------------
    # Master toggle. When False, the pipeline uses the legacy single-model path.
    enable_routing: bool = False

    # Named model profile: light | balanced | heavy | custom. "custom" uses
    # the per-route ROUTE_*_MODEL_PATH values below verbatim, which is how
    # this behaved before profiles existed.
    model_profile: str = "custom"

    # Per-route GGUF paths. Empty strings disable that route (it falls back to
    # the default route). Override via env: ROUTE_<NAME>_MODEL_PATH.
    route_math_model_path: Path = Path("./models/DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf")
    route_code_model_path: Path = Path("./models/Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf")
    route_translation_model_path: Path = Path("./models/salamandraTA-7B-instruct.Q4_K_M.gguf")
    route_default_model_path: Path = Path("./models/Qwen3-8B-Q4_K_M.gguf")

    # Per-route context window. 4096 keeps VRAM headroom on 8 GB cards.
    route_n_ctx: int = 4096

    # Sticky loader: how many GPU layers to push (0 = CPU; -1 = all).
    route_n_gpu_layers: int = 0

    # Hybrid resolution thresholds.
    # Margin between top-1 and top-2 query scores below which the query is
    # considered ambiguous and metadata is consulted.
    route_query_margin: float = 0.10
    # Minimum fraction of top-k retrieved chunks that must agree on a subject
    # before metadata can override an ambiguous query.
    route_metadata_threshold: float = 0.60

    # Translation route extras: requires explicit translate-intent keyword on
    # top of the prototype score (because SalamandraTA is translation-only).
    route_translation_requires_intent: bool = True

    # --- Helpers / validations (explicitly called by runtime code) ---

    def resolved_ingest_threads(self) -> int:
        """Threads for page chunking. 0 means derive from the CPU count."""
        import os as _os

        if self.ingest_threads > 0:
            return self.ingest_threads
        return max(2, (_os.cpu_count() or 4) // 2)

    def validate_for_embeddings(self) -> None:
        """
        Validate embedding settings when first creating the embedding model.
        Currently no network/API secrets required.
        """
        if not self.embedding_model_name:
            raise RuntimeError("EMBEDDING_MODEL_NAME is not set.")

    def validate_for_llm(self) -> Path:
        """
        Validate LLM settings and return a usable local path to the model file.
        If auto-download is desired, higher-level code will call the model fetcher.
        """
        path = self.llm_model_path.expanduser().resolve()
        return path


# Single, cached instance after first load
__CONFIG_SINGLETON: Config | None = None


def _read_secret_setting(*names: str) -> str | None:
    """
    First non-empty value among `names`, checking the environment before the
    credential store.

    Imported lazily: rag.secrets imports from here, and doing it at module
    scope would be circular.
    """
    from rag.secrets import read_secret

    for name in names:
        value = read_secret(name)
        if value:
            return value
    return None


def load_config(reload: bool = False) -> Config:
    """
    Load configuration from environment and .env (once) with defaults.
    Use reload=True to force re-reading.
    """
    global __CONFIG_SINGLETON
    if __CONFIG_SINGLETON is not None and not reload:
        return __CONFIG_SINGLETON

    # Load .env only once; do not override already-set env vars.
    load_dotenv(override=False)

    cfg = Config(
        embedding_model_name=_getenv_str("EMBEDDING_MODEL_NAME", "intfloat/multilingual-e5-base")
        or "intfloat/multilingual-e5-base",
        llm_backend=_getenv_str("LLM_BACKEND", "llama_cpp") or "llama_cpp",
        llm_provider=(_getenv_str("LLM_PROVIDER", "llama_cpp") or "llama_cpp").strip().lower(),
        anthropic_api_key=_read_secret_setting("ANTHROPIC_API_KEY"),
        anthropic_model=_getenv_str("ANTHROPIC_MODEL"),
        openai_api_key=_read_secret_setting("OPENAI_API_KEY"),
        openai_model=_getenv_str("OPENAI_MODEL"),
        openai_base_url=_getenv_str("OPENAI_BASE_URL"),
        llm_model_path=Path(
            _getenv_str("LLM_MODEL_PATH", "./models/Llama-3.1-8B-Instruct.Q4_K_M.gguf")
            or "./models/Llama-3.1-8B-Instruct.Q4_K_M.gguf"
        ),
        # Environment first, then the credential store. Reading it here
        # keeps every caller on one precedence rule (#48, #95).
        hf_token=_read_secret_setting("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "CLASSMATE_RAG_HF_TOKEN"),
        llm_repo_id=_getenv_str("LLM_REPO_ID"),
        llm_filename=_getenv_str("LLM_FILENAME"),
        chroma_persist_directory=resolve_data_path(
            _getenv_str("CHROMA_PERSIST_DIRECTORY", "./indexes/chroma") or "./indexes/chroma"
        ),
        bm25_directory=resolve_data_path(_getenv_str("BM25_DIRECTORY", "./indexes/bm25") or "./indexes/bm25"),
        emb_cache_directory=resolve_data_path(
            _getenv_str("EMB_CACHE_DIR", "./indexes/emb_cache") or "./indexes/emb_cache"
        ),
        chroma_collection_name=_getenv_str("CHROMA_COLLECTION_NAME", "classmate_rag") or "classmate_rag",
        chunk_size=_getenv_int("CHUNK_SIZE", 1000),
        chunk_overlap=_getenv_int("CHUNK_OVERLAP", 150),
        k_vector=_getenv_int("K_VECTOR", 8),
        k_bm25=_getenv_int("K_BM25", 8),
        use_hybrid=_getenv_bool("USE_HYBRID", True),
        enable_ocr=_getenv_bool("ENABLE_OCR", False),
        enable_language_detection=_getenv_bool("ENABLE_LANGUAGE_DETECTION", True),
        default_language=_getenv_str("DEFAULT_LANGUAGE", "auto") or "auto",
        enable_neighbor_expansion=_getenv_bool("ENABLE_NEIGHBOR_EXPANSION", True),
        neighbor_radius=_getenv_int("NEIGHBOR_RADIUS", 1),
        doc_diversity_cap=_getenv_int("DOC_DIVERSITY_CAP", 3),
        ingest_threads=_getenv_int("INGEST_THREADS", 0),
        dedup_chunks=_getenv_bool("DEDUP_CHUNKS", False),
        dedup_threshold=_getenv_float("DEDUP_THRESHOLD", 0.92),
        route_max_tokens=_getenv_int("ROUTE_MAX_TOKENS", 768),
        route_temperature=_getenv_float("ROUTE_TEMPERATURE", 0.2),
        route_top_p=_getenv_float("ROUTE_TOP_P", 0.95),
        strict_citations=_getenv_bool("STRICT_CITATIONS", False),
        append_sources_block=_getenv_bool("APPEND_SOURCES_BLOCK", False),
        translate_on_miss=_getenv_bool("TRANSLATE_ON_MISS", False),
        log_level=_getenv_str("LOG_LEVEL", "INFO") or "INFO",
        enable_routing=_getenv_bool("ENABLE_ROUTING", False),
        model_profile=(_getenv_str("MODEL_PROFILE", "custom") or "custom").strip().lower(),
        route_math_model_path=Path(
            _getenv_str("ROUTE_MATH_MODEL_PATH", "./models/DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf")
            or "./models/DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf"
        ),
        route_code_model_path=Path(
            _getenv_str("ROUTE_CODE_MODEL_PATH", "./models/Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf")
            or "./models/Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf"
        ),
        route_translation_model_path=Path(
            _getenv_str("ROUTE_TRANSLATION_MODEL_PATH", "./models/salamandraTA-7B-instruct.Q4_K_M.gguf")
            or "./models/salamandraTA-7B-instruct.Q4_K_M.gguf"
        ),
        route_default_model_path=Path(
            _getenv_str("ROUTE_DEFAULT_MODEL_PATH", "./models/Qwen3-8B-Q4_K_M.gguf") or "./models/Qwen3-8B-Q4_K_M.gguf"
        ),
        route_n_ctx=_getenv_int("ROUTE_N_CTX", 4096),
        route_n_gpu_layers=_getenv_int("ROUTE_N_GPU_LAYERS", 0),
        route_query_margin=_getenv_float("ROUTE_QUERY_MARGIN", 0.10),
        route_metadata_threshold=_getenv_float("ROUTE_METADATA_THRESHOLD", 0.60),
        route_translation_requires_intent=_getenv_bool("ROUTE_TRANSLATION_REQUIRES_INTENT", True),
    )

    __CONFIG_SINGLETON = cfg
    return cfg


def configure_logging(level: str | None = None) -> None:
    """
    Install a basic logging configuration from LOG_LEVEL.

    Config has carried a log_level for a while but nothing ever called
    basicConfig, so every log record in the codebase was discarded. The model
    load and evict messages in rag/routing/loader.py are the ones worth
    seeing, since they explain why a query took forty seconds.

    Records go to stderr on purpose. The CLI prints JSON on stdout and that
    has to stay machine-readable.
    """
    import logging
    import sys

    name = str(level or load_config().log_level or "INFO").strip().upper()
    resolved = getattr(logging, name, None)
    if not isinstance(resolved, int):
        resolved = logging.INFO

    root = logging.getLogger()
    if root.handlers:
        # Something already configured logging (pytest, an embedding host
        # app). Respect their handlers, just apply our level.
        root.setLevel(resolved)
        return

    logging.basicConfig(
        level=resolved,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    _quiet_noisy_libraries(resolved)


# Libraries that log a line per HTTP request, or per model file touched, at
# INFO. At the default level they would bury the handful of messages this
# project actually emits.
_NOISY_LOGGERS = (
    "httpx",
    "httpcore",
    "urllib3",
    "chromadb",
    "posthog",
    "sentence_transformers",
    "transformers",
    "filelock",
)


def _quiet_noisy_libraries(level: int) -> None:
    """Hold third-party loggers at WARNING unless we are actually debugging."""
    import logging

    if level <= logging.DEBUG:
        return  # the user asked for everything, so give them everything
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


# Convenience getters (optional, to align with older code styles)
def get_embedding_model_name() -> str:
    return load_config().embedding_model_name


def get_llm_backend() -> str:
    return load_config().llm_backend


def get_llm_model_path() -> Path:
    return load_config().llm_model_path


def get_chroma_settings() -> tuple[Path, str]:
    c = load_config()
    return c.chroma_persist_directory, c.chroma_collection_name


def get_retrieval_settings() -> dict:
    c = load_config()
    return {
        "chunk_size": c.chunk_size,
        "chunk_overlap": c.chunk_overlap,
        "k_vector": c.k_vector,
        "k_bm25": c.k_bm25,
        "use_hybrid": c.use_hybrid,
    }


def get_processing_toggles() -> dict:
    c = load_config()
    return {
        "enable_ocr": c.enable_ocr,
        "enable_language_detection": c.enable_language_detection,
        "default_language": c.default_language,
    }
