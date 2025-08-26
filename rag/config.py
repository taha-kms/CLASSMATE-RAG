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
from typing import List, Optional

from dotenv import load_dotenv
from rag.config import * 

def _getenv_str(name: str, default: Optional[str] = None) -> Optional[str]:
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


@dataclass(frozen=True)
class Config:
    # Embeddings
    embedding_model_name: str = "intfloat/multilingual-e5-base"

    # Local LLM (llama.cpp by default)
    llm_backend: str = "llama_cpp"
    llm_model_path: Path = Path("./models/Llama-3.1-8B-Instruct.Q4_K_M.gguf")

    # Optional auto-download parameters (used if model file missing)
    hf_token: Optional[str] = None
    llm_repo_id: Optional[str] = None
    llm_filename: Optional[str] = None

    # Chroma
    chroma_persist_directory: Path = Path("./indexes/chroma")
    chroma_collection_name: str = "classmate_rag"

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

    # Logging
    log_level: str = "INFO"

    # --- Helpers / validations (explicitly called by runtime code) ---

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
        # We do NOT require that the file exists here; caller can auto-download if missing.
        # Just return the resolved path to be used by the loader.
        return path


# Single, cached instance after first load
__CONFIG_SINGLETON: Optional[Config] = None


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
        llm_model_path=Path(_getenv_str("LLM_MODEL_PATH", "./models/Llama-3.1-8B-Instruct.Q4_K_M.gguf") or "./models/Llama-3.1-8B-Instruct.Q4_K_M.gguf"),
        hf_token=_getenv_str("HF_TOKEN")
        or _getenv_str("HUGGINGFACE_HUB_TOKEN")
        or _getenv_str("CLASSMATE_RAG_HF_TOKEN"),
        llm_repo_id=_getenv_str("LLM_REPO_ID"),
        llm_filename=_getenv_str("LLM_FILENAME"),
        chroma_persist_directory=Path(_getenv_str("CHROMA_PERSIST_DIRECTORY", "./indexes/chroma") or "./indexes/chroma"),
        chroma_collection_name=_getenv_str("CHROMA_COLLECTION_NAME", "classmate_rag") or "classmate_rag",
        chunk_size=_getenv_int("CHUNK_SIZE", 1000),
        chunk_overlap=_getenv_int("CHUNK_OVERLAP", 150),
        k_vector=_getenv_int("K_VECTOR", 8),
        k_bm25=_getenv_int("K_BM25", 8),
        use_hybrid=_getenv_bool("USE_HYBRID", True),
        enable_ocr=_getenv_bool("ENABLE_OCR", False),
        enable_language_detection=_getenv_bool("ENABLE_LANGUAGE_DETECTION", True),
        default_language=_getenv_str("DEFAULT_LANGUAGE", "auto") or "auto",
        log_level=_getenv_str("LOG_LEVEL", "INFO") or "INFO",
    )

    __CONFIG_SINGLETON = cfg
    return cfg


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
    return dict(
        chunk_size=c.chunk_size,
        chunk_overlap=c.chunk_overlap,
        k_vector=c.k_vector,
        k_bm25=c.k_bm25,
        use_hybrid=c.use_hybrid,
    )


def get_processing_toggles() -> dict:
    c = load_config()
    return dict(
        enable_ocr=c.enable_ocr,
        enable_language_detection=c.enable_language_detection,
        default_language=c.default_language,
    )
