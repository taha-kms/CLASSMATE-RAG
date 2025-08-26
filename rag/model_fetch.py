"""
Auto-check and download the Llama 3.1-8B GGUF model into ./models if missing.

Env vars used (via .env):
  - LLM_MODEL_PATH  : target local path (e.g., ./models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf)
  - LLM_REPO_ID     : HF repo id hosting the GGUF (e.g., bartowski/Meta-Llama-3.1-8B-Instruct-GGUF)
  - LLM_FILENAME    : exact GGUF filename to fetch
  - HF_TOKEN        : Hugging Face token (or HUGGINGFACE_HUB_TOKEN / CLASSMATE_RAG_HF_TOKEN)

Usage:
  from rag.model_fetch import ensure_llama_model_available
  path = ensure_llama_model_available()
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from huggingface_hub import snapshot_download


def _read_env(var: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(var)
    if v is None or v == "":
        return default
    return v


def ensure_llama_model_available() -> Path:
    """
    Ensures the GGUF model file defined by LLM_MODEL_PATH exists under ./models.
    If not present, downloads LLM_FILENAME from LLM_REPO_ID into ./models/ using HF token.
    Returns the absolute Path to the model file; raises RuntimeError on failure.
    """
    load_dotenv(override=False)

    model_path_str = _read_env("LLM_MODEL_PATH")
    repo_id = _read_env("LLM_REPO_ID")
    filename = _read_env("LLM_FILENAME")
    token = (
        _read_env("HF_TOKEN")
        or _read_env("HUGGINGFACE_HUB_TOKEN")
        or _read_env("CLASSMATE_RAG_HF_TOKEN")
    )

    if not model_path_str:
        raise RuntimeError(
            "LLM_MODEL_PATH is not set. Please set it in your .env "
            "(e.g., ./models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf)."
        )

    model_path = Path(model_path_str).expanduser().resolve()
    models_dir = Path("./models").resolve()

    # If the file already exists, we're done.
    if model_path.exists() and model_path.is_file():
        return model_path

    # Need to download. Validate required info.
    if not repo_id or not filename:
        raise RuntimeError(
            "Model file is missing and auto-download parameters are incomplete.\n"
            "Please set LLM_REPO_ID and LLM_FILENAME in your .env (and HF_TOKEN if required)."
        )

    if not token:
        print(
            "Warning: No HF token found in HF_TOKEN/HUGGINGFACE_HUB_TOKEN/CLASSMATE_RAG_HF_TOKEN. "
            "Attempting download without a token (may fail for gated repos).",
            file=sys.stderr,
        )

    # Ensure ./models exists
    models_dir.mkdir(parents=True, exist_ok=True)

    # Download only the specified filename into ./models (resumable, no symlinks)
    try:
        snapshot_download(
            repo_id=repo_id,
            allow_patterns=[filename],
            local_dir=str(models_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
            token=token,
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to download '{filename}' from '{repo_id}'. "
            f"Check HF token access and repo/filename. Original error: {e}"
        ) from e

    # Verify the file landed where we expect
    candidate = models_dir / filename
    if not candidate.exists():
        matches = list(models_dir.rglob(filename))
        if matches:
            candidate = matches[0]
        else:
            raise RuntimeError(
                f"Downloaded file '{filename}' not found under {models_dir}. "
                "Please verify LLM_FILENAME."
            )

    # If LLM_MODEL_PATH points elsewhere, return the actual downloaded path.
    return candidate.resolve()


if __name__ == "__main__":
    try:
        p = ensure_llama_model_available()
        print(f"Model ready at: {p}")
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise SystemExit(1)
