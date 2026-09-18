"""
Embedding utilities using SentenceTransformers (E5 model).

Provides:
- E5MultilingualEmbedder: wrapper for intfloat/multilingual-e5-base
"""

from __future__ import annotations

import os
import threading
from collections.abc import Iterable
from pathlib import Path

import numpy as np

# sentence_transformers pulls in torch, which is several seconds and several
# gigabytes. Imported inside the constructor so merely importing this module,
# or anything re-exporting it, stays cheap.


def _resolve_cache_dir() -> str | None:
    """
    Pick a directory for caching downloaded models.
    Priority:
      1) SENTENCE_TRANSFORMERS_HOME
      2) HUGGINGFACE_HUB_CACHE
      3) HF_HOME
    Returns an absolute path or None.
    """
    for key in ("SENTENCE_TRANSFORMERS_HOME", "HUGGINGFACE_HUB_CACHE", "HF_HOME"):
        v = os.getenv(key)
        if v and v.strip():
            p = Path(v).expanduser().resolve()
            p.mkdir(parents=True, exist_ok=True)
            return str(p)
    return None


class E5MultilingualEmbedder:
    """
    Wrapper around SentenceTransformer('intfloat/multilingual-e5-base').

    - Adds "query:" / "passage:" prefixes as required by the model.
    - Normalizes embeddings (L2).
    - Supports custom cache directory and HuggingFace token from env.
    """

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-base",
        device: str | None = None,
        normalize: bool = True,
    ) -> None:
        cache_dir = _resolve_cache_dir()
        hf_token = os.getenv("HF_TOKEN") or None

        st_kwargs = {
            "model_name_or_path": model_name,
            "device": device,
            "trust_remote_code": False,
        }
        if cache_dir:
            st_kwargs["cache_folder"] = cache_dir
        if hf_token:
            st_kwargs["token"] = hf_token

        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(**st_kwargs)
        self.normalize = bool(normalize)

    # ------------------------------
    # Helpers for E5 input formatting
    # ------------------------------

    @staticmethod
    def _fmt_queries(queries: Iterable[str]) -> list[str]:
        """Add 'query:' prefix required by E5."""
        return [f"query: {q}" for q in queries]

    @staticmethod
    def _fmt_passages(texts: Iterable[str]) -> list[str]:
        """Add 'passage:' prefix required by E5."""
        return [f"passage: {t}" for t in texts]

    # ------------------------------
    # Public API
    # ------------------------------

    def encode_queries(self, queries: Iterable[str]) -> np.ndarray:
        """Encode queries into float32 embeddings."""
        vecs = self.model.encode(
            self._fmt_queries(queries),
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
            batch_size=32,
        )
        return vecs.astype("float32", copy=False)

    def encode_passages(self, texts: Iterable[str]) -> np.ndarray:
        """Encode passages into float32 embeddings."""
        vecs = self.model.encode(
            self._fmt_passages(texts),
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
            batch_size=32,
        )
        return vecs.astype("float32", copy=False)


__all__ = ["E5MultilingualEmbedder"]


# One embedder per (model, device, normalize) for the whole process.
#
# Two reasons, and the first is a crash rather than a nicety. Building a
# SentenceTransformer is not thread safe: four concurrent /ask requests each
# constructed their own and two died with "Cannot copy out of meta tensor",
# from torch moving the same weights to a device from several threads at once.
# The second is cost -- the model is about 1.1 GB and several seconds, and
# every request was paying it.
_EMBEDDER_CACHE: dict[tuple[str, str | None, bool], E5MultilingualEmbedder] = {}
_EMBEDDER_LOCK = threading.Lock()


def shared_embedder(
    model_name: str = "intfloat/multilingual-e5-base",
    device: str | None = None,
    normalize: bool = True,
) -> E5MultilingualEmbedder:
    """Return the process-wide embedder for these settings, building it once.

    Prefer this to constructing E5MultilingualEmbedder directly anywhere that
    can run concurrently. The constructor stays public because the tests and
    one-shot scripts want a throwaway instance.

    The lock is held across construction, not just the dictionary write, so
    that two callers racing on a cold cache cannot both enter SentenceTransformer.
    """
    key = (model_name, device, normalize)
    with _EMBEDDER_LOCK:
        embedder = _EMBEDDER_CACHE.get(key)
        if embedder is None:
            embedder = E5MultilingualEmbedder(model_name=model_name, device=device, normalize=normalize)
            _EMBEDDER_CACHE[key] = embedder
        return embedder
