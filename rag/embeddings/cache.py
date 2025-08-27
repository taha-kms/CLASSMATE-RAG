"""
Disk-level embedding cache for E5-style encoders.

- Keyed by (model_name, mode=query|passage, sha1(content))
- Stores .npy vectors (float32) one-per-file for robustness
- Batch encode APIs that transparently fill the cache

Usage:
    from rag.embeddings import E5MultilingualEmbedder
    from rag.embeddings.cache import CachingEmbedder

    base = E5MultilingualEmbedder(model_name="intfloat/multilingual-e5-base")
    emb = CachingEmbedder(base, cache_dir="./indexes/emb_cache")
    vecs = emb.encode_passages(["hello", "world"])
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np


def _sha1_bytes(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def _norm_bytes(s: str) -> bytes:
    # Stable normalization: strip trailing spaces; preserve case and punctuation
    return (s or "").strip().encode("utf-8", "ignore")


class CachingEmbedder:
    """
    Thin wrapper with the same public API as the base embedder:
      - encode_queries(list[str]) -> np.ndarray (N,D)
      - encode_passages(list[str]) -> np.ndarray (N,D)
    """

    def __init__(self, base, cache_dir: Optional[str] = None) -> None:
        self.base = base
        root = cache_dir or os.getenv("EMB_CACHE_DIR") or "./indexes/emb_cache"
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

        # Model-scoped subdir to avoid vector-size conflicts across models
        model_name = getattr(base, "model", None)
        if model_name and hasattr(model_name, "name_or_path"):
            mname = str(model_name.name_or_path)
        else:
            mname = getattr(base, "model_name", "unknown-model")
        # sanitize path-ish strings
        safe = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in mname)
        self.model_dir = self.root / safe
        self.model_dir.mkdir(parents=True, exist_ok=True)

    # ------------- internal helpers -------------

    def _key_path(self, mode: str, text: str) -> Path:
        h = _sha1_bytes(_norm_bytes(text))
        return self.model_dir / mode / f"{h}.npy"

    def _ensure_mode_dir(self, mode: str) -> Path:
        p = self.model_dir / mode
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _get_many(
        self, mode: str, texts: Iterable[str]
    ) -> Tuple[List[Optional[np.ndarray]], List[int], List[str], List[Path]]:
        """
        Returns:
            cached: list of vec or None (parallel to input order)
            miss_idx: indices of misses
            miss_texts: texts for misses
            miss_paths: file paths where new vecs should be saved
        """
        self._ensure_mode_dir(mode)
        cached: List[Optional[np.ndarray]] = []
        miss_idx: List[int] = []
        miss_texts: List[str] = []
        miss_paths: List[Path] = []

        for i, t in enumerate(texts):
            fp = self._key_path(mode, t)
            if fp.exists():
                try:
                    vec = np.load(fp)
                    cached.append(vec.astype("float32", copy=False))
                    continue
                except Exception:
                    # Treat corrupted entries as a miss; they will be overwritten
                    pass
            cached.append(None)
            miss_idx.append(i)
            miss_texts.append(t)
            miss_paths.append(fp)
        return cached, miss_idx, miss_texts, miss_paths

    @staticmethod
    def _fill(cached: List[Optional[np.ndarray]], miss_idx: List[int], miss_vecs: np.ndarray) -> np.ndarray:
        out: List[np.ndarray] = []
        it = iter(miss_vecs)
        for i, slot in enumerate(cached):
            if slot is None:
                out.append(next(it))
            else:
                out.append(slot)
        return np.vstack(out).astype("float32", copy=False)

    # ------------- public API -------------

    def encode_queries(self, queries: Iterable[str]) -> np.ndarray:
        items = list(queries)
        cached, miss_idx, miss_items, miss_paths = self._get_many("query", items)

        if miss_items:
            miss_vecs = self.base.encode_queries(miss_items)
            # persist
            for j, fp in enumerate(miss_paths):
                try:
                    fp.parent.mkdir(parents=True, exist_ok=True)
                    np.save(fp, miss_vecs[j])
                except Exception:
                    pass
        else:
            miss_vecs = np.zeros((0, 0), dtype="float32")

        return self._fill(cached, miss_idx, miss_vecs)

    def encode_passages(self, texts: Iterable[str]) -> np.ndarray:
        items = list(texts)
        cached, miss_idx, miss_items, miss_paths = self._get_many("passage", items)

        if miss_items:
            miss_vecs = self.base.encode_passages(miss_items)
            for j, fp in enumerate(miss_paths):
                try:
                    fp.parent.mkdir(parents=True, exist_ok=True)
                    np.save(fp, miss_vecs[j])
                except Exception:
                    pass
        else:
            miss_vecs = np.zeros((0, 0), dtype="float32")

        return self._fill(cached, miss_idx, miss_vecs)
