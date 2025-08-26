from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List

import numpy as np
from sentence_transformers import SentenceTransformer


def _resolve_cache_dir() -> str | None:
    """
    Decide where to cache embedding models.
    Priority:
      1) SENTENCE_TRANSFORMERS_HOME
      2) HUGGINGFACE_HUB_CACHE
      3) HF_HOME
    If set, return an ABSOLUTE path (created if missing).
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

    - Adds 'query:' / 'passage:' prefixes as required by E5.
    - L2-normalizes embeddings.
    - Forces cache directory from env so models can live under ./models/hf_cache if desired.
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
        # sentence-transformers >=2.2 supports 'cache_folder' and 'token'
        if cache_dir:
            st_kwargs["cache_folder"] = cache_dir
        if hf_token:
            st_kwargs["token"] = hf_token

        self.model = SentenceTransformer(**st_kwargs)
        self.normalize = bool(normalize)

    # ---- E5 formatting helpers ----

    @staticmethod
    def _fmt_queries(queries: Iterable[str]) -> List[str]:
        return [f"query: {q}" for q in queries]

    @staticmethod
    def _fmt_passages(texts: Iterable[str]) -> List[str]:
        return [f"passage: {t}" for t in texts]

    # ---- Public encode APIs ----

    def encode_queries(self, queries: Iterable[str]) -> np.ndarray:
        vecs = self.model.encode(
            self._fmt_queries(queries),
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
            batch_size=32,
        )
        return vecs.astype("float32", copy=False)

    def encode_passages(self, texts: Iterable[str]) -> np.ndarray:
        vecs = self.model.encode(
            self._fmt_passages(texts),
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
            batch_size=32,
        )
        return vecs.astype("float32", copy=False)
