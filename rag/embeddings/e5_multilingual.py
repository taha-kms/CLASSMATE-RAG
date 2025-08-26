"""
Embedding integration for intfloat/multilingual-e5-base.

Key behaviors:
- Formats inputs with the required prefixes:
    query  -> "query:  <text>"
    passage-> "passage:<text>"
- L2 normalizes embeddings (recommended for cosine similarity in vector DBs).
- Supports batched encoding and returns numpy arrays.
- No fail-on-import; model loads lazily on first use.

Usage (example):
    emb = E5MultilingualEmbedder("intfloat/multilingual-e5-base")
    q_vecs = emb.encode_queries(["cos'è una lista collegata?"])
    p_vecs = emb.encode_passages(["Le liste collegate ..."])
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    # Avoid division by zero
    norms[norms == 0] = 1.0
    return mat / norms


@dataclass
class E5MultilingualEmbedder:
    model_name: str = "intfloat/multilingual-e5-base"
    device: Optional[str] = None      # e.g., "cuda", "cpu"; None lets ST pick
    batch_size: int = 32
    normalize: bool = True

    # internal
    _model: Optional[SentenceTransformer] = None

    def _ensure_model(self) -> SentenceTransformer:
        if self._model is None:
            # SentenceTransformer handles device placement automatically if device is None
            self._model = SentenceTransformer(self.model_name, device=self.device)
            # Disable gradients for safety/speed
            try:
                self._model.requires_grad_(False)
            except Exception:
                pass
        return self._model

    @staticmethod
    def _format_queries(texts: Iterable[str]) -> List[str]:
        # e5 expects the "query:" prefix
        return [f"query: {t.strip()}" for t in texts]

    @staticmethod
    def _format_passages(texts: Iterable[str]) -> List[str]:
        # e5 expects the "passage:" prefix
        return [f"passage: {t.strip()}" for t in texts]

    def encode_queries(self, texts: Iterable[str]) -> np.ndarray:
        """
        Encode user queries (with 'query:' prefix). Returns shape (N, D).
        """
        model = self._ensure_model()
        formatted = self._format_queries(texts)
        vecs = model.encode(
            formatted,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,  # we apply our own normalization for clarity
            show_progress_bar=False,
        )
        return _l2_normalize(vecs) if self.normalize else vecs

    def encode_passages(self, texts: Iterable[str]) -> np.ndarray:
        """
        Encode passages/chunks (with 'passage:' prefix). Returns shape (N, D).
        """
        model = self._ensure_model()
        formatted = self._format_passages(texts)
        vecs = model.encode(
            formatted,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        return _l2_normalize(vecs) if self.normalize else vecs
