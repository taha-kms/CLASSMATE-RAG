"""
Hybrid retrieval with Reciprocal Rank Fusion (RRF) and de-duplication.

This module combines:
- Vector results from Chroma (cosine distance; lower is better)
- Lexical results from BM25 (higher score is better)

We do not mix raw scores. Instead we rank each list and fuse by RRF:
    RRF_k = 60 by default
    fused_score(doc) = sum_i [ weight_i * 1 / (RRF_k + rank_i(doc)) ]

Ranks start at 1 within each list. Missing docs get no contribution.

Returned items include the fused score and per-source diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from rag.embeddings import E5MultilingualEmbedder
from rag.retrieval.vector_chroma import ChromaVectorStore, build_where_filter
from rag.retrieval.bm25 import BM25Store


def rrf_fuse(
    *,
    rank_lists: Sequence[Sequence[str]],
    weights: Optional[Sequence[float]] = None,
    rrf_k: int = 60,
) -> Dict[str, float]:
    """
    Fuse multiple ranked ID lists into a single score map.
    - rank_lists: each is an ordered list of IDs (rank 1 at index 0)
    - weights: optional per-list weights (defaults to 1.0)
    Returns: dict id -> fused_score (higher is better)
    """
    if not rank_lists:
        return {}

    n = len(rank_lists)
    if weights is None:
        weights = [1.0] * n
    else:
        if len(weights) != n:
            raise ValueError("weights length must match rank_lists length")

    scores: Dict[str, float] = {}
    for li, ids in enumerate(rank_lists):
        w = float(weights[li])
        for rank_idx, _id in enumerate(ids):
            # Reciprocal rank contribution: 1/(k + rank), ranks start at 1
            contrib = w * (1.0 / (rrf_k + (rank_idx + 1)))
            scores[_id] = scores.get(_id, 0.0) + contrib
    return scores


@dataclass
class HybridRetriever:
    """
    Orchestrates vector + BM25 retrieval with metadata filters and RRF fusion.
    """
    vector_store: ChromaVectorStore
    bm25_store: BM25Store
    embedder: E5MultilingualEmbedder

    # knobs
    k_vector: int = 8
    k_bm25: int = 8
    rrf_k: int = 60
    weight_vector: float = 1.0
    weight_bm25: float = 1.0

    def _vector_search(
        self,
        *,
        query: str,
        where: Optional[Mapping[str, object]],
        k: int,
    ) -> List[Mapping[str, object]]:
        q_vec = self.embedder.encode_queries([query])
        res = self.vector_store.query(
            query_embeddings=q_vec[0],
            where=where or {},
            top_k=k,
            include_documents=True,
        )
        # Already ranked by similarity (ascending distance)
        return res

    def _bm25_search(
        self,
        *,
        query: str,
        where: Optional[Mapping[str, object]],
        k: int,
    ) -> List[Mapping[str, object]]:
        res = self.bm25_store.search(query=query, where=where, top_k=k)
        # Already ranked by score (descending)
        return res

    def retrieve(
        self,
        *,
        question: str,
        filters: Optional[Mapping[str, object]] = None,
        top_k: int = 8,
        hybrid: bool = True,
    ) -> List[Dict[str, object]]:
        """
        Run retrieval (vector-only or hybrid) with metadata filters and return
        a fused, de-duplicated top_k list.

        Output item schema:
            {
                "id": str,
                "document": str,
                "metadata": dict,
                "scores": {
                    "vector_distance": float | None,
                    "bm25_score": float | None,
                    "fused": float
                }
            }
        """
        where = build_where_filter(filters or {}) if filters else None

        # Run searches
        vec_res: List[Mapping[str, object]] = []
        bm25_res: List[Mapping[str, object]] = []

        if hybrid:
            vec_res = self._vector_search(query=question, where=where, k=self.k_vector)
            bm25_res = self._bm25_search(query=question, where=where, k=self.k_bm25)
        else:
            vec_res = self._vector_search(query=question, where=where, k=max(top_k, self.k_vector))

        # Prepare rank lists by ID
        vec_ids = [r["id"] for r in vec_res]
        bm_ids = [r["id"] for r in bm25_res]

        # Compute fused scores
        if hybrid:
            fused = rrf_fuse(
                rank_lists=[vec_ids, bm_ids],
                weights=[self.weight_vector, self.weight_bm25],
                rrf_k=self.rrf_k,
            )
        else:
            fused = rrf_fuse(rank_lists=[vec_ids], weights=[1.0], rrf_k=self.rrf_k)

        # Build a union map of documents/metadata and per-source scores
        by_id: Dict[str, Dict[str, object]] = {}

        # Vector diagnostics: distances, docs, metadata
        for rank_idx, r in enumerate(vec_res):
            _id = r["id"]
            item = by_id.setdefault(
                _id,
                {"id": _id, "document": None, "metadata": {}, "scores": {"vector_distance": None, "bm25_score": None, "fused": 0.0}},
            )
            item["document"] = item["document"] or r.get("document")
            item["metadata"] = item["metadata"] or r.get("metadata") or {}
            item["scores"]["vector_distance"] = r.get("distance")

        # BM25 diagnostics: scores, docs, metadata (fill blanks if vector didn't have document)
        for rank_idx, r in enumerate(bm25_res):
            _id = r["id"]
            item = by_id.setdefault(
                _id,
                {"id": _id, "document": None, "metadata": {}, "scores": {"vector_distance": None, "bm25_score": None, "fused": 0.0}},
            )
            # If document not present from vector path, use bm25 text
            if not item["document"] and r.get("document"):
                item["document"] = r.get("document")
            # Merge metadata if empty
            if not item["metadata"] and r.get("metadata"):
                item["metadata"] = r.get("metadata") or {}
            item["scores"]["bm25_score"] = r.get("score")

        # Attach fused scores
        for _id, score in fused.items():
            if _id in by_id:
                by_id[_id]["scores"]["fused"] = float(score)

        # Sort by fused score desc; tie-break by better vector distance if available
        def _sort_key(it):
            s = it["scores"]
            fused_score = s.get("fused") or 0.0
            vd = s.get("vector_distance")
            # vector distance: lower is better -> invert for tie-breaker
            vd_term = -(vd if isinstance(vd, (int, float)) else 0.0)
            return (fused_score, vd_term)

        fused_list = sorted(by_id.values(), key=_sort_key, reverse=True)
        return fused_list[:top_k]
