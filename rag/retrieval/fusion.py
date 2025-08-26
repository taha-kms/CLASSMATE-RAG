"""
Hybrid retrieval with Reciprocal Rank Fusion (RRF) + optional MMR diversification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence

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
    if not rank_lists:
        return {}
    n = len(rank_lists)
    if weights is None:
        weights = [1.0] * n
    elif len(weights) != n:
        raise ValueError("weights length must match rank_lists length")
    scores: Dict[str, float] = {}
    for li, ids in enumerate(rank_lists):
        w = float(weights[li])
        for rank_idx, _id in enumerate(ids):
            contrib = w * (1.0 / (rrf_k + (rank_idx + 1)))
            scores[_id] = scores.get(_id, 0.0) + contrib
    return scores


def _mmr_order(q: np.ndarray, cands: np.ndarray, ids: List[str], k: int, lambd: float = 0.5) -> List[int]:
    if len(ids) == 0:
        return []
    q = q.reshape(1, -1).astype("float32")
    sims_q = (cands @ q.T).ravel()
    sims_cc = cands @ cands.T
    selected: List[int] = []
    remaining = set(range(len(ids)))
    first = int(np.argmax(sims_q))
    selected.append(first)
    remaining.discard(first)
    while remaining and len(selected) < min(k, len(ids)):
        best_idx = None
        best_score = -1e9
        for i in list(remaining):
            diversity = np.max(sims_cc[i, selected]) if selected else 0.0
            score = lambd * sims_q[i] - (1.0 - lambd) * diversity
            if score > best_score:
                best_score = score
                best_idx = i
        selected.append(int(best_idx))
        remaining.discard(int(best_idx))
    return selected


@dataclass
class HybridRetriever:
    vector_store: ChromaVectorStore
    bm25_store: BM25Store
    embedder: E5MultilingualEmbedder

    k_vector: int = 8
    k_bm25: int = 8
    rrf_k: int = 60
    weight_vector: float = 1.0
    weight_bm25: float = 1.0

    use_mmr: bool = True
    mmr_lambda: float = 0.5
    mmr_max_pool: int = 24

    def _vector_search(self, *, query: str, where: Optional[Mapping[str, object]], k: int) -> List[Mapping[str, object]]:
        q_vec = self.embedder.encode_queries([query])[0]
        pool_size = max(k, self.mmr_max_pool) if self.use_mmr else k
        res = self.vector_store.query(
            query_embeddings=q_vec,
            where=where,                     # pass Chroma-style where (or None)
            top_k=pool_size,
            include_documents=True,
            include_embeddings=self.use_mmr,
        )
        if not self.use_mmr:
            return res[:k]
        ids, embs = [], []
        for r in res:
            if "embedding" in r and isinstance(r["embedding"], np.ndarray):
                ids.append(r["id"])
                embs.append(r["embedding"])
        if not ids:
            return res[:k]
        cand = np.stack(embs, axis=0)
        order = _mmr_order(q=q_vec, cands=cand, ids=ids, k=k, lambd=self.mmr_lambda)
        id_to_item = {r["id"]: r for r in res}
        return [id_to_item[ids[i]] for i in order if ids[i] in id_to_item]

    def _bm25_search(self, *, query: str, where: Optional[Mapping[str, object]], k: int) -> List[Mapping[str, object]]:
        # BM25 expects simple metadata dict (not Chroma '$and' format)
        return self.bm25_store.search(query=query, where=where, top_k=k)

    def retrieve(
        self,
        *,
        question: str,
        filters: Optional[Mapping[str, object]] = None,
        top_k: int = 8,
        hybrid: bool = True,
    ) -> List[Dict[str, object]]:
        raw_filters = filters or {}
        chroma_where = build_where_filter(raw_filters) if raw_filters else None
        bm_where = raw_filters or None

        vec_res: List[Mapping[str, object]] = []
        bm25_res: List[Mapping[str, object]] = []

        if hybrid:
            vec_res = self._vector_search(query=question, where=chroma_where, k=self.k_vector)
            bm25_res = self._bm25_search(query=question, where=bm_where, k=self.k_bm25)
        else:
            vec_res = self._vector_search(query=question, where=chroma_where, k=max(top_k, self.k_vector))

        vec_ids = [r["id"] for r in vec_res]
        bm_ids = [r["id"] for r in bm25_res]

        fused = rrf_fuse(
            rank_lists=[vec_ids, bm_ids] if hybrid else [vec_ids],
            weights=[self.weight_vector, self.weight_bm25] if hybrid else [1.0],
            rrf_k=self.rrf_k,
        )

        by_id: Dict[str, Dict[str, object]] = {}
        for r in vec_res:
            _id = r["id"]
            item = by_id.setdefault(_id, {"id": _id, "document": None, "metadata": {}, "scores": {"vector_distance": None, "bm25_score": None, "fused": 0.0}})
            item["document"] = item["document"] or r.get("document")
            item["metadata"] = item["metadata"] or r.get("metadata") or {}
            item["scores"]["vector_distance"] = r.get("distance")

        for r in bm25_res:
            _id = r["id"]
            item = by_id.setdefault(_id, {"id": _id, "document": None, "metadata": {}, "scores": {"vector_distance": None, "bm25_score": None, "fused": 0.0}})
            if not item["document"] and r.get("document"):
                item["document"] = r.get("document")
            if not item["metadata"] and r.get("metadata"):
                item["metadata"] = r.get("metadata") or {}
            item["scores"]["bm25_score"] = r.get("score")

        for _id, score in fused.items():
            if _id in by_id:
                by_id[_id]["scores"]["fused"] = float(score)

        def _sort_key(it):
            s = it["scores"]
            fused_score = s.get("fused") or 0.0
            vd = s.get("vector_distance")
            vd_term = -(vd if isinstance(vd, (int, float)) else 0.0)
            return (fused_score, vd_term)

        fused_list = sorted(by_id.values(), key=_sort_key, reverse=True)
        return fused_list[:top_k]
