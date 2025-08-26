"""
Admin/observability helpers for CLASSMATE-RAG.

- retrieve_preview(): run retrieval only and show what would be fed to the model
- index_stats(): quick index health (counts + disk usage)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional

from rag.config import load_config
from rag.embeddings import E5MultilingualEmbedder
from rag.retrieval import ChromaVectorStore, BM25Store
from rag.retrieval.fusion import HybridRetriever
from rag.generation import format_context_blocks


def _du_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += (Path(root) / f).stat().st_size
            except Exception:
                pass
    return total


def retrieve_preview(
    *,
    question: str,
    filters: Mapping[str, object] | None = None,
    top_k: int = 8,
    hybrid: bool = True,
) -> List[Dict[str, object]]:
    """
    Returns a list of retrieved items with provenance, snippets, and scores — no generation.
    """
    cfg = load_config()
    vec_store = ChromaVectorStore.from_config()
    bm25_store = BM25Store.load_or_create("./indexes/bm25")
    embedder = E5MultilingualEmbedder(model_name=cfg.embedding_model_name)

    retriever = HybridRetriever(
        vector_store=vec_store,
        bm25_store=bm25_store,
        embedder=embedder,
        k_vector=int(cfg.k_vector),
        k_bm25=int(cfg.k_bm25),
        rrf_k=60,
        weight_vector=1.0,
        weight_bm25=1.0,
    )

    results = retriever.retrieve(
        question=question,
        filters=filters or {},
        top_k=int(top_k),
        hybrid=bool(hybrid),
    )

    # Build provenance list in the same order
    _ctx_text, prov = format_context_blocks(results, max_total_chars=None)
    preview = []
    for i, r in enumerate(results):
        doc = (r.get("document") or "")[:240].replace("\n", " ")
        meta = r.get("metadata") or {}
        sc = r.get("scores") or {}
        preview.append(
            {
                "n": i + 1,
                "id": r.get("id"),
                "prov": prov[i] if i < len(prov) else None,
                "snippet": doc,
                "scores": {
                    "fused": sc.get("fused"),
                    "vector_distance": sc.get("vector_distance"),
                    "bm25_score": sc.get("bm25_score"),
                },
                "metadata": meta,
            }
        )
    return preview


def index_stats() -> Dict[str, object]:
    """
    Quick index health: vector count and disk usage.
    """
    cfg = load_config()
    vec = ChromaVectorStore.from_config()
    chroma_count = 0
    try:
        chroma_count = vec.count()
    except Exception:
        chroma_count = -1  # signal unknown

    chroma_dir = cfg.chroma_persist_directory
    bm25_dir = Path("./indexes/bm25")

    return {
        "vector_count": int(chroma_count),
        "chroma": {
            "persist_dir": str(chroma_dir),
            "disk_bytes": _du_bytes(chroma_dir),
        },
        "bm25": {
            "persist_dir": str(bm25_dir.resolve()),
            "disk_bytes": _du_bytes(bm25_dir),
        },
        "collection": str(cfg.chroma_collection_name),
        "embedding_model": str(cfg.embedding_model_name),
    }
