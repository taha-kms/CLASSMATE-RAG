"""
Tools to check and debug the RAG indexes.

Functions:
- retrieve_preview: run retrieval only (no LLM) and show what would be returned.
- index_stats: report vector count and disk usage for indexes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping

from rag.config import load_config
from rag.embeddings import E5MultilingualEmbedder
from rag.retrieval import ChromaVectorStore, BM25Store
from rag.retrieval.fusion import HybridRetriever
from rag.generation import format_context_blocks


# ------------------------------
# Helpers
# ------------------------------

def _du_bytes(path: Path) -> int:
    """Return disk usage (in bytes) of a file or directory."""
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


# ------------------------------
# Public functions
# ------------------------------

def retrieve_preview(
    *,
    question: str,
    filters: Mapping[str, object] | None = None,
    top_k: int = 8,
    hybrid: bool = True,
) -> List[Dict[str, object]]:
    """
    Run retrieval only (no generation).
    Returns a list of items with:
    - provenance
    - snippet of text
    - fused/vector/BM25 scores
    - metadata
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

    # Format provenance for preview
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
    Report index health.
    Returns:
    - vector count
    - disk usage for Chroma and BM25
    - collection name
    - embedding model
    """
    cfg = load_config()
    vec = ChromaVectorStore.from_config()
    chroma_count = 0
    try:
        chroma_count = vec.count()
    except Exception:
        chroma_count = -1  # unknown / error

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
