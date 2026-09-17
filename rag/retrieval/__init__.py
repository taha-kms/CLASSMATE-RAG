"""Retrieval layer: Chroma vectors, BM25 lexical search, and RRF fusion.

Names resolve lazily; see rag._lazy for why.
"""

from typing import TYPE_CHECKING

from rag._lazy import lazy_exports

__all__ = [
    "ChromaVectorStore",
    "build_where_filter",
    "BM25Store",
    "rrf_fuse",
    "HybridRetriever",
]

__getattr__, __dir__ = lazy_exports(
    __name__,
    {
        "ChromaVectorStore": ".vector_chroma",
        "build_where_filter": ".vector_chroma",
        "BM25Store": ".bm25",
        "rrf_fuse": ".fusion",
        "HybridRetriever": ".fusion",
    },
)

if TYPE_CHECKING:  # pragma: no cover
    from .bm25 import BM25Store
    from .fusion import HybridRetriever, rrf_fuse
    from .vector_chroma import ChromaVectorStore, build_where_filter
