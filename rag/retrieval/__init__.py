"""Retrieval layer: Chroma vectors, BM25 lexical search, and RRF fusion.

Names are resolved lazily. Importing this package used to pull in
sentence-transformers and chromadb through .fusion, which meant nothing under
rag/retrieval could be imported, or tested, without the whole ML stack
installed. The public surface is unchanged: `from rag.retrieval import
BM25Store` still works, it just no longer drags the embedder in with it.
"""

from typing import TYPE_CHECKING

__all__ = [
    "ChromaVectorStore",
    "build_where_filter",
    "BM25Store",
    "rrf_fuse",
    "HybridRetriever",
]

# Attribute name -> module it lives in, relative to this package.
_EXPORTS = {
    "ChromaVectorStore": ".vector_chroma",
    "build_where_filter": ".vector_chroma",
    "BM25Store": ".bm25",
    "rrf_fuse": ".fusion",
    "HybridRetriever": ".fusion",
}

if TYPE_CHECKING:  # pragma: no cover - for type checkers and editors only
    from .bm25 import BM25Store
    from .fusion import HybridRetriever, rrf_fuse
    from .vector_chroma import ChromaVectorStore, build_where_filter


def __getattr__(name: str):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value  # cache so later lookups skip __getattr__
    return value


def __dir__():
    return sorted(__all__)
