"""
Pipeline facade for CLASSMATE-RAG.

Why this file exists
--------------------
This package initializer acts as a *stable import surface* for code that uses
the end-to-end RAG pipeline. Instead of importing deep modules, callers can do:

    from rag.pipeline import ingest_file, ask_question, retrieve_preview, index_stats

What it re-exports
------------------
- ingest_file     : End-to-end ingestion of a document (load → chunk → embed → index)
                    Implemented in rag/pipeline/rag.py.

- ask_question    : Retrieval + generation for a user question, returning an
                    answer (with citations), language, and provenance list.
                    Implemented in rag/pipeline/rag.py.

- retrieve_preview: Retrieval-only preview (ranked items, scores, provenance).
                    Implemented in rag/admin/inspect.py and re-exported via rag.admin.

- index_stats     : Quick index health snapshot (vector count, disk usage).
                    Implemented in rag/admin/inspect.py and re-exported via rag.admin.

Keeping this file small and explicit lets the CLI and other callers rely on
`rag.pipeline` as the single entrypoint for the pipeline, without knowing where
internals live. (The project’s CLI imports exactly these four symbols from here.)
"""

# Re-export the core pipeline entrypoints from the concrete implementation.
from .rag import (
    ingest_file,
    ask_question,
)

# Re-export admin/observability helpers so callers can find them under
# the same 'rag.pipeline' namespace.
from rag.admin import (
    retrieve_preview,
    index_stats,
)

# Public API of this package: keep it explicit and minimal.
__all__ = [
    "ingest_file",
    "ask_question",
    "retrieve_preview",
    "index_stats",
]
