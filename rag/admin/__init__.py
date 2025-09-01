"""
Admin package public API.

This package groups admin/observability helpers (previewing retrieval results,
showing index stats, etc.). To give callers a clean import surface without
diving into submodules, we re-export the two main entry points here:

- retrieve_preview : Run retrieval only (no generation) and return ranked items
                     with provenance/snippets/scores. Implemented in
                     rag/admin/inspect.py.
- index_stats      : Quick index health snapshot (vector count, disk usage).
                     Implemented in rag/admin/inspect.py.

Keeping these symbols at the package level allows convenient imports like:

    from rag.admin import retrieve_preview, index_stats
"""

# Re-export the concrete implementations from the local 'inspect' module.
from .inspect import (
    retrieve_preview,
    index_stats,
)

# Explicitly declare the public API of this package to avoid leaking internals.
__all__ = [
    "retrieve_preview",
    "index_stats",
]
