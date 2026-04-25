"""
Admin package public API.

Re-exports `retrieve_preview` and `index_stats` from `rag.admin.inspect`, but
loads them lazily so that importing lightweight siblings (e.g.
`rag.admin.manage._matches_simple` from the unit tests) does not transitively
pull the heavy ML/IO stack required by `inspect`.
"""

from typing import Any

__all__ = ["retrieve_preview", "index_stats"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from .inspect import retrieve_preview, index_stats
        return {"retrieve_preview": retrieve_preview, "index_stats": index_stats}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
