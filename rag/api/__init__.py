"""Local HTTP API package.

`create_app` is resolved lazily so that importing `rag.api` does not require
FastAPI to be installed. The API is an optional extra (`pip install -e ".[api]"`),
and the CLI imports this package's siblings freely.
"""

from typing import Any

__all__ = ["create_app"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from rag.api.app import create_app

        return create_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
