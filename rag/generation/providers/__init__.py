"""
Hosted generation providers.

Each registers itself on import and keeps its SDK import inside the
backend, so a provider nobody has selected costs nothing and an SDK nobody
has installed is not an error until it is chosen.
"""

from __future__ import annotations


def register_all() -> None:
    """Make every bundled provider selectable."""
    from . import (
        anthropic_backend,  # noqa: F401  (import registers it)
        openai_backend,  # noqa: F401
    )
