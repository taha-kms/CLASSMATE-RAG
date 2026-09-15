"""Helper for packages whose submodules pull in the heavy ML stack.

Importing a package used to import everything it re-exported, so reaching
`rag.retrieval.BM25Store` dragged in sentence-transformers through
`.fusion`, and reaching anything in `rag.generation` dragged in llama_cpp.
That made most of the codebase impossible to import, and therefore to test,
without torch, chromadb and a compiled llama.cpp present.

`lazy_exports` wires up the PEP 562 module `__getattr__` that resolves each
name on first use instead. The public surface is unchanged: callers still
write `from rag.generation import enforce_citations`, it just no longer
costs an import of llama_cpp to get there.
"""

from __future__ import annotations

from importlib import import_module
from typing import Callable, Dict, Tuple


def lazy_exports(
    package: str,
    exports: Dict[str, str],
) -> Tuple[Callable[[str], object], Callable[[], list]]:
    """
    Build the `__getattr__` and `__dir__` a lazy package needs.

    `exports` maps each public name to the module it lives in, relative to
    the package (".fusion", ".post" and so on). Usage:

        __getattr__, __dir__ = lazy_exports(__name__, {"rrf_fuse": ".fusion"})
    """

    def __getattr__(name: str) -> object:
        module_name = exports.get(name)
        if module_name is None:
            raise AttributeError(f"module {package!r} has no attribute {name!r}")
        value = getattr(import_module(module_name, package), name)
        # Cache on the module so subsequent lookups skip __getattr__ entirely.
        import sys

        setattr(sys.modules[package], name, value)
        return value

    def __dir__() -> list:
        return sorted(exports)

    return __getattr__, __dir__
