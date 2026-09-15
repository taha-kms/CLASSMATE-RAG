"""
Subject-aware routing for CLASSMATE-RAG.

Pipeline integration:
- At ingest: SubjectClassifier.classify_text() assigns a `subject` to each
  document (or each chunk-sample) and writes it into chunk metadata.
- At ask:    HybridRouter.decide() consumes the user query and the retrieved
  chunks' subject metadata, returns a RouteDecision; StickyModelLoader
  loads the matching GGUF (swapping if the route changed) and serves
  the chat call.

All four routes use the same E5 embedding model and the same prototype
phrases, so ingest-time and query-time classification are consistent.
"""

from typing import TYPE_CHECKING

from rag._lazy import lazy_exports

__all__ = [
    "Route",
    "RouteDecision",
    "ROUTES",
    "DEFAULT_ROUTE",
    "SUBJECT_PROTOTYPES",
    "SubjectClassifier",
    "HybridRouter",
    "ModelSpec",
    "get_model_spec",
    "route_model_paths",
    "StickyModelLoader",
    "system_prompt_for",
]

# .classifier needs sentence-transformers and .loader needs llama_cpp. The
# route types, prompts and registry are pure Python and stay reachable
# without either.
__getattr__, __dir__ = lazy_exports(__name__, {
    "Route": ".types",
    "RouteDecision": ".types",
    "ROUTES": ".types",
    "DEFAULT_ROUTE": ".types",
    "SUBJECT_PROTOTYPES": ".prototypes",
    "SubjectClassifier": ".classifier",
    "HybridRouter": ".router",
    "ModelSpec": ".registry",
    "get_model_spec": ".registry",
    "route_model_paths": ".registry",
    "StickyModelLoader": ".loader",
    "system_prompt_for": ".prompts",
})

if TYPE_CHECKING:  # pragma: no cover
    from .classifier import SubjectClassifier
    from .loader import StickyModelLoader
    from .prompts import system_prompt_for
    from .prototypes import SUBJECT_PROTOTYPES
    from .registry import ModelSpec, get_model_spec, route_model_paths
    from .router import HybridRouter
    from .types import DEFAULT_ROUTE, ROUTES, Route, RouteDecision
