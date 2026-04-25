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

from .types import Route, RouteDecision, ROUTES, DEFAULT_ROUTE
from .prototypes import SUBJECT_PROTOTYPES
from .classifier import SubjectClassifier
from .router import HybridRouter
from .registry import ModelSpec, get_model_spec, route_model_paths
from .loader import StickyModelLoader
from .prompts import system_prompt_for

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
