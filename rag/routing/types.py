"""
Routing types: the four canonical routes and the decision record returned
by the hybrid router.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Tuple

# Canonical route names. The string values are stored in chunk metadata
# under the `subject` key, so changing them is a breaking change for
# previously-ingested data.
Route = Literal["math", "code", "translation", "default"]

ROUTES: Tuple[Route, ...] = ("math", "code", "translation", "default")
DEFAULT_ROUTE: Route = "default"


@dataclass(frozen=True)
class RouteDecision:
    """
    Record produced by HybridRouter.decide() for one query.

    Attributes:
        route:        chosen route (one of ROUTES)
        reason:       short tag explaining why ("query_confident",
                      "metadata_override", "ambiguous_default",
                      "translation_intent", "forced", ...)
        query_scores: cosine similarity from query → each subject prototype
        meta_scores:  fraction of top-k retrieved chunks tagged per subject
        margin:       top1 - top2 of query_scores
    """

    route: Route
    reason: str
    query_scores: Dict[Route, float] = field(default_factory=dict)
    meta_scores: Dict[Route, float] = field(default_factory=dict)
    margin: float = 0.0

    def short_log(self) -> str:
        """One-line summary for logs / debugging."""
        q = ",".join(f"{k}:{v:.2f}" for k, v in self.query_scores.items())
        m = ",".join(f"{k}:{v:.2f}" for k, v in self.meta_scores.items())
        return (
            f"route={self.route} reason={self.reason} "
            f"margin={self.margin:.2f} Q={{{q}}} M={{{m}}}"
        )
