"""
Hybrid route resolver.

Combines two signals:
- Q[route]: cosine similarity from query → subject prototype (classifier)
- M[route]: fraction of top-k retrieved chunks tagged with each subject

Resolution rules (Section 2 of the design):

  1. If the query is confident (top1 - top2 >= query_margin):
       - Use top-1 query route, even if metadata disagrees.
       - Honor the translation-intent guard for the translation route.
  2. Else if metadata is confident (top-fraction >= metadata_threshold):
       - Use the top metadata route.
  3. Else:
       - Fall back to DEFAULT_ROUTE.

A forced subject (e.g. set explicitly by the user) short-circuits to "forced".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

from .classifier import SubjectClassifier
from .prototypes import TRANSLATION_INTENT_KEYWORDS
from .types import DEFAULT_ROUTE, ROUTES, Route, RouteDecision


@dataclass
class HybridRouter:
    """
    Decides the route for a single query. Stateless across calls — the same
    instance can be reused.
    """

    classifier: SubjectClassifier
    query_margin: float = 0.10
    metadata_threshold: float = 0.60
    translation_requires_intent: bool = True

    def decide(
        self,
        question: str,
        retrieved_metas: Sequence[Dict[str, object]] | None = None,
        *,
        forced_subject: Optional[Route] = None,
    ) -> RouteDecision:
        # Hard override: user (or upstream config) named the route.
        if forced_subject in ROUTES:
            return RouteDecision(
                route=forced_subject,  # type: ignore[arg-type]
                reason="forced",
                query_scores={r: 0.0 for r in ROUTES},
                meta_scores=_meta_fractions(retrieved_metas or []),
                margin=0.0,
            )

        q_scores = self.classifier.score_query(question)
        m_scores = _meta_fractions(retrieved_metas or [])

        top_q_route, top_q_score, second_q_score = _top_two(q_scores)
        margin = top_q_score - second_q_score

        # 1) Query-confident path
        if top_q_route is not None and margin >= self.query_margin:
            chosen, reason = self._guard_translation(
                top_q_route, question, q_scores, m_scores, margin,
            )
            return RouteDecision(
                route=chosen,
                reason=reason,
                query_scores=q_scores,
                meta_scores=m_scores,
                margin=margin,
            )

        # 2) Metadata-confident path (query was ambiguous)
        top_m_route, top_m_frac = _top_fraction(m_scores)
        if top_m_route is not None and top_m_frac >= self.metadata_threshold:
            chosen, reason = self._guard_translation(
                top_m_route, question, q_scores, m_scores, margin,
                base_reason="metadata_override",
            )
            return RouteDecision(
                route=chosen,
                reason=reason,
                query_scores=q_scores,
                meta_scores=m_scores,
                margin=margin,
            )

        # 3) Both ambiguous → safe default
        return RouteDecision(
            route=DEFAULT_ROUTE,
            reason="ambiguous_default",
            query_scores=q_scores,
            meta_scores=m_scores,
            margin=margin,
        )

    # ------------------------------------------------------------------
    # Translation guard
    # ------------------------------------------------------------------

    def _guard_translation(
        self,
        proposed: Route,
        question: str,
        q_scores: Dict[Route, float],
        m_scores: Dict[Route, float],
        margin: float,
        *,
        base_reason: str = "query_confident",
    ) -> tuple[Route, str]:
        """
        SalamandraTA is translation-only. Even if its prototype score wins,
        require an explicit translate-intent keyword. Otherwise demote to
        the default route, which can also handle Italian via Qwen3.
        """
        if proposed != "translation":
            return proposed, base_reason
        if not self.translation_requires_intent:
            return proposed, "translation_intent"
        if _has_translation_intent(question):
            return proposed, "translation_intent"
        return DEFAULT_ROUTE, "translation_demoted_no_intent"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _meta_fractions(retrieved_metas: Sequence[Dict[str, object]]) -> Dict[Route, float]:
    """
    Count the `subject` tag across retrieved chunks and return per-route
    fractions. Chunks without a tag don't contribute.
    """
    counts: Dict[Route, int] = {r: 0 for r in ROUTES}
    tagged = 0
    for meta in retrieved_metas:
        if not isinstance(meta, dict):
            continue
        s = meta.get("subject")
        if isinstance(s, str) and s in counts:
            counts[s] += 1  # type: ignore[index]
            tagged += 1
    if tagged == 0:
        return {r: 0.0 for r in ROUTES}
    return {r: counts[r] / tagged for r in ROUTES}


def _top_two(scores: Dict[Route, float]) -> tuple[Optional[Route], float, float]:
    """Return (top_route, top_score, second_score). Excludes 0.0 scores."""
    cand = [(r, s) for r, s in scores.items() if s > 0.0]
    if not cand:
        return None, 0.0, 0.0
    cand.sort(key=lambda kv: kv[1], reverse=True)
    top_r, top_s = cand[0]
    second = cand[1][1] if len(cand) > 1 else 0.0
    return top_r, top_s, second


def _top_fraction(scores: Dict[Route, float]) -> tuple[Optional[Route], float]:
    cand = [(r, s) for r, s in scores.items() if s > 0.0]
    if not cand:
        return None, 0.0
    cand.sort(key=lambda kv: kv[1], reverse=True)
    return cand[0]


def _has_translation_intent(question: str) -> bool:
    if not question:
        return False
    q = question.lower()
    return any(kw in q for kw in TRANSLATION_INTENT_KEYWORDS)
