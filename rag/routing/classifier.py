"""
Embedding-prototype subject classifier.

Used in two places:
- Ingest: classify_text(doc_sample) → subject tag for chunk metadata.
- Ask:    score_query(question) → per-route cosine similarity dict consumed
          by the HybridRouter.

Implementation notes:
- Each route's prototype is the L2-normalized mean of its seed embeddings.
- E5 expects different prefixes for queries vs. passages, so seed phrases
  are treated as queries (they describe what a query about that subject
  looks like). Documents are treated as passages.
- Prototype embeddings are computed once at construction and cached on the
  instance — no per-query model calls beyond the input encoding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from rag.embeddings import E5MultilingualEmbedder

from .prototypes import SUBJECT_PROTOTYPES
from .types import DEFAULT_ROUTE, ROUTES, Route


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


@dataclass
class ClassificationResult:
    """Returned by classify_text(): a subject and the full score map."""
    subject: Route
    scores: Dict[Route, float]
    margin: float


class SubjectClassifier:
    """
    Zero-shot route classifier driven by E5 embeddings of seed phrases.

    Reuses an externally-supplied embedder when given (so the ingest pipeline
    doesn't load E5 twice). Builds its own when called standalone.
    """

    def __init__(
        self,
        embedder: Optional[E5MultilingualEmbedder] = None,
        prototypes: Optional[Dict[Route, List[str]]] = None,
    ) -> None:
        self.embedder = embedder or E5MultilingualEmbedder()
        self._prototype_map: Dict[Route, np.ndarray] = {}
        self._build_prototypes(prototypes or SUBJECT_PROTOTYPES)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build_prototypes(self, proto: Dict[Route, List[str]]) -> None:
        """Embed each route's seed phrases and store the L2-normalized mean."""
        for route in ROUTES:
            seeds = proto.get(route, [])
            if not seeds:
                # default route has no prototype — handled by elimination
                continue
            # Seeds describe queries; encode them with the query prefix.
            vecs = self.embedder.encode_queries(seeds)
            mean = vecs.mean(axis=0)
            self._prototype_map[route] = _l2_normalize(mean.astype("float32"))

    # ------------------------------------------------------------------
    # Public scoring API
    # ------------------------------------------------------------------

    def score_query(self, question: str) -> Dict[Route, float]:
        """
        Cosine similarity of the question against each prototype.
        Routes without a prototype (e.g. "default") get score 0.0.
        """
        if not question or not question.strip():
            return {r: 0.0 for r in ROUTES}
        q = self.embedder.encode_queries([question])[0]
        q = _l2_normalize(q.astype("float32"))
        return {
            r: float(np.dot(q, self._prototype_map[r])) if r in self._prototype_map else 0.0
            for r in ROUTES
        }

    def score_passage(self, text: str) -> Dict[Route, float]:
        """Cosine similarity of a passage (document chunk) against each prototype."""
        if not text or not text.strip():
            return {r: 0.0 for r in ROUTES}
        p = self.embedder.encode_passages([text])[0]
        p = _l2_normalize(p.astype("float32"))
        return {
            r: float(np.dot(p, self._prototype_map[r])) if r in self._prototype_map else 0.0
            for r in ROUTES
        }

    def classify_text(
        self,
        text: str,
        *,
        min_margin: float = 0.05,
    ) -> ClassificationResult:
        """
        Assign a single subject to a passage. If the top-1 score isn't
        clearly above top-2 (margin < min_margin), return DEFAULT_ROUTE.
        Used at ingest time to tag documents.
        """
        scores = self.score_passage(text)
        top_route, top_score, margin = _top_with_margin(scores)
        if top_route is None or margin < min_margin:
            return ClassificationResult(subject=DEFAULT_ROUTE, scores=scores, margin=margin)
        return ClassificationResult(subject=top_route, scores=scores, margin=margin)

    def classify_chunks(
        self,
        chunk_texts: Iterable[str],
        *,
        sample_size: int = 8,
        min_margin: float = 0.05,
    ) -> ClassificationResult:
        """
        Pool-classify a document by averaging passage scores across a sample
        of its chunks. Cheaper and more stable than classifying every chunk.
        """
        texts: List[str] = [t for t in chunk_texts if t and t.strip()]
        if not texts:
            return ClassificationResult(
                subject=DEFAULT_ROUTE,
                scores={r: 0.0 for r in ROUTES},
                margin=0.0,
            )
        # Evenly-spaced sample to cover the document, not just its head.
        if len(texts) > sample_size:
            step = max(1, len(texts) // sample_size)
            sampled = texts[::step][:sample_size]
        else:
            sampled = texts

        agg: Dict[Route, float] = {r: 0.0 for r in ROUTES}
        for t in sampled:
            for r, s in self.score_passage(t).items():
                agg[r] += s
        n = float(len(sampled))
        scores = {r: v / n for r, v in agg.items()}
        top_route, _, margin = _top_with_margin(scores)
        if top_route is None or margin < min_margin:
            return ClassificationResult(subject=DEFAULT_ROUTE, scores=scores, margin=margin)
        return ClassificationResult(subject=top_route, scores=scores, margin=margin)


def _top_with_margin(scores: Dict[Route, float]) -> Tuple[Optional[Route], float, float]:
    """
    Return (top_route, top_score, margin = top1 - top2). Routes with a
    score of 0.0 (no prototype) are excluded from the top-N selection,
    so the default route never wins by default.
    """
    candidates = [(r, s) for r, s in scores.items() if s > 0.0]
    if not candidates:
        return None, 0.0, 0.0
    candidates.sort(key=lambda kv: kv[1], reverse=True)
    top_route, top_score = candidates[0]
    second_score = candidates[1][1] if len(candidates) > 1 else 0.0
    return top_route, top_score, top_score - second_score
