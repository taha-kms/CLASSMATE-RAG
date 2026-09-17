"""
Simple near-duplicate chunk filtering.

We approximate near-duplicates by Jaccard similarity between token 5-gram shingles.
This is fast and robust enough at per-file scale.

Public API:
    dedup_text_blocks(blocks: list[str], *, jaccard_threshold=0.92) -> list[str]
"""

from __future__ import annotations

import re
from typing import Iterable, List, Set, Tuple

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def _norm_tokens(s: str) -> list[str]:
    s = (s or "").lower()
    s = _PUNCT.sub(" ", s)
    s = _WS.sub(" ", s).strip()
    return s.split() if s else []


def _shingles(tokens: Iterable[str], k: int = 5) -> Set[Tuple[str, ...]]:
    toks = list(tokens)
    if len(toks) < k:
        return {tuple(toks)} if toks else set()
    return {tuple(toks[i : i + k]) for i in range(0, len(toks) - k + 1)}


def _jaccard(a: Set[Tuple[str, ...]], b: Set[Tuple[str, ...]]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def dedup_block_indices(blocks: List[str], *, jaccard_threshold: float = 0.92) -> List[int]:
    """
    Indices of the blocks to keep, in order.

    Returning positions rather than text lets callers filter a parallel list
    (page numbers, chunk ids) without matching strings back up afterwards,
    which is both fragile and quadratic when blocks repeat.
    """
    kept: List[int] = []
    kept_sh: List[Set[Tuple[str, ...]]] = []

    for i, text in enumerate(blocks):
        sh = _shingles(_norm_tokens(text), k=5)
        if any(_jaccard(sh, ksh) >= jaccard_threshold for ksh in kept_sh):
            continue
        kept.append(i)
        kept_sh.append(sh)

    return kept


def dedup_text_blocks(blocks: List[str], *, jaccard_threshold: float = 0.92) -> List[str]:
    """
    Preserve order; drop any block whose shingle Jaccard with any previously
    kept block exceeds threshold.
    """
    keep = dedup_block_indices(blocks, jaccard_threshold=jaccard_threshold)
    return [blocks[i] for i in keep]
