"""
Neighbor expansion and document-level diversity helpers.

- Expands retrieved results with neighbor chunks (same file, chunk_id ± radius).
- Enforces a per-document cap to encourage diversity.

This module reads the BM25 catalog JSONL to fetch neighbor texts/metadata
without needing vector-store round trips.

Inputs/Outputs use the common retrieval dict shape:
    { "id": str, "document": str, "score": float, "metadata": dict }
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from rag.utils import stable_chunk_id

_BM25_JSONL = Path("./indexes/bm25/bm25_index.jsonl")


@dataclass
class Retrieved:
    id: str
    document: str
    score: float
    metadata: Dict[str, object]

    def to_dict(self) -> Dict[str, object]:
        return {"id": self.id, "document": self.document, "score": self.score, "metadata": self.metadata}


def _load_bm25_catalog() -> Dict[str, Tuple[str, Dict[str, object]]]:
    """
    Returns a dict: id -> (text, metadata)
    If the catalog file is missing, returns an empty dict.
    """
    out: Dict[str, Tuple[str, Dict[str, object]]] = {}
    if not _BM25_JSONL.exists():
        return out
    with _BM25_JSONL.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                cid = str(obj.get("id") or "")
                if not cid:
                    continue
                txt = str(obj.get("text") or "")
                meta = obj.get("metadata") or {}
                out[cid] = (txt, dict(meta))
            except Exception:
                continue
    return out


def _neighbor_ids(meta: Dict[str, object], *, radius: int) -> List[str]:
    """
    Build neighbor IDs for the same source file using stable_chunk_id.
    Requires: source_path, page, chunk_id (current), optionally course/unit to match hashing.
    """
    sp = meta.get("source_path")
    page = meta.get("page")
    cid = meta.get("chunk_id")
    if sp is None or page is None or cid is None:
        return []
    try:
        base_page = int(page)
        base_cid = int(cid)
    except Exception:
        return []

    course = meta.get("course") or None
    unit = meta.get("unit") or None

    nids: List[str] = []
    for d in range(-radius, radius + 1):
        if d == 0:
            continue
        nids.append(
            stable_chunk_id(
                source_path=Path(str(sp)),
                page=base_page,
                chunk_index=base_cid + d,
                course=course,
                unit=unit,
            )
        )
    return nids


def expand_with_neighbors(
    results: Sequence[Dict[str, object]],
    *,
    radius: int = 1,
    max_per_doc: Optional[int] = None,
    neighbor_penalty: float = 0.001,
) -> List[Dict[str, object]]:
    """
    Expand 'results' with neighbor chunks (same file, c±radius), then enforce per-document cap.
    - Neighbor entries inherit metadata from the catalog.
    - Neighbor scores are original_score - neighbor_penalty (kept just below the seed hit).
    - Deduplicates by ID, preserving original ordering as much as possible.
    """
    catalog = _load_bm25_catalog()
    seen_ids = set()
    expanded: List[Retrieved] = []

    # Seed: copy originals first
    for r in results:
        rid = str(r.get("id") or "")
        doc = str(r.get("document") or "")
        sc = float(r.get("score") or 0.0)
        meta = dict(r.get("metadata") or {})
        if not rid:
            continue
        if rid in seen_ids:
            continue
        seen_ids.add(rid)
        expanded.append(Retrieved(id=rid, document=doc, score=sc, metadata=meta))

        # Neighbors
        if radius > 0:
            for nid in _neighbor_ids(meta, radius=radius):
                if nid in seen_ids:
                    continue
                if nid not in catalog:
                    continue
                ntext, nmeta = catalog[nid]
                if not (ntext or "").strip():
                    continue
                expanded.append(Retrieved(id=nid, document=ntext, score=sc - neighbor_penalty, metadata=nmeta))
                seen_ids.add(nid)

    # Enforce per-document cap (diversity)
    if max_per_doc and max_per_doc > 0:
        counts: Dict[str, int] = {}
        kept: List[Retrieved] = []
        for it in expanded:
            sp = str(it.metadata.get("source_path") or "")
            cnt = counts.get(sp, 0)
            if cnt < max_per_doc:
                kept.append(it)
                counts[sp] = cnt + 1
        expanded = kept

    return [it.to_dict() for it in expanded]
