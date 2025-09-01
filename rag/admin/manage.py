"""
Helpers to manage the corpus across BM25 and Chroma indexes.

Provides functions to:
- list entries
- show entries by ID
- resolve chunk IDs
- delete chunks
- reingest files
- list source paths

BM25 JSONL file is used as the main catalog. 
All operations are safe to repeat (idempotent).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from rag.config import load_config
from rag.metadata import DocumentMetadata
from rag.pipeline import ingest_file
from rag.retrieval import ChromaVectorStore, BM25Store


# ------------------------------
# Constants
# ------------------------------

_BM25_DIR = Path("./indexes/bm25")
_BM25_JSONL = _BM25_DIR / "bm25_index.jsonl"


# ------------------------------
# Data type
# ------------------------------

@dataclass(frozen=True)
class CatalogEntry:
    """Represents one entry (chunk) in the BM25 catalog."""
    id: str
    text: str
    metadata: Dict[str, object]


# ------------------------------
# Internal helpers
# ------------------------------

def _read_bm25_catalog() -> List[CatalogEntry]:
    """Read BM25 catalog JSONL and return a list of entries."""
    out: List[CatalogEntry] = []
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
                txt = str(obj.get("text") or "")
                meta = obj.get("metadata") or {}
                if cid:
                    out.append(CatalogEntry(id=cid, text=txt, metadata=dict(meta)))
            except Exception:
                continue
    return out


def _matches_simple(meta: Mapping[str, object], where: Mapping[str, object]) -> bool:
    """Check if metadata matches filter values (simple equality and tag checks)."""
    if not where:
        return True
    for k, v in where.items():
        if v is None:
            continue
        if k == "tags":
            # Tags can be string or list; check tag_* flags in metadata
            tags: List[str] = []
            if isinstance(v, (list, tuple)):
                tags = [str(x).strip().lower() for x in v if str(x).strip()]
            else:
                tags = [p.strip().lower() for p in str(v).split(",") if p.strip()]
            for t in tags:
                if not meta.get(f"tag_{t}", False):
                    return False
            continue
        if str(meta.get(k, "")).strip() != str(v).strip():
            return False
    return True


def _collect_source_paths(entries: Sequence[CatalogEntry]) -> List[str]:
    """Return unique source_path values from catalog entries."""
    paths: List[str] = []
    seen: set[str] = set()
    for e in entries:
        sp = str(e.metadata.get("source_path") or "").strip()
        if sp and sp not in seen:
            seen.add(sp)
            paths.append(sp)
    return paths


def _group_by_source(entries: Sequence[CatalogEntry]) -> Dict[str, List[CatalogEntry]]:
    """Group entries by their source_path."""
    by: Dict[str, List[CatalogEntry]] = {}
    for e in entries:
        sp = str(e.metadata.get("source_path") or "")
        by.setdefault(sp, []).append(e)
    return by


# ------------------------------
# Public API
# ------------------------------

def list_entries(
    *,
    where: Optional[Mapping[str, object]] = None,
    limit: Optional[int] = None,
    offset: int = 0,
) -> List[CatalogEntry]:
    """List catalog entries filtered by metadata, with optional paging."""
    cat = _read_bm25_catalog()
    filt = [e for e in cat if _matches_simple(e.metadata, where or {})]
    if offset < 0:
        offset = 0
    if limit is None or limit <= 0:
        return filt[offset:]
    return filt[offset : offset + limit]


def show_entries_by_id(ids: Iterable[str]) -> List[CatalogEntry]:
    """Return catalog entries matching the given IDs (preserve order)."""
    want = [str(x) for x in ids]
    if not want:
        return []
    cat = _read_bm25_catalog()
    index = {e.id: e for e in cat}
    out: List[CatalogEntry] = []
    for i in want:
        e = index.get(i)
        if e:
            out.append(e)
    return out


def resolve_ids(
    *,
    ids: Optional[Iterable[str]] = None,
    where: Optional[Mapping[str, object]] = None,
    path: Optional[str] = None,
) -> List[str]:
    """
    Get chunk IDs from ids, path, or filter.
    - If ids provided → use directly (if they exist).
    - If path provided → select all from that file.
    - Otherwise → filter by metadata.
    """
    cat = _read_bm25_catalog()
    if ids:
        index = {e.id: e for e in cat}
        return [i for i in ids if i in index]

    if path:
        path = str(Path(path).resolve())
        return [e.id for e in cat if str(e.metadata.get("source_path") or "") == path]

    return [e.id for e in cat if _matches_simple(e.metadata, where or {})]


def delete_by_ids(ids: Sequence[str]) -> Tuple[int, int]:
    """
    Delete given chunk IDs from both vector and BM25 stores.
    Returns (num_deleted_from_vector, num_deleted_from_bm25).
    """
    if not ids:
        return (0, 0)

    vec = ChromaVectorStore.from_config()
    n_vec = 0
    try:
        n_vec = vec.delete(ids=list(ids)) or 0
    except Exception:
        n_vec = len(ids)

    bm = BM25Store.load_or_create(str(_BM25_DIR))
    n_bm25 = bm.delete_many(ids=list(ids))
    bm.save()

    return (n_vec, n_bm25)


def reingest_paths(paths: Sequence[str]) -> List[Dict[str, object]]:
    """
    Reingest files by their paths.
    Metadata is inferred from existing catalog entries:
    - first non-empty values for fields
    - tags combined as a union
    Returns a list of ingest results.
    """
    if not paths:
        return []

    cat = _read_bm25_catalog()
    by_path = _group_by_source(cat)

    results: List[Dict[str, object]] = []
    for raw in paths:
        p = Path(raw).expanduser().resolve()
        key = str(p)
        mlist = by_path.get(key, [])
        course = unit = language = doc_type = author = semester = None
        tags_set: set[str] = set()

        for e in mlist:
            md = e.metadata
            course = course or md.get("course")
            unit = unit or md.get("unit")
            language = language or md.get("language")
            doc_type = doc_type or md.get("doc_type")
            author = author or md.get("author")
            semester = semester or md.get("semester")
            for mk, mv in md.items():
                if mk.startswith("tag_") and mv:
                    tags_set.add(mk[len("tag_") :])

        # Build DocumentMetadata (language/doc_type enums accept strings from normalize)
        doc_meta = DocumentMetadata(
            course=course or None,
            unit=unit or None,
            language=language or None,
            doc_type=doc_type or None,
            author=author or None,
            semester=semester or None,
            tags=sorted(tags_set) or None,
            source_path=None,
            created_at=None,
        )

        res = ingest_file(path=str(p), doc_meta=doc_meta)
        results.append({
            "path": res.path,
            "doc_type": res.doc_type,
            "total_pages": res.total_pages,
            "total_chunks": res.total_chunks,
            "upserted": res.upserted,
            "created_at": res.created_at,
        })
    return results


def list_source_paths(
    *,
    where: Optional[Mapping[str, object]] = None
) -> List[str]:
    """Return all unique source_path values matching a filter."""
    entries = list_entries(where=where)
    return _collect_source_paths(entries)
