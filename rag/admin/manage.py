"""
Corpus management helpers for CLASSMATE-RAG.

Provides list/show/delete/reingest operations that work across BOTH
Chroma (vector index) and the BM25 lexical index.

Design goals
- Deterministic, metadata-driven curation.
- Safe delete (both stores), with --dry-run preview support at the CLI layer.
- Minimal coupling: BM25 JSONL acts as a readable "catalog" of chunks.
- Idempotent operations: re-running is safe.

Notes
- BM25 persistence is expected at: ./indexes/bm25/bm25_index.jsonl
  (See rag/retrieval/bm25.py docstring.)
- Vector store operations are invoked via ChromaVectorStore wrapper.

This module intentionally has NO print/log side effects:
the CLI layer is responsible for user I/O.
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


# ---- Constants ----

_BM25_DIR = Path("./indexes/bm25")
_BM25_JSONL = _BM25_DIR / "bm25_index.jsonl"


# ---- Data types ----

@dataclass(frozen=True)
class CatalogEntry:
    id: str
    text: str
    metadata: Dict[str, object]


# ---- Internal helpers ----

def _read_bm25_catalog() -> List[CatalogEntry]:
    """
    Read the BM25 JSONL catalog. Each line is an object with at least:
      { "id": str, "text": str, "metadata": { ... } }
    Returns empty list if file missing.
    """
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
                # ignore malformed lines
                continue
    return out


def _matches_simple(meta: Mapping[str, object], where: Mapping[str, object]) -> bool:
    """
    Simple equality-based filtering on known fields, plus tag_* booleans.
    Values in 'where' that are None/empty are ignored.
    """
    if not where:
        return True
    for k, v in where.items():
        if v is None:
            continue
        if k == "tags":
            # tags can be comma-separated or list -> expand to tag_* booleans
            tags: List[str] = []
            if isinstance(v, (list, tuple)):
                tags = [str(x).strip().lower() for x in v if str(x).strip()]
            else:
                tags = [p.strip().lower() for p in str(v).split(",") if p.strip()]
            for t in tags:
                if not meta.get(f"tag_{t}", False):
                    return False
            continue
        # regular equality
        if str(meta.get(k, "")).strip() != str(v).strip():
            return False
    return True


def _collect_source_paths(entries: Sequence[CatalogEntry]) -> List[str]:
    paths: List[str] = []
    seen: set[str] = set()
    for e in entries:
        sp = str(e.metadata.get("source_path") or "").strip()
        if sp and sp not in seen:
            seen.add(sp)
            paths.append(sp)
    return paths


def _group_by_source(entries: Sequence[CatalogEntry]) -> Dict[str, List[CatalogEntry]]:
    by: Dict[str, List[CatalogEntry]] = {}
    for e in entries:
        sp = str(e.metadata.get("source_path") or "")
        by.setdefault(sp, []).append(e)
    return by


# ---- Public API ----

def list_entries(
    *,
    where: Optional[Mapping[str, object]] = None,
    limit: Optional[int] = None,
    offset: int = 0,
) -> List[CatalogEntry]:
    """
    Return catalog entries filtered by metadata, with optional paging.
    """
    cat = _read_bm25_catalog()
    filt = [e for e in cat if _matches_simple(e.metadata, where or {})]
    if offset < 0:
        offset = 0
    if limit is None or limit <= 0:
        return filt[offset:]
    return filt[offset : offset + limit]


def show_entries_by_id(ids: Iterable[str]) -> List[CatalogEntry]:
    """
    Return catalog entries for the given IDs, preserving input order.
    """
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
    Resolve a concrete list of chunk IDs by (ids | path | where).
    - If 'ids' provided, use them directly (filtering out unknown).
    - Else if 'path' provided, select all chunks from that source_path.
    - Else use 'where' filter.
    """
    cat = _read_bm25_catalog()
    if ids:
        index = {e.id: e for e in cat}
        return [i for i in ids if i in index]

    if path:
        path = str(Path(path).resolve())
        return [e.id for e in cat if str(e.metadata.get("source_path") or "") == path]

    # where
    return [e.id for e in cat if _matches_simple(e.metadata, where or {})]


def delete_by_ids(ids: Sequence[str]) -> Tuple[int, int]:
    """
    Delete the given chunk IDs from BOTH vector store and BM25 store.

    Returns:
        (n_deleted_vectors, n_deleted_bm25)
    """
    if not ids:
        return (0, 0)

    # Vector store delete
    vec = ChromaVectorStore.from_config()
    n_vec = 0
    try:
        n_vec = vec.delete(ids=list(ids)) or 0  # delete should return count; tolerate None
    except Exception:
        # If wrapper doesn't return a count, we cannot estimate reliably
        n_vec = len(ids)

    # BM25 store delete
    bm = BM25Store.load_or_create(str(_BM25_DIR))
    n_bm25 = bm.delete_many(ids=list(ids))
    bm.save()

    return (n_vec, n_bm25)


def reingest_paths(paths: Sequence[str]) -> List[Dict[str, object]]:
    """
    Reingest the given file paths using metadata inferred from existing catalog entries.
    For each unique source_path, we consolidate metadata across its chunks:
    - course/unit/language/doc_type/author/semester/tags
    (Preference: first non-empty; tags = union)

    Returns a list of ingest summaries compatible with IngestResult.__dict__.
    """
    if not paths:
        return []

    cat = _read_bm25_catalog()
    by_path = _group_by_source(cat)

    results: List[Dict[str, object]] = []
    for raw in paths:
        p = Path(raw).expanduser().resolve()
        key = str(p)
        # consolidate metadata from existing chunks (if any)
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
            # tags are flattened to tag_* booleans — recover names by stripping prefix
            for mk, mv in md.items():
                if mk.startswith("tag_") and mv:
                    tags_set.add(mk[len("tag_") :])

        # Build DocumentMetadata (language/doc_type enums accept strings from normalize)
        doc_meta = DocumentMetadata(
            course=course or None,
            unit=unit or None,
            language=language or None,  # normalize_cli_metadata inside ingest handles auto/en/it
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
    """
    Return unique source_path values matching the filter.
    """
    entries = list_entries(where=where)
    return _collect_source_paths(entries)