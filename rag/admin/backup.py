"""
Backup, export, and migration utilities for CLASSMATE-RAG.

Exports a JSONL dump with one object per chunk:
{
  "id": "<stable chunk id>",
  "text": "<plain text>",
  "metadata": {...},           # sanitized, CLI-safe metadata
  "text_sha1": "<sha1 of utf8(text)>",
  "embedding_model": "intfloat/multilingual-e5-base",
  "embedding_sha1": "<sha1 of float32 embedding bytes>"   # optional
}

Operations:
- dump_index(path, include_embedding_checksum=True, batch_size=256)
- restore_dump(path, batch_size=256)
- vacuum_indexes()
- rebuild_embeddings(new_model_name, batch_size=256)
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np

from rag.config import load_config
from rag.embeddings import E5MultilingualEmbedder
from rag.embeddings.cache import CachingEmbedder
from rag.retrieval import ChromaVectorStore, BM25Store


# --- Helpers -----------------------------------------------------------------

def _sha1_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()

def _sha1_text(s: str) -> str:
    return _sha1_bytes((s or "").encode("utf-8", "ignore"))

def _iter_bm25_catalog(path: Path = Path("./indexes/bm25/bm25_index.jsonl")) -> Iterator[Tuple[str, str, Dict[str, object]]]:
    """
    Yield (id, text, metadata) from the BM25 catalog JSONL.
    This is the most complete, authoritative view of the corpus content.
    """
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = (line or "").strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            cid = str(obj.get("id") or "")
            text = str(obj.get("text") or "")
            meta = obj.get("metadata") or {}
            if cid and text:
                yield cid, text, dict(meta)


def _batched(items: List, n: int) -> Iterator[List]:
    if n <= 0:
        n = 256
    for i in range(0, len(items), n):
        yield items[i : i + n]


# --- Public API ---------------------------------------------------------------

def dump_index(
    out_path: str | Path,
    *,
    include_embedding_checksum: bool = True,
    batch_size: int = 256,
) -> int:
    """
    Export the corpus to JSONL at out_path.
    Returns number of records written.
    """
    cfg = load_config()
    model_name = str(cfg.embedding_model_name)
    # Build a lazy embedder for checksums only (no cache needed)
    base_embedder = E5MultilingualEmbedder(model_name=model_name)

    # Collect all entries from BM25 catalog
    entries: List[Tuple[str, str, Dict[str, object]]] = list(_iter_bm25_catalog())
    if not entries:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text("", encoding="utf-8")
        return 0

    # If including embedding checksum, we only compute and hash the vectors; we do NOT store vectors.
    with Path(out_path).open("w", encoding="utf-8") as w:
        total = 0
        if include_embedding_checksum:
            for batch in _batched(entries, batch_size):
                texts = [t for (_id, t, _m) in batch]
                vecs = base_embedder.encode_passages(texts)  # np.ndarray (B,D) float32
                for (cid, text, meta), vec in zip(batch, vecs):
                    obj = {
                        "id": cid,
                        "text": text,
                        "metadata": meta,
                        "text_sha1": _sha1_text(text),
                        "embedding_model": model_name,
                        "embedding_sha1": _sha1_bytes(np.asarray(vec, dtype="float32").tobytes()),
                    }
                    w.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    total += 1
        else:
            for (cid, text, meta) in entries:
                obj = {
                    "id": cid,
                    "text": text,
                    "metadata": meta,
                    "text_sha1": _sha1_text(text),
                    "embedding_model": model_name,
                    "embedding_sha1": None,
                }
                w.write(json.dumps(obj, ensure_ascii=False) + "\n")
                total += 1
    return total


def restore_dump(
    dump_path: str | Path,
    *,
    batch_size: int = 256,
) -> int:
    """
    Restore (bulk upsert) from a JSONL dump into Chroma + BM25.
    Returns number of restored records.
    """
    p = Path(dump_path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Dump not found: {p}")

    cfg = load_config()
    base_embedder = E5MultilingualEmbedder(model_name=str(cfg.embedding_model_name))
    embedder = CachingEmbedder(base_embedder, cache_dir="./indexes/emb_cache")
    vec_store = ChromaVectorStore.from_config()
    bm25_store = BM25Store.load_or_create("./indexes/bm25")

    # Read all lines (we want deterministic order)
    lines = [ln for ln in p.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip()]
    if not lines:
        return 0

    # Parse to tuples
    items: List[Tuple[str, str, Dict[str, object]]] = []
    for ln in lines:
        try:
            obj = json.loads(ln)
        except Exception:
            continue
        cid = str(obj.get("id") or "")
        text = str(obj.get("text") or "")
        meta = obj.get("metadata") or {}
        if cid and text:
            items.append((cid, text, dict(meta)))

    restored = 0
    for batch in _batched(items, batch_size):
        ids = [cid for (cid, _t, _m) in batch]
        texts = [t for (_cid, t, _m) in batch]
        metas = [m for (_cid, _t, m) in batch]

        # Embed
        emb = embedder.encode_passages(texts)

        # Upsert both stores
        vec_store.upsert(ids=ids, documents=texts, metadatas=metas, embeddings=emb)
        bm25_store.upsert_many(ids=ids, texts=texts, metadatas=metas)
        restored += len(batch)

    bm25_store.save()
    return restored


def vacuum_indexes() -> Dict[str, str]:
    """
    Housekeeping for indexes. Best-effort:
      - BM25: rewrite JSONL (save()).
      - Chroma: call 'compact' if wrapper supports; otherwise no-op.

    Returns status dict.
    """
    bm25_store = BM25Store.load_or_create("./indexes/bm25")
    bm25_store.save()

    vec_store = ChromaVectorStore.from_config()
    status = {"bm25": "saved"}
    try:
        # If wrapper provides it, compact/persist
        if hasattr(vec_store, "compact"):
            vec_store.compact()  # type: ignore[attr-defined]
            status["chroma"] = "compacted"
        elif hasattr(vec_store, "persist"):
            vec_store.persist()  # type: ignore[attr-defined]
            status["chroma"] = "persisted"
        else:
            status["chroma"] = "no-op"
    except Exception as e:
        status["chroma"] = f"error: {e}"

    return status


def rebuild_embeddings(
    new_model_name: str,
    *,
    batch_size: int = 256,
) -> Dict[str, object]:
    """
    Re-embed *all* corpus texts with `new_model_name` and refresh the vector store.

    Returns summary dict with counts and model info.
    """
    # Load all entries from BM25
    entries = list(_iter_bm25_catalog())
    if not entries:
        return {"updated": 0, "model": new_model_name}

    # Prepare stores and embedder
    vec_store = ChromaVectorStore.from_config()
    bm25_store = BM25Store.load_or_create("./indexes/bm25")  # ensures catalog exists/valid

    base_embedder = E5MultilingualEmbedder(model_name=new_model_name)
    embedder = CachingEmbedder(base_embedder, cache_dir="./indexes/emb_cache")

    updated = 0
    for batch in _batched(entries, batch_size):
        ids = [cid for (cid, _t, _m) in batch]
        texts = [t for (_cid, t, _m) in batch]
        metas = [m for (_cid, _t, m) in batch]

        # New embeddings
        emb = embedder.encode_passages(texts)
        # Upsert into vector store (replaces embeddings)
        vec_store.upsert(ids=ids, documents=texts, metadatas=metas, embeddings=emb)
        updated += len(batch)

    # Optionally re-save BM25 to refresh timestamps (no content change)
    bm25_store.save()

    return {"updated": updated, "model": new_model_name}
