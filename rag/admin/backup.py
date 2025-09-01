"""
Backup and restore utilities for the RAG system.

This module allows you to:
- Export all indexed data to a JSONL file (with optional embedding checksums).
- Restore data from a JSONL dump back into BM25 and vector stores.
- Compact/clean existing indexes.
- Rebuild embeddings with a new model.

The BM25 JSONL catalog is treated as the source of truth for all chunks.
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import numpy as np

from rag.config import load_config
from rag.embeddings import E5MultilingualEmbedder
from rag.embeddings.cache import CachingEmbedder
from rag.retrieval import ChromaVectorStore, BM25Store


# ------------------------------
# Internal helper functions
# ------------------------------

def _sha1_bytes(b: bytes) -> str:
    """Return SHA1 hash of raw bytes as hex string."""
    return hashlib.sha1(b).hexdigest()

def _sha1_text(s: str) -> str:
    """Return SHA1 hash of a UTF-8 encoded string."""
    return _sha1_bytes((s or "").encode("utf-8", "ignore"))

def _iter_bm25_catalog(path: Path = Path("./indexes/bm25/bm25_index.jsonl")) -> Iterator[Tuple[str, str, Dict[str, object]]]:
    """
    Yield all entries from the BM25 catalog.
    Each line contains (id, text, metadata).
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
    """Split a list into batches of size n."""
    if n <= 0:
        n = 256
    for i in range(0, len(items), n):
        yield items[i : i + n]


# ------------------------------
# Public API
# ------------------------------

def dump_index(
    out_path: str | Path,
    *,
    include_embedding_checksum: bool = True,
    batch_size: int = 256,
) -> int:
    """
    Write all chunks from BM25 to a JSONL file.
    Optionally include embedding SHA1 hashes for integrity checks.
    Returns the number of written records.
    """
    cfg = load_config()
    model_name = str(cfg.embedding_model_name)
    embedder = E5MultilingualEmbedder(model_name=model_name)

    entries = list(_iter_bm25_catalog())
    if not entries:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text("", encoding="utf-8")
        return 0

    with Path(out_path).open("w", encoding="utf-8") as w:
        total = 0
        if include_embedding_checksum:
            for batch in _batched(entries, batch_size):
                texts = [t for (_id, t, _m) in batch]
                vecs = embedder.encode_passages(texts)
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
    Load chunks from a JSONL dump and insert them into BM25 and Chroma.
    Returns the number of restored records.
    """
    p = Path(dump_path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Dump not found: {p}")

    cfg = load_config()
    base_embedder = E5MultilingualEmbedder(model_name=str(cfg.embedding_model_name))
    embedder = CachingEmbedder(base_embedder, cache_dir="./indexes/emb_cache")
    vec_store = ChromaVectorStore.from_config()
    bm25_store = BM25Store.load_or_create("./indexes/bm25")

    lines = [ln for ln in p.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip()]
    if not lines:
        return 0

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

        emb = embedder.encode_passages(texts)

        vec_store.upsert(ids=ids, documents=texts, metadatas=metas, embeddings=emb)
        bm25_store.upsert_many(ids=ids, texts=texts, metadatas=metas)
        restored += len(batch)

    bm25_store.save()
    return restored


def vacuum_indexes() -> Dict[str, str]:
    """
    Compact and clean indexes.
    - BM25: rewrite JSONL file.
    - Chroma: compact or persist if supported.
    Returns a status dictionary.
    """
    bm25_store = BM25Store.load_or_create("./indexes/bm25")
    bm25_store.save()

    vec_store = ChromaVectorStore.from_config()
    status = {"bm25": "saved"}
    try:
        if hasattr(vec_store, "compact"):
            vec_store.compact()
            status["chroma"] = "compacted"
        elif hasattr(vec_store, "persist"):
            vec_store.persist()
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
    Recompute embeddings for all texts using a new model.
    Updates the vector store but keeps BM25 unchanged.
    Returns a summary with counts and model name.
    """
    entries = list(_iter_bm25_catalog())
    if not entries:
        return {"updated": 0, "model": new_model_name}

    vec_store = ChromaVectorStore.from_config()
    bm25_store = BM25Store.load_or_create("./indexes/bm25")

    base_embedder = E5MultilingualEmbedder(model_name=new_model_name)
    embedder = CachingEmbedder(base_embedder, cache_dir="./indexes/emb_cache")

    updated = 0
    for batch in _batched(entries, batch_size):
        ids = [cid for (cid, _t, _m) in batch]
        texts = [t for (_cid, t, _m) in batch]
        metas = [m for (_cid, _t, m) in batch]

        emb = embedder.encode_passages(texts)
        vec_store.upsert(ids=ids, documents=texts, metadatas=metas, embeddings=emb)
        updated += len(batch)

    bm25_store.save()
    return {"updated": updated, "model": new_model_name}
