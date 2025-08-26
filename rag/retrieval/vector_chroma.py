"""
Chroma vector store wrapper for CLASSMATE-RAG.

Features:
- Persistent client bound to CHROMA_PERSIST_DIRECTORY and collection name.
- Idempotent upsert: delete existing IDs then add embeddings/documents/metadata.
- Metadata-aware query with simple equality and tag inclusion filters.
- Cosine space by default (pairs well with L2-normalized e5 embeddings).

We keep embeddings external (already computed via the e5 embedder) and pass
them explicitly to add/query; this avoids hidden embedding behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import chromadb

from rag.config import load_config


# ---------------------------
# Filter builder
# ---------------------------

def build_where_filter(meta_like: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Convert a metadata-like dict (e.g., from DocumentMetadata.to_dict()) into a Chroma 'where' filter.
    - Only includes non-empty fields among: course, unit, language, doc_type, author, semester
    - For 'tags': if a list is provided, require each tag to be present (AND of $contains)
    Returns None if no filterable fields are set.
    """
    if not meta_like:
        return None

    simple_fields = ["course", "unit", "language", "doc_type", "author", "semester"]
    clauses: List[Dict[str, Any]] = []

    for f in simple_fields:
        v = meta_like.get(f)
        if v is None:
            continue
        if isinstance(v, str) and not v.strip():
            continue
        clauses.append({f: v})

    tags = meta_like.get("tags")
    if tags:
        if isinstance(tags, (list, tuple)):
            for t in tags:
                if isinstance(t, str) and t.strip():
                    # Chroma supports $contains on array fields
                    clauses.append({"tags": {"$contains": t}})
        elif isinstance(tags, str) and tags.strip():
            clauses.append({"tags": {"$contains": tags.strip()}})

    if not clauses:
        return None

    # If there's more than one clause, AND them together
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


# ---------------------------
# Vector store wrapper
# ---------------------------

@dataclass
class ChromaVectorStore:
    persist_dir: Path
    collection_name: str = "classmate_rag"
    distance: str = "cosine"  # cosine recommended for L2-normalized embeddings

    # internal
    _client: Optional[chromadb.PersistentClient] = None
    _collection: Optional[Any] = None

    def _ensure_client(self) -> chromadb.PersistentClient:
        if self._client is None:
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(self.persist_dir))
        return self._client

    def _ensure_collection(self):
        if self._collection is None:
            client = self._ensure_client()
            # We don't pass an embedding function; we supply embeddings explicitly
            self._collection = client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": self.distance},
            )
        return self._collection

    # ---- Upsert ----

    def upsert(
        self,
        *,
        ids: Sequence[str],
        documents: Sequence[str],
        metadatas: Sequence[Mapping[str, Any]],
        embeddings: np.ndarray,
        batch_size: int = 512,
    ) -> None:
        """
        Idempotent add: delete any existing records with these IDs, then add.
        Expects embeddings as (N, D) numpy array (float32/float64).
        """
        if len(ids) != len(documents) or len(ids) != len(metadatas) or len(ids) != len(embeddings):
            raise ValueError("Lengths of ids, documents, metadatas, and embeddings must match.")

        col = self._ensure_collection()

        # Delete existing IDs in batches
        for i in range(0, len(ids), batch_size):
            batch_ids = list(ids[i : i + batch_size])
            try:
                col.delete(ids=batch_ids)
            except Exception:
                # Ignore missing IDs
                pass

        # Add new records in batches
        for i in range(0, len(ids), batch_size):
            batch_ids = list(ids[i : i + batch_size])
            batch_docs = list(documents[i : i + batch_size])
            batch_meta = list(metadatas[i : i + batch_size])
            batch_emb = embeddings[i : i + batch_size]
            # Ensure serializable types
            emb_list = batch_emb.astype("float32").tolist()
            col.add(
                ids=batch_ids,
                documents=batch_docs,
                metadatas=batch_meta,
                embeddings=emb_list,
            )

    # ---- Query ----

    def query(
        self,
        *,
        query_embeddings: np.ndarray,
        where: Optional[Dict[str, Any]] = None,
        top_k: int = 8,
        include_documents: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Perform a vector similarity query.
        Returns a list of result dicts:
            {
              "id": str,
              "document": str,
              "metadata": dict,
              "distance": float
            }
        """
        if query_embeddings.ndim == 1:
            q = [query_embeddings.astype("float32").tolist()]
        else:
            q = query_embeddings.astype("float32").tolist()

        col = self._ensure_collection()
        include = ["ids", "metadatas", "distances"]
        if include_documents:
            include.append("documents")

        res = col.query(
            query_embeddings=q,
            n_results=top_k,
            where=where or {},
            include=include,
        )

        # Chroma returns lists per query; we only pass one query vector at a time here.
        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0] if include_documents else [None] * len(ids)
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]

        out: List[Dict[str, Any]] = []
        for i in range(len(ids)):
            out.append(
                {
                    "id": ids[i],
                    "document": docs[i],
                    "metadata": metas[i] if i < len(metas) else {},
                    "distance": dists[i] if i < len(dists) else None,
                }
            )
        return out

    # ---- Maintenance / stats ----

    def count(self) -> int:
        """Return number of items in the collection (approx)."""
        col = self._ensure_collection()
        try:
            return col.count()
        except Exception:
            return 0

    def reset_collection(self) -> None:
        """
        Drop and recreate the collection (dangerous). Use only for full reindex.
        """
        client = self._ensure_client()
        try:
            client.delete_collection(self.collection_name)
        except Exception:
            pass
        self._collection = None
        self._ensure_collection()

    # ---- Factory ----

    @classmethod
    def from_config(cls) -> "ChromaVectorStore":
        cfg = load_config()
        return cls(
            persist_dir=cfg.chroma_persist_directory,
            collection_name=cfg.chroma_collection_name,
            distance="cosine",
        )
