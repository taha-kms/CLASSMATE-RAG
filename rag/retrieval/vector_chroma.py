"""
Chroma vector store wrapper for CLASSMATE-RAG.

Dual-mode client:
- If env CHROMA_HTTP_URL is set -> use HttpClient (thin client; no default EF; no onnx)
- Else -> use PersistentClient (full library; we still set embedding_function=None)

We always supply embeddings explicitly (from E5), so we NEVER want Chroma's default
embedding function. Passing embedding_function=None on collection creation is essential.

This module lazy-imports chromadb to avoid import-time side effects on envs
where the full library may try to init a default embedder.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, List

import numpy as np

from rag.config import load_config


def build_where_filter(meta_like: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
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
                    clauses.append({"tags": {"$contains": t}})
        elif isinstance(tags, str) and tags.strip():
            clauses.append({"tags": {"$contains": tags.strip()}})
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


@dataclass
class ChromaVectorStore:
    persist_dir: Path
    collection_name: str = "classmate_rag"
    distance: str = "cosine"  # cosine for L2-normalized e5 embeddings

    _client: Optional[Any] = None
    _collection: Optional[Any] = None
    _mode_http: bool = False

    def _import_chromadb(self):
        import importlib
        return importlib.import_module("chromadb")

    def _ensure_client(self):
        """
        Use HTTP client if CHROMA_HTTP_URL is set (recommended on Windows),
        else use local PersistentClient.
        """
        if self._client is not None:
            return self._client

        chromadb = self._import_chromadb()
        http_url = os.getenv("CHROMA_HTTP_URL", "").strip()

        if http_url:
            # -------- HTTP (thin) client mode --------
            self._mode_http = True

            # Parse host/port from URL
            host, port = "127.0.0.1", 8000
            try:
                from urllib.parse import urlparse
                parsed = urlparse(http_url)
                if parsed.hostname:
                    host = parsed.hostname
                if parsed.port:
                    port = parsed.port
            except Exception:
                pass

            # Build Settings with the impl string the client expects.
            # Some client builds want "chromadb.api.fastapi.FastAPI",
            # others accept "rest". We'll try FastAPI first, then fallback to "rest".
            from chromadb.config import Settings

            def make_http_client(impl: str):
                settings = Settings(
                    chroma_api_impl=impl,
                    chroma_server_host=host,
                    chroma_server_http_port=port,
                    anonymized_telemetry=False,
                )
                # Newer thin clients accept (settings=...) only; keep host/port for older ones
                try:
                    return chromadb.HttpClient(settings=settings)
                except TypeError:
                    return chromadb.HttpClient(host=host, port=port, settings=settings)

            try:
                self._client = make_http_client("chromadb.api.fastapi.FastAPI")
            except Exception:
                # Fallback for older docs/clients
                self._client = make_http_client("rest")

            return self._client

        # -------- Local persistent mode (full library) --------
        self._mode_http = False
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self.persist_dir))
        return self._client

    def _ensure_collection(self):
        if self._collection is not None:
            return self._collection
        client = self._ensure_client()
        # Always disable default EF. We supply embeddings explicitly.
        try:
            self._collection = client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": self.distance},
                embedding_function=None,
            )
        except TypeError:
            # Some thin-client versions don't accept embedding_function kwarg
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
        if len(ids) != len(documents) or len(ids) != len(metadatas) or len(ids) != len(embeddings):
            raise ValueError("Lengths of ids, documents, metadatas, and embeddings must match.")
        col = self._ensure_collection()

        # Delete existing IDs in batches
        for i in range(0, len(ids), batch_size):
            batch_ids = list(ids[i : i + batch_size])
            try:
                col.delete(ids=batch_ids)
            except Exception:
                pass

        # Add new records in batches
        for i in range(0, len(ids), batch_size):
            batch_ids = list(ids[i : i + batch_size])
            batch_docs = list(documents[i : i + batch_size])
            batch_meta = list(metadatas[i : i + batch_size])
            batch_emb = embeddings[i : i + batch_size]
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

    # ---- Maintenance ----

    def count(self) -> int:
        col = self._ensure_collection()
        try:
            return col.count()
        except Exception:
            return 0

    def reset_collection(self) -> None:
        client = self._ensure_client()
        try:
            client.delete_collection(self.collection_name)
        except Exception:
            pass
        self._collection = None
        self._ensure_collection()

    @classmethod
    def from_config(cls) -> "ChromaVectorStore":
        cfg = load_config()
        return cls(
            persist_dir=cfg.chroma_persist_directory,
            collection_name=cfg.chroma_collection_name,
            distance="cosine",
        )
