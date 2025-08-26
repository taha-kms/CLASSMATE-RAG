"""
Chroma vector store wrapper for CLASSMATE-RAG.

Dual-mode client:
- If env CHROMA_HTTP_URL is set -> use HttpClient (thin client; no default EF; no onnx)
- Else -> use PersistentClient (full library; we still set embedding_function=None)

We always supply embeddings explicitly (from E5).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, List

import numpy as np

from rag.config import load_config


def _slug_tag(t: str) -> str:
    import re
    s = (t or "").lower().strip()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def _parse_tags(obj) -> List[str]:
    if not obj:
        return []
    if isinstance(obj, (list, tuple)):
        vals = [str(x) for x in obj]
    else:
        vals = str(obj).split(",")
    out = []
    for v in vals:
        v = v.strip()
        if v:
            out.append(v)
    return out


def build_where_filter(meta_like: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Build a Chroma 'where' dict from simple CLI-style filters.
    - Equality on simple fields.
    - Tags become boolean flags: tag_<slug>: True
    - Ignore placeholder doc_type="other".
    """
    if not meta_like:
        return None

    clauses: List[Dict[str, Any]] = []

    for f in ["course", "unit", "language", "doc_type", "author", "semester"]:
        v = meta_like.get(f)
        if v is None:
            continue
        if isinstance(v, str):
            v = v.strip()
            if not v:
                continue
            if f == "doc_type" and v.lower() == "other":
                continue
        clauses.append({f: v})

    for t in _parse_tags(meta_like.get("tags")):
        slug = _slug_tag(t)
        if slug:
            clauses.append({f"tag_{slug}": True})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


@dataclass
class ChromaVectorStore:
    persist_dir: Path
    collection_name: str = "classmate_rag"
    distance: str = "cosine"

    _client: Optional[Any] = None
    _collection: Optional[Any] = None
    _mode_http: bool = False

    def _import_chromadb(self):
        import importlib
        return importlib.import_module("chromadb")

    @staticmethod
    def _normalize_host(host: str) -> str:
        h = (host or "").strip().lower()
        if h in ("127.0.0.1", "localhost", "::1", ""):
            return "localhost"
        return h

    def _ensure_client(self):
        if self._client is not None:
            return self._client

        chromadb = self._import_chromadb()
        http_url = os.getenv("CHROMA_HTTP_URL", "").strip()

        if http_url:
            # HTTP thin client
            self._mode_http = True
            host, port = "localhost", 8000
            try:
                from urllib.parse import urlparse
                parsed = urlparse(http_url)
                if parsed.hostname:
                    host = parsed.hostname
                if parsed.port:
                    port = parsed.port
            except Exception:
                pass
            host = self._normalize_host(host)
            from chromadb.config import Settings

            def make_http_client(impl: str):
                settings = Settings(
                    chroma_api_impl=impl,
                    chroma_server_host=host,
                    chroma_server_http_port=port,
                    anonymized_telemetry=False,
                )
                try:
                    return chromadb.HttpClient(host=host, port=port, settings=settings)
                except TypeError:
                    return chromadb.HttpClient(settings=settings)

            try:
                self._client = make_http_client("chromadb.api.fastapi.FastAPI")
            except Exception:
                self._client = make_http_client("rest")
            return self._client

        # Local persistent client
        self._mode_http = False
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self.persist_dir))
        return self._client

    def _ensure_collection(self):
        if self._collection is not None:
            return self._collection
        client = self._ensure_client()
        try:
            self._collection = client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": self.distance},
                embedding_function=None,
            )
        except TypeError:
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

        # Delete existing IDs
        for i in range(0, len(ids), batch_size):
            batch_ids = list(ids[i:i + batch_size])
            try:
                col.delete(ids=batch_ids)
            except Exception:
                pass

        # Add batches
        for i in range(0, len(ids), batch_size):
            batch_ids = list(ids[i:i + batch_size])
            batch_docs = list(documents[i:i + batch_size])
            batch_meta = list(metadatas[i:i + batch_size])
            batch_emb = embeddings[i:i + batch_size]
            col.add(
                ids=batch_ids,
                documents=batch_docs,
                metadatas=batch_meta,
                embeddings=batch_emb.astype("float32").tolist(),
            )

    # ---- Query ----

    def query(
        self,
        *,
        query_embeddings: np.ndarray,
        where: Optional[Dict[str, Any]] = None,  # already Chroma-style
        top_k: int = 8,
        include_documents: bool = True,
        include_embeddings: bool = False,
    ) -> List[Dict[str, Any]]:
        q = query_embeddings.astype("float32")
        if q.ndim == 1:
            q = q[None, :]

        col = self._ensure_collection()

        include = ["metadatas", "distances"]
        if include_documents:
            include.append("documents")
        if include_embeddings:
            include.append("embeddings")

        kwargs = dict(
            query_embeddings=q.tolist(),
            n_results=top_k,
            include=include,
        )
        # Only include 'where' when we actually have one
        if where:
            kwargs["where"] = where

        res = col.query(**kwargs)

        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0] if include_documents else [None] * len(ids)
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        embs = (res.get("embeddings") or [[]])[0] if include_embeddings else [None] * len(ids)

        out: List[Dict[str, Any]] = []
        for i in range(len(ids)):
            item = {
                "id": ids[i],
                "document": docs[i] if i < len(docs) else None,
                "metadata": metas[i] if i < len(metas) else {},
                "distance": dists[i] if i < len(dists) else None,
            }
            if include_embeddings and i < len(embs) and embs[i] is not None:
                item["embedding"] = np.array(embs[i], dtype="float32")
            out.append(item)
        return out

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
