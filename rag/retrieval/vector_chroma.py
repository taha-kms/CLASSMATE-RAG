"""
Chroma vector store wrapper for CLASSMATE-RAG.

Dual-mode client:
- If env CHROMA_HTTP_URL is set -> use HttpClient (thin client; no default EF; no onnx)
- Else -> use PersistentClient (full library; we still set embedding_function=None)

We always supply embeddings explicitly (from E5).
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rag.config import load_config

log = logging.getLogger(__name__)


class VectorStoreUnavailable(RuntimeError):
    """The vector store could not be reached.

    chromadb reports an unreachable server as a plain ValueError, which is
    indistinguishable from a bug in our own call. Callers that want to degrade
    gracefully when the database is down need to catch *that* and nothing else,
    so the boundary is translated here once.
    """


def _slug_tag(t: str) -> str:
    import re

    s = (t or "").lower().strip()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def _parse_tags(obj) -> list[str]:
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


def build_where_filter(meta_like: Mapping[str, Any]) -> dict[str, Any] | None:
    """
    Build a Chroma 'where' dict from simple CLI-style filters.
    - Equality on simple fields.
    - Tags become boolean flags: tag_<slug>: True
    - Ignore placeholder doc_type="other".
    """
    if not meta_like:
        return None

    clauses: list[dict[str, Any]] = []

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

    _client: Any | None = None
    _collection: Any | None = None
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
            from urllib.parse import urlparse

            try:
                parsed = urlparse(http_url)
            except ValueError:
                # A malformed CHROMA_HTTP_URL is worth saying out loud rather
                # than silently falling back to localhost:8000, which then
                # fails somewhere less obvious.
                raise VectorStoreUnavailable(f"CHROMA_HTTP_URL is not a usable URL: {http_url!r}") from None
            if parsed.hostname:
                host = parsed.hostname
            if parsed.port:
                port = parsed.port
            host = self._normalize_host(host)
            from chromadb.config import Settings

            try:
                self._client = chromadb.HttpClient(
                    host=host,
                    port=port,
                    settings=Settings(anonymized_telemetry=False),
                )
            except ValueError as e:
                # chromadb reports an unreachable server as a bare ValueError.
                # Translate it so callers can tell "database is down" from
                # "we called this wrong"; everything else propagates.
                raise VectorStoreUnavailable(f"Could not reach the Chroma server at {host}:{port} ({e})") from e
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

        # Replace any existing rows for these ids. A failure here would leave
        # the collection holding both the old and the new copy, so it is not
        # something to swallow.
        for i in range(0, len(ids), batch_size):
            col.delete(ids=list(ids[i : i + batch_size]))

        # Add batches
        for i in range(0, len(ids), batch_size):
            batch_ids = list(ids[i : i + batch_size])
            batch_docs = list(documents[i : i + batch_size])
            batch_meta = list(metadatas[i : i + batch_size])
            batch_emb = embeddings[i : i + batch_size]
            col.add(
                ids=batch_ids,
                documents=batch_docs,
                metadatas=batch_meta,
                embeddings=batch_emb.astype("float32").tolist(),
            )

    def delete(self, *, ids: Sequence[str], batch_size: int = 512) -> int:
        """
        Remove ids from the collection and report how many rows actually went.

        Measured by counting before and after rather than returning len(ids):
        ids that were not present remove nothing, and a caller that trusts the
        request length cannot tell the difference. Exceptions propagate, so a
        failed delete is a failure rather than a reassuring number.
        """
        if not ids:
            return 0

        col = self._ensure_collection()
        before = col.count()
        for i in range(0, len(ids), batch_size):
            col.delete(ids=list(ids[i : i + batch_size]))
        return max(0, before - col.count())

    def all_ids(self, *, batch_size: int = 1000) -> list[str]:
        """
        Every id in the collection.

        Needed to compare the two stores. Paged, because a corpus large
        enough to matter is large enough that fetching it whole is unkind.
        """
        col = self._ensure_collection()
        ids: list[str] = []
        offset = 0
        while True:
            got = col.get(limit=batch_size, offset=offset, include=[])
            batch = got.get("ids") or []
            if not batch:
                break
            ids.extend(batch)
            if len(batch) < batch_size:
                break
            offset += batch_size
        return ids

    # ---- Query ----

    def query(
        self,
        *,
        query_embeddings: np.ndarray,
        where: dict[str, Any] | None = None,  # already Chroma-style
        top_k: int = 8,
        include_documents: bool = True,
        include_embeddings: bool = False,
    ) -> list[dict[str, Any]]:
        q = query_embeddings.astype("float32")
        if q.ndim == 1:
            q = q[None, :]

        col = self._ensure_collection()

        include = ["metadatas", "distances"]
        if include_documents:
            include.append("documents")
        if include_embeddings:
            include.append("embeddings")

        kwargs = {
            "query_embeddings": q.tolist(),
            "n_results": top_k,
            "include": include,
        }
        # Only include 'where' when we actually have one
        if where:
            kwargs["where"] = where

        res = col.query(**kwargs)

        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0] if include_documents else [None] * len(ids)
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        embs = (res.get("embeddings") or [[]])[0] if include_embeddings else [None] * len(ids)

        out: list[dict[str, Any]] = []
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
        # Deliberately not guarded. A swallowed failure here returns 0, which
        # reads as "the corpus is empty" -- the most plausible wrong answer
        # available, and the shape of #100. Callers that want to degrade decide
        # that for themselves.
        return self._ensure_collection().count()

    def reset_collection(self) -> None:
        client = self._ensure_client()
        # Imported here, not at module scope: chromadb is loaded lazily so that
        # importing rag.retrieval does not drag it in (see test_lazy_imports).
        from chromadb.errors import NotFoundError

        try:
            client.delete_collection(self.collection_name)
        except NotFoundError:
            pass  # already absent, which is the state we wanted
        self._collection = None
        self._ensure_collection()

    @classmethod
    def from_config(cls) -> ChromaVectorStore:
        cfg = load_config()
        return cls(
            persist_dir=cfg.chroma_persist_directory,
            collection_name=cfg.chroma_collection_name,
            distance="cosine",
        )
