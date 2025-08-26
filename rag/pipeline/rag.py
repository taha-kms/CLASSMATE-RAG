"""
End-to-end RAG pipeline used by the CLI.

- ingest_file(): loads a document, chunks it, (optionally) detects language per chunk,
  embeds with e5, and upserts into Chroma (vector) and BM25 (lexical) stores.
- ask_question(): runs filtered hybrid retrieval, builds grounded messages,
  and generates an answer with Llama 3.1 (llama.cpp), returning answer + provenance.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from rag.config import load_config
from rag.metadata import DocumentMetadata
from rag.loaders import (
    infer_doc_type_from_path,
    load_document_by_type,
)
from rag.chunking import chunk_pages
from rag.utils import detect_lang_tag, stable_chunk_id
from rag.embeddings import E5MultilingualEmbedder
from rag.retrieval import ChromaVectorStore, BM25Store
from rag.retrieval.fusion import HybridRetriever
from rag.generation import (
    LlamaCppRunner,
    build_grounded_messages,
    build_general_messages, 
    format_context_blocks,
)

@dataclass
class IngestResult:
    path: str
    doc_type: str
    total_pages: int
    total_chunks: int
    upserted: int
    created_at: str


@dataclass
class AskResult:
    question: str
    answer: str
    language: str
    top_k: int
    sources: List[str]
    retrieved: List[Dict[str, object]]
    filters_applied: Dict[str, object]
    hybrid: bool


# -----------------------------
# Helpers (metadata sanitization)
# -----------------------------

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


def _expand_tag_flags(tags_field) -> Dict[str, bool]:
    flags: Dict[str, bool] = {}
    for t in _parse_tags(tags_field):
        slug = _slug_tag(t)
        if slug:
            flags[f"tag_{slug}"] = True
    return flags


def _sanitize_metadata(meta: Dict[str, object]) -> Dict[str, object]:
    """
    Keep only types allowed by the thin client: str, int, float, bool.
    Drop None/empty. Convert tags -> tag_* booleans. Keep page/chunk_id ints.
    """
    clean: Dict[str, object] = {}

    # Expand tag flags first, then drop original 'tags'
    if "tags" in meta:
        clean.update(_expand_tag_flags(meta.get("tags")))
    # Core fields
    for k in (
        "course", "unit", "language", "doc_type", "author",
        "semester", "source_path", "created_at", "page", "chunk_id"
    ):
        v = meta.get(k)
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            # skip empty strings
            if isinstance(v, str) and not v.strip():
                continue
            clean[k] = v
        else:
            # cast everything else to str (e.g., Paths)
            s = str(v).strip()
            if s:
                clean[k] = s

    return clean


# -----------------------------
# Ingest
# -----------------------------

def ingest_file(
    *,
    path: str | Path,
    doc_meta: DocumentMetadata,
) -> IngestResult:
    """
    Load a document, chunk it, embed, and upsert into vector + BM25 stores.
    """
    cfg = load_config()
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    doc_type = (doc_meta.doc_type.value if doc_meta.doc_type else None) or infer_doc_type_from_path(p)

    # Load into (page, text) pairs
    pages = load_document_by_type(p, doc_type, enable_ocr=bool(cfg.enable_ocr))
    total_pages = len(pages)

    # Chunk
    chunks = chunk_pages(
        pages,
        chunk_size=int(cfg.chunk_size),
        chunk_overlap=int(cfg.chunk_overlap),
        starting_chunk_id=0,
    )

    created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Prepare containers
    ids: List[str] = []
    texts: List[str] = []
    metas: List[Dict[str, object]] = []

    # Components
    embedder = E5MultilingualEmbedder(model_name=cfg.embedding_model_name)
    vec_store = ChromaVectorStore.from_config()
    bm25_store = BM25Store.load_or_create("./indexes/bm25")

    base_lang = doc_meta.language.value if doc_meta.language else "auto"

    for (page, chunk_id, text) in chunks:
        if not text.strip():
            continue

        # Per-chunk language: detect if auto
        lang = base_lang
        if lang == "auto" and cfg.enable_language_detection:
            lang = detect_lang_tag(text)

        raw_meta = {
            "course": doc_meta.course,
            "unit": doc_meta.unit,
            "language": lang,
            "doc_type": doc_type,
            "author": doc_meta.author,
            "semester": doc_meta.semester,
            "tags": doc_meta.tags,           # will be expanded to tag_* booleans
            "source_path": str(p),
            "page": int(page),
            "chunk_id": int(chunk_id),
            "created_at": created_at,
        }
        meta = _sanitize_metadata(raw_meta)

        cid = stable_chunk_id(
            source_path=p,
            page=int(page),
            chunk_index=int(chunk_id),
            course=doc_meta.course,
            unit=doc_meta.unit,
        )

        ids.append(cid)
        texts.append(text)
        metas.append(meta)

    total_chunks = len(texts)
    if total_chunks == 0:
        return IngestResult(
            path=str(p),
            doc_type=doc_type,
            total_pages=total_pages,
            total_chunks=0,
            upserted=0,
            created_at=created_at,
        )

    # Embed & upsert
    emb = embedder.encode_passages(texts)
    vec_store.upsert(ids=ids, documents=texts, metadatas=metas, embeddings=emb)

    bm25_store.upsert_many(ids=ids, texts=texts, metadatas=metas)
    bm25_store.save()

    return IngestResult(
        path=str(p),
        doc_type=doc_type,
        total_pages=total_pages,
        total_chunks=total_chunks,
        upserted=total_chunks,
        created_at=created_at,
    )



# ... AskResult dataclass unchanged ...

def _looks_unknown(ans: str, lang: str) -> bool:
    a = (ans or "").strip().lower()
    if not a:
        return True
    if lang == "it":
        return ("non lo so" in a) or ("non so" in a)
    return ("i don't know" in a) or ("i dont know" in a)




# -----------------------------
# Ask
# -----------------------------

def ask_question(
    *,
    question: str,
    filters: DocumentMetadata,
    top_k: int = 8,
    hybrid: bool = True,
) -> AskResult:
    cfg = load_config()

    vec_store = ChromaVectorStore.from_config()
    bm25_store = BM25Store.load_or_create("./indexes/bm25")
    embedder = E5MultilingualEmbedder(model_name=cfg.embedding_model_name)

    retriever = HybridRetriever(
        vector_store=vec_store,
        bm25_store=bm25_store,
        embedder=embedder,
        k_vector=int(cfg.k_vector),
        k_bm25=int(cfg.k_bm25),
        rrf_k=60,
        weight_vector=1.0,
        weight_bm25=1.0,
    )

    where = filters.to_dict()
    results = retriever.retrieve(
        question=question,
        filters=where,
        top_k=int(top_k),
        hybrid=bool(hybrid),
    )

    forced_lang = None
    if filters.language and filters.language.value in ("en", "it"):
        forced_lang = filters.language.value

    messages = build_grounded_messages(
        question=question,
        contexts=results,
        forced_language=forced_lang,
        default_language=str(cfg.default_language),
        max_context_chars=3500,
    )
    # language we asked for
    lang = "it" if (forced_lang == "it") else "en"

    context_text, prov = format_context_blocks(results)

    runner = LlamaCppRunner()
    answer = runner.chat(messages).strip()

    # Fallback: if model says it doesn't know (or empty), provide a short general definition without citations
    if _looks_unknown(answer, lang):
        fallback_msgs = build_general_messages(question=question, language=lang, style="concise")
        fallback = runner.chat(fallback_msgs).strip()
        if fallback:
            note = " (General knowledge — no in-corpus source)" if lang == "en" else " (Conoscenza generale — nessuna fonte nel corpus)"
            answer = fallback + note

    return AskResult(
        question=question,
        answer=answer,
        language=lang,
        top_k=int(top_k),
        sources=prov,
        retrieved=results,
        filters_applied=where,
        hybrid=bool(hybrid),
    )
