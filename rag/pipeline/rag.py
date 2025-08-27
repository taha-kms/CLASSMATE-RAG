"""
End-to-end RAG pipeline used by the CLI.

- ingest_file(): loads a document, chunks it (concurrently), optionally deduplicates,
  embeds with e5 (with optional disk cache), and upserts into Chroma (vector) and BM25 (lexical) stores.
- ask_question(): runs filtered hybrid retrieval, builds grounded messages,
  and generates an answer with Llama 3.1 (llama.cpp), returning answer + provenance.

Step 18: strict citation post-processing
Step 19: embedding cache, concurrent chunking, and near-duplicate filtering
Step 20: multilingual robustness (translate-on-miss preserving [n])
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from rag.config import load_config
from rag.metadata import DocumentMetadata
from rag.loaders import (
    infer_doc_type_from_path,
    load_document_by_type,
)
from rag.chunking import chunk_text
from rag.utils import detect_lang_tag, stable_chunk_id
from rag.utils.dedup import dedup_text_blocks
from rag.embeddings import E5MultilingualEmbedder
from rag.embeddings.cache import CachingEmbedder
from rag.retrieval import ChromaVectorStore, BM25Store
from rag.retrieval.fusion import HybridRetriever
from rag.generation import (
    LlamaCppRunner,
    build_grounded_messages,
    build_general_messages, 
    format_context_blocks,
)
from rag.generation.post import enforce_citations

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

def _concurrent_chunk_pages(
    pages: Sequence[Tuple[int, str]],
    *,
    chunk_size: int,
    chunk_overlap: int,
    max_workers: int,
) -> List[Tuple[int, int, str]]:
    """
    Concurrent page-wise chunking using a bounded thread pool.
    Returns a flattened list of (page, chunk_id, text) with global monotonic chunk_id to preserve determinism.
    """
    # First, chunk each page independently
    results: Dict[int, List[str]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        fut2page = {
            ex.submit(chunk_text, text, chunk_size=chunk_size, chunk_overlap=chunk_overlap, page=page, starting_chunk_id=0): page
            for (page, text) in pages
        }
        for fut in as_completed(fut2page):
            page = fut2page[fut]
            try:
                chs = fut.result()
            except Exception:
                chs = []
            # only store text blocks; we will reassign global chunk ids later
            results[page] = [c.text for c in chs if (c.text or "").strip()]

    # Flatten in page order and assign global chunk_id (stable)
    out: List[Tuple[int, int, str]] = []
    next_cid = 0
    for page, blocks in sorted(results.items(), key=lambda kv: kv[0]):
        for text in blocks:
            out.append((page, next_cid, text))
            next_cid += 1
    return out


def ingest_file(
    *,
    path: str | Path,
    doc_meta: DocumentMetadata,
) -> IngestResult:
    """
    Load a document, chunk it (concurrently), optionally deduplicate, embed (with cache),
    and upsert into vector + BM25 stores.
    """
    cfg = load_config()
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    doc_type = (doc_meta.doc_type.value if doc_meta.doc_type else None) or infer_doc_type_from_path(p)

    # Load into (page, text) pairs
    pages = load_document_by_type(p, doc_type, enable_ocr=bool(cfg.enable_ocr))
    total_pages = len(pages)

    # Concurrent chunking
    max_workers_env = os.getenv("INGEST_THREADS")
    max_workers = int(max_workers_env) if (max_workers_env and max_workers_env.isdigit()) else max(2, (os.cpu_count() or 4) // 2)
    chunks = _concurrent_chunk_pages(
        pages,
        chunk_size=int(cfg.chunk_size),
        chunk_overlap=int(cfg.chunk_overlap),
        max_workers=max_workers,
    )

    # Optional near-duplicate filtering (env toggle)
    dedup_on = str(os.getenv("DEDUP_CHUNKS", "")).strip().lower() in {"1", "true", "yes"}
    dedup_thr = float(os.getenv("DEDUP_THRESHOLD", "0.92"))
    if dedup_on and chunks:
        blocks = [t for (_pg, _cid, t) in chunks]
        kept_blocks = dedup_text_blocks(blocks, jaccard_threshold=dedup_thr)
        # Rebuild chunks with only kept blocks, reassigning chunk_id sequentially
        chunks = []
        cid = 0
        for page, _old_cid, text in _concurrent_chunk_pages(pages, chunk_size=int(cfg.chunk_size), chunk_overlap=int(cfg.chunk_overlap), max_workers=1):
            if text in kept_blocks:
                chunks.append((page, cid, text))
                kept_blocks.remove(text)
                cid += 1

    created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Prepare containers
    ids: List[str] = []
    texts: List[str] = []
    metas: List[Dict[str, object]] = []

    # Components
    base_embedder = E5MultilingualEmbedder(model_name=cfg.embedding_model_name)
    embedder = CachingEmbedder(base_embedder, cache_dir=os.getenv("EMB_CACHE_DIR") or "./indexes/emb_cache")
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

    # Embed (cached) & upsert
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



def _looks_unknown(ans: str, lang: str) -> bool:
    a = (ans or "").strip().lower()
    if not a:
        return True
    if lang == "it":
        return ("non lo so" in a) or ("non so" in a)
    return ("i don't know" in a) or ("i dont know" in a)


def _needs_translation(answer: str, target_lang: str) -> bool:
    """
    Heuristic: translate when the detected language is different from target_lang.
    """
    if not answer.strip():
        return False
    det = detect_lang_tag(answer)
    return det in {"en", "it"} and det != target_lang


def _translate_text(text: str, target_lang: str, runner: Optional[LlamaCppRunner] = None) -> str:
    """
    Translate text to target_lang ('en' or 'it') using the local Llama runner.
    Preserve bracketed citations like [1], [2]; do not add or remove citations.
    """
    if not text.strip():
        return text
    if runner is None:
        runner = LlamaCppRunner()

    if target_lang == "it":
        sys = (
            "Sei un traduttore. Traduci fedelmente in italiano il seguente testo.\n"
            "Mantieni *esattamente* i riferimenti tra parentesi quadre come [1], [2], ecc.\n"
            "Non aggiungere né rimuovere citazioni o contenuti; restituisci solo il testo tradotto."
        )
        prompt = f"Testo da tradurre:\n{text}"
    else:
        sys = (
            "You are a translator. Translate the following text faithfully into English.\n"
            "Preserve bracketed citations like [1], [2], etc. exactly.\n"
            "Do not add or remove citations or content; return only the translated text."
        )
        prompt = f"Text to translate:\n{text}"

    msgs = [{"role": "system", "content": sys}, {"role": "user", "content": prompt}]
    out = runner.chat(msgs, temperature=0.0, top_p=1.0, repeat_penalty=1.0, max_tokens=2048)
    return out.strip() or text


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
    base_embedder = E5MultilingualEmbedder(model_name=cfg.embedding_model_name)
    embedder = CachingEmbedder(base_embedder, cache_dir=os.getenv("EMB_CACHE_DIR") or "./indexes/emb_cache")

    retriever = HybridRetriever(
        vector_store=vec_store,
        bm25_store=bm25_store,
        embedder=embedder,   # cached embeddings for queries too
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

    # --- Step 20: optional translate-on-miss (wrong language) ---
    translate_flag = (
        bool(getattr(cfg, "translate_on_miss", False)) or
        str(os.getenv("TRANSLATE_ON_MISS", "")).strip().lower() in {"1", "true", "yes"}
    )
    if translate_flag and _needs_translation(answer, lang):
        answer = _translate_text(answer, lang, runner=runner)

    # --- Step 18: citation integrity (STRICT_CITATIONS) ---
    strict_flag = (
        bool(getattr(cfg, "strict_citations", False)) or
        str(os.getenv("STRICT_CITATIONS", "")).strip().lower() in {"1", "true", "yes"}
    )
    if strict_flag:
        # Optionally append a "Sources" block if configured via env
        add_sources = str(os.getenv("APPEND_SOURCES_BLOCK", "")).strip().lower() in {"1", "true", "yes"}
        answer = enforce_citations(
            answer=answer,
            provenance=prov,
            add_sources_block=add_sources,
            sources_title="Sources" if lang == "en" else "Fonti",
        )

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
