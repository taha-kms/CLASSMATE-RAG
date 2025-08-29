# rag/pipeline/rag.py
"""
End-to-end RAG pipeline used by the CLI.

What lives here:
- ingest_file(): load → chunk (concurrently) → optional dedup → embed (with cache) → upsert into
  both Chroma (vectors) and BM25 (lexical).
- ask_question(): retrieve (hybrid) → apply neighbor expansion + doc-level diversity →
  build grounded prompt → run local LLM → optional translate-on-miss → strict citations.
- index_stats(): quick health snapshot (counts + on-disk usage).

Design notes
- Chroma is used via the thin HTTP client; the actual Chroma server runs in Docker.
- Embeddings use multilingual-e5-base; we wrap it with a small on-disk cache.
- Retrieval is hybrid (vector + BM25) with RRF fusion and MMR-style diversity via
  neighbor expansion on contiguous chunk IDs.
- Stable chunk IDs make re-ingest idempotent across runs and machines.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from concurrent.futures import ThreadPoolExecutor, as_completed

# ---- Local modules ----------------------------------------------------------

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
from rag.retrieval.expand import expand_with_neighbors
from rag.generation import (
    LlamaCppRunner,
    build_grounded_messages,
    build_general_messages,
    format_context_blocks,
)
from rag.generation.prompting import choose_answer_language
from rag.generation.post import enforce_citations


# =============================================================================
# Data models returned by the pipeline
# =============================================================================

@dataclass
class IngestResult:
    """Summary returned by ingest_file()."""
    path: str
    doc_type: str
    total_pages: int
    total_chunks: int
    upserted: int
    created_at: str


@dataclass
class AskResult:
    """Summary returned by ask_question()."""
    question: str
    answer: str
    language: str
    top_k: int
    sources: List[str]                    # provenance strings aligned with [n] blocks
    retrieved: List[Dict[str, object]]    # raw retrieved items (id, metadata, scores…)
    filters_applied: Dict[str, object]
    hybrid: bool


# =============================================================================
# Helpers — tags and metadata sanitization
# =============================================================================

def _slug_tag(t: str) -> str:
    """Lowercase + snake_case for tag names (conservative charset)."""
    import re
    s = (t or "").lower().strip()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def _parse_tags(obj) -> List[str]:
    """Accept comma-separated string or list/tuple; return clean list."""
    if not obj:
        return []
    if isinstance(obj, (list, tuple)):
        vals = [str(x) for x in obj]
    else:
        vals = str(obj).split(",")
    out: List[str] = []
    for v in vals:
        v = v.strip()
        if v:
            out.append(v)
    return out


def _expand_tag_flags(tags_field) -> Dict[str, bool]:
    """
    Convert tags to booleans: "oop,exam" -> {"tag_oop": True, "tag_exam": True}
    This keeps the vector DB metadata simple and filterable.
    """
    flags: Dict[str, bool] = {}
    for t in _parse_tags(tags_field):
        slug = _slug_tag(t)
        if slug:
            flags[f"tag_{slug}"] = True
    return flags


def _sanitize_metadata(meta: Dict[str, object]) -> Dict[str, object]:
    """
    Keep only types allowed by the thin Chroma client (str, int, float, bool).
    Drop None/empty. Also expand tags -> tag_* booleans. Keep page/chunk_id ints.
    """
    clean: Dict[str, object] = {}

    # Expand tag flags first, then drop original 'tags'
    if "tags" in meta:
        clean.update(_expand_tag_flags(meta.get("tags")))

    # Whitelist core fields (others are ignored or stringified)
    for k in (
        "course", "unit", "language", "doc_type", "author",
        "semester", "source_path", "created_at", "page", "chunk_id"
    ):
        v = meta.get(k)
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            if isinstance(v, str) and not v.strip():   # skip empty strings
                continue
            clean[k] = v
        else:
            # be forgiving: stringify unknown types (e.g., Path)
            s = str(v).strip()
            if s:
                clean[k] = s
    return clean


# =============================================================================
# Ingest
# =============================================================================

def _concurrent_chunk_pages(
    pages: Sequence[Tuple[int, str]],
    *,
    chunk_size: int,
    chunk_overlap: int,
    max_workers: int,
) -> List[Tuple[int, int, str]]:
    """
    Concurrent page-wise chunking using a bounded thread pool.
    Returns flattened (page, global_chunk_id, text) tuples. The chunk_id is
    reassigned globally, monotonically across all pages to keep deterministic
    stable IDs. :contentReference[oaicite:4]{index=4}
    """
    # 1) Chunk each page independently, in parallel
    results: Dict[int, List[str]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        fut2page = {
            ex.submit(
                chunk_text,
                text,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                page=page,
                starting_chunk_id=0,   # local id (discarded later)
            ): page
            for (page, text) in pages
        }
        for fut in as_completed(fut2page):
            page = fut2page[fut]
            try:
                chs = fut.result()
            except Exception:
                chs = []
            # only keep the chunk text; we’ll rebuild global ids
            results[page] = [c.text for c in chs if (c.text or "").strip()]

    # 2) Flatten in page order and assign a single global chunk_id sequence
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
    Load a doc, chunk it (fast, concurrent), optionally deduplicate near-duplicates,
    embed with cache, and upsert to both stores.

    Returns IngestResult with counts etc. :contentReference[oaicite:5]{index=5}
    """
    cfg = load_config()
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    # Detect doc type if not provided in metadata
    doc_type = (doc_meta.doc_type.value if doc_meta.doc_type else None) or infer_doc_type_from_path(p)

    # Load -> list[(page_number, page_text)]
    pages = load_document_by_type(p, doc_type, enable_ocr=bool(cfg.enable_ocr))
    total_pages = len(pages)

    # Concurrent chunking (defaults scale with CPU cores)
    max_workers_env = os.getenv("INGEST_THREADS")
    max_workers = int(max_workers_env) if (max_workers_env and max_workers_env.isdigit()) else max(2, (os.cpu_count() or 4) // 2)
    chunks = _concurrent_chunk_pages(
        pages,
        chunk_size=int(cfg.chunk_size),
        chunk_overlap=int(cfg.chunk_overlap),
        max_workers=max_workers,
    )

    # Optional near-duplicate filtering (Jaccard on shingles, tuned by env)
    dedup_on = str(os.getenv("DEDUP_CHUNKS", "")).strip().lower() in {"1", "true", "yes"}
    dedup_thr = float(os.getenv("DEDUP_THRESHOLD", "0.92"))
    if dedup_on and chunks:
        blocks = [t for (_pg, _cid, t) in chunks]
        kept_blocks = dedup_text_blocks(blocks, jaccard_threshold=dedup_thr)
        # Rebuild chunks keeping order and reassigning global ids
        chunks = []
        cid = 0
        # reuse sequential chunking deterministically (single-thread to preserve order)
        for page, _old_cid, text in _concurrent_chunk_pages(
            pages, chunk_size=int(cfg.chunk_size), chunk_overlap=int(cfg.chunk_overlap), max_workers=1
        ):
            if text in kept_blocks:
                chunks.append((page, cid, text))
                kept_blocks.remove(text)
                cid += 1

    created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Collect per-chunk arrays ready for upsert
    ids: List[str] = []
    texts: List[str] = []
    metas: List[Dict[str, object]] = []

    # Components (embedder + caches + stores)
    base_embedder = E5MultilingualEmbedder(model_name=cfg.embedding_model_name)
    embedder = CachingEmbedder(base_embedder, cache_dir=os.getenv("EMB_CACHE_DIR") or "./indexes/emb_cache")
    vec_store = ChromaVectorStore.from_config()
    bm25_store = BM25Store.load_or_create("./indexes/bm25")

    # Language policy: chunk-level detection when metadata requests "auto"
    base_lang = doc_meta.language.value if doc_meta.language else "auto"

    for (page, chunk_id, text) in chunks:
        if not text.strip():
            continue

        # Chunk-level language (keeps BM25 tokenization & prompt language consistent)
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

        # Stable ID ties (path, page, chunk) + (course, unit) so re-ingest overwrites deterministically
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

    # Embed (cached) and upsert to BOTH stores (vector + lexical)
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


# =============================================================================
# Retrieval ergonomics used by ask/preview
# =============================================================================

def _apply_expansion_and_diversity(
    results: List[Dict[str, object]],
) -> List[Dict[str, object]]:
    """
    Expand each top hit with its neighbor chunks (same doc, adjacent chunk_ids)
    and cap how many chunks per doc we keep to balance breadth vs depth.

    Tunables come from env or config:
      - ENABLE_NEIGHBOR_EXPANSION (default: true)
      - NEIGHBOR_RADIUS (default: 1)
      - DOC_DIVERSITY_CAP (default: 3) :contentReference[oaicite:6]{index=6}
    """
    cfg = load_config()
    enable_expand = str(os.getenv("ENABLE_NEIGHBOR_EXPANSION", "true")).strip().lower() in {"1", "true", "yes"}
    radius = int(os.getenv("NEIGHBOR_RADIUS", "1"))
    cap = int(os.getenv("DOC_DIVERSITY_CAP", "3"))

    # Config can override env
    radius = int(getattr(cfg, "neighbor_radius", radius))
    cap = int(getattr(cfg, "doc_diversity_cap", cap))
    if hasattr(cfg, "enable_neighbor_expansion"):
        enable_expand = bool(getattr(cfg, "enable_neighbor_expansion"))

    if enable_expand and radius > 0:
        return expand_with_neighbors(results, radius=radius, max_per_doc=cap)
    # Even if expansion is off, still enforce doc-level cap for diversity
    return expand_with_neighbors(results, radius=0, max_per_doc=cap)


# =============================================================================
# Ask
# =============================================================================

def _looks_unknown(ans: str, lang: str) -> bool:
    """Heuristic to detect explicit 'I don't know' answers in EN/IT."""
    a = (ans or "").strip().lower()
    if not a:
        return True
    if lang == "it":
        return ("non lo so" in a) or ("non so" in a)
    return ("i don't know" in a) or ("i dont know" in a)


def _needs_translation(answer: str, target_lang: str) -> bool:
    """
    Decide whether to run a quick translation pass while preserving [n] citations.
    We only translate between EN/IT; other languages default to EN. :contentReference[oaicite:7]{index=7}
    """
    if not answer.strip():
        return False
    det = detect_lang_tag(answer)
    return det in {"en", "it"} and det != target_lang


def _translate_text(text: str, target_lang: str, runner: Optional[LlamaCppRunner] = None) -> str:
    """
    Translate to `target_lang` via the same local LLM runner, explicitly asking it to
    preserve bracketed citations like [1], [2] exactly. :contentReference[oaicite:8]{index=8}
    """
    if not text.strip():
        return text
    if runner is None:
        runner = LlamaCppRunner()

    if target_lang == "it":
        sys = (
            "Sei un traduttore. Traduci fedelmente in italiano il seguente testo.\n"
            "Mantieni esattamente i riferimenti tra parentesi quadre come [1], [2]."
        )
        prompt = f"Testo da tradurre:\n{text}"
    else:
        sys = (
            "You are a translator. Translate the following text faithfully into English.\n"
            "Preserve bracketed citations like [1], [2] exactly."
        )
        prompt = f"Text to translate:\n{text}"

    msgs = [{"role": "system", "content": sys}, {"role": "user", "content": prompt}]
    out = runner.chat(msgs, temperature=0.0, top_p=1.0, repeat_penalty=1.0, max_tokens=2048)
    return out.strip() or text


def ask_question(
    *,
    question: str,
    filters: DocumentMetadata,
    top_k: int = 8,
    hybrid: bool = True,
) -> AskResult:
    """
    Full RAG query path:
      1) Build retriever (vector + BM25) with e5 embedder (cached).
      2) Retrieve with metadata filters; then expand neighbors + doc diversity.
      3) Choose answer language (forced by filters.language if en/it, otherwise config).
      4) Build grounded messages (context blocks with [n]) and run local LLM.
      5) Optional: translate-on-miss (if LLM answered in the wrong language).
      6) Optional: strict citations (enforce [n] usage and append sources list). :contentReference[oaicite:9]{index=9}
    """
    cfg = load_config()

    # Components
    vec_store = ChromaVectorStore.from_config()
    bm25_store = BM25Store.load_or_create("./indexes/bm25")
    base_embedder = E5MultilingualEmbedder(model_name=cfg.embedding_model_name)
    embedder = CachingEmbedder(base_embedder, cache_dir=os.getenv("EMB_CACHE_DIR") or "./indexes/emb_cache")

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

    # Retrieval with metadata filters (converted to dict)
    where = filters.to_dict()
    results = retriever.retrieve(
        question=question,
        filters=where,
        top_k=int(top_k),
        hybrid=bool(hybrid),
    )

    # Retrieval ergonomics: neighbor expansion + per-doc cap
    results = _apply_expansion_and_diversity(list(results))

    # Decide target language consistent with prompting
    forced_lang = None
    if filters.language and filters.language.value in ("en", "it"):
        forced_lang = filters.language.value
    target_lang = choose_answer_language(
        question=question,
        forced_language=forced_lang,
        default_language=str(cfg.default_language),
    )

    # Build the grounded prompt (with compact, numbered context blocks)
    messages = build_grounded_messages(
        question=question,
        contexts=results,
        forced_language=forced_lang,
        default_language=str(cfg.default_language),
        max_context_chars=3500,  # keep under llama.cpp context to avoid truncation
    )

    # We also pre-compute the provenance list aligned with [n] blocks
    _context_text, prov = format_context_blocks(results)

    # Run local LLM
    runner = LlamaCppRunner()
    answer = runner.chat(messages).strip()

    # If the model essentially said “I don’t know”, fall back to a short general answer (no citations).
    if _looks_unknown(answer, target_lang):
        gm = build_general_messages(question=question, language=target_lang)
        answer = runner.chat(gm).strip()

    # Translate-on-miss (if enabled in cfg/env) — but preserve [n]
    translate_flag = (
        bool(getattr(cfg, "translate_on_miss", False)) or
        str(os.getenv("TRANSLATE_ON_MISS", "")).strip().lower() in {"1", "true", "yes"}
    )
    if translate_flag and _needs_translation(answer, target_lang):
        answer = _translate_text(answer, target_lang, runner=runner)

    # Strict citation enforcement (post-process)
    strict_flag = (
        bool(getattr(cfg, "strict_citations", False)) or
        str(os.getenv("STRICT_CITATIONS", "")).strip().lower() in {"1", "true", "yes"}
    )
    if strict_flag:
        add_sources = str(os.getenv("APPEND_SOURCES_BLOCK", "")).strip().lower() in {"1", "true", "yes"}
        answer = enforce_citations(
            answer=answer,
            provenance=prov,
            add_sources_block=add_sources,
            sources_title="Sources" if target_lang == "en" else "Fonti",
        )

    return AskResult(
        question=question,
        answer=answer,
        language=target_lang,
        top_k=int(top_k),
        sources=prov,
        retrieved=results,
        filters_applied=where,
        hybrid=bool(hybrid),
    )


# =============================================================================
# Stats
# =============================================================================

def index_stats() -> Dict[str, object]:
    """
    Return index health snapshot (best-effort):
      - vector & BM25 counts (−1 when unavailable)
      - disk usage for chroma / bm25 / embedding cache directories
    """
    vec = ChromaVectorStore.from_config()
    bm = BM25Store.load_or_create("./indexes/bm25")

    try:
        vcount = int(vec.count())
    except Exception:
        vcount = -1

    try:
        bcount = int(bm.count())
    except Exception:
        bcount = -1

    def _du(path: Path) -> int:
        """Rough recursive disk usage in bytes."""
        if not path.exists():
            return 0
        if path.is_file():
            return path.stat().st_size
        total = 0
        for p in path.rglob("*"):
            try:
                if p.is_file():
                    total += p.stat().st_size
            except Exception:
                continue
        return total

    usage = {
        "chroma_bytes": _du(Path("./indexes/chroma")),
        "bm25_bytes": _du(Path("./indexes/bm25")),
        "emb_cache_bytes": _du(Path("./indexes/emb_cache")),
    }
    return {"vectors": vcount, "bm25": bcount, **usage}
