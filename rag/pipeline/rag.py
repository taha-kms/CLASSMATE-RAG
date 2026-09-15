# rag/pipeline/rag.py
"""
End-to-end RAG pipeline used by the CLI.

What lives here:
- ingest_file(): load → chunk (concurrently) → optional dedup → embed (with cache) → upsert into
  both Chroma (vectors) and BM25 (lexical).
- ask_question(): retrieve (hybrid) → apply neighbor expansion + doc-level diversity →
  build grounded prompt → run local LLM → optional translate-on-miss → strict citations.

Index health lives in rag.admin.inspect.index_stats, which is what
rag.pipeline re-exports.

Design notes
- Chroma is used via the thin HTTP client; the actual Chroma server runs in Docker.
- Embeddings use multilingual-e5-base; we wrap it with a small on-disk cache.
- Retrieval is hybrid (vector + BM25) with RRF fusion and MMR-style diversity via
  neighbor expansion on contiguous chunk IDs.
- Stable chunk IDs make re-ingest idempotent across runs and machines.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Sequence, Tuple

from rag.chunking import chunk_text

# ---- Local modules ----------------------------------------------------------
from rag.config import load_config
from rag.embeddings.cache import CachingEmbedder
from rag.generation import (
    build_general_messages,
    build_grounded_messages,
    format_context_blocks,
)
from rag.loaders import (
    infer_doc_type_from_path,
    load_document_by_type,
)
from rag.metadata import DocumentMetadata
from rag.retrieval import BM25Store, ChromaVectorStore
from rag.retrieval.expand import expand_with_neighbors
from rag.retrieval.fusion import HybridRetriever
from rag.utils import detect_lang_tag, stable_chunk_id
from rag.utils.dedup import dedup_text_blocks

if TYPE_CHECKING:  # pragma: no cover - annotation only
    pass
from rag.generation.post import enforce_citations

# Route names and decision types are pure Python. The classifier, the
# sticky loader and the prompt table are imported where they are used, so
# importing this module does not load sentence-transformers or llama_cpp.
from rag.routing.types import ROUTES, Route, RouteDecision

if TYPE_CHECKING:  # pragma: no cover - annotations only
    from rag.embeddings import E5MultilingualEmbedder
    from rag.routing import HybridRouter, StickyModelLoader, SubjectClassifier


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
    sources: List[str]  # provenance strings aligned with [n] blocks
    retrieved: List[Dict[str, object]]  # raw retrieved items (id, metadata, scores…)
    filters_applied: Dict[str, object]
    hybrid: bool
    # Populated when routing is enabled; None on the legacy single-model path.
    route: Optional[str] = None
    route_reason: Optional[str] = None


# =============================================================================
# Routing singletons
# =============================================================================
# The classifier embeds prototype phrases once; reusing the instance avoids
# repeating that work on every ingest/ask. The loader holds at most one
# llama.cpp model resident across calls.
_SUBJECT_CLASSIFIER: Optional[SubjectClassifier] = None
_HYBRID_ROUTER: Optional[HybridRouter] = None
_MODEL_LOADER: Optional[StickyModelLoader] = None


def _get_subject_classifier(embedder: Optional["E5MultilingualEmbedder"] = None) -> "SubjectClassifier":
    """Lazy singleton. Pass an embedder to share E5 with the ingest path."""
    global _SUBJECT_CLASSIFIER
    if _SUBJECT_CLASSIFIER is None:
        from rag.routing import SubjectClassifier

        _SUBJECT_CLASSIFIER = SubjectClassifier(embedder=embedder)
    return _SUBJECT_CLASSIFIER


def _get_hybrid_router() -> "HybridRouter":
    global _HYBRID_ROUTER
    if _HYBRID_ROUTER is None:
        from rag.routing import HybridRouter

        cfg = load_config()
        _HYBRID_ROUTER = HybridRouter(
            classifier=_get_subject_classifier(),
            query_margin=float(cfg.route_query_margin),
            metadata_threshold=float(cfg.route_metadata_threshold),
            translation_requires_intent=bool(cfg.route_translation_requires_intent),
        )
    return _HYBRID_ROUTER


def _get_model_loader() -> "StickyModelLoader":
    global _MODEL_LOADER
    if _MODEL_LOADER is None:
        from rag.routing import StickyModelLoader

        _MODEL_LOADER = StickyModelLoader()
    return _MODEL_LOADER


def _folder_subject_hint(p: Path) -> Optional[str]:
    """
    Map the parent folder name to a canonical route, when it matches a known
    alias (math/code/translation/default + a few synonyms). Returns None for
    unrecognized folder names — caller can then auto-classify.
    """
    name = (p.parent.name or "").strip().lower()
    if not name:
        return None
    aliases = {
        "math": "math",
        "mathematics": "math",
        "matematica": "math",
        "code": "code",
        "coding": "code",
        "programming": "code",
        "informatica": "code",
        "cs": "code",
        "translation": "translation",
        "translate": "translation",
        "traduzione": "translation",
        "lang": "translation",
        "default": "default",
        "general": "default",
        "other": "default",
    }
    return aliases.get(name)


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
        "course",
        "unit",
        "language",
        "doc_type",
        "author",
        "semester",
        "source_path",
        "created_at",
        "page",
        "chunk_id",
        "subject",
    ):
        v = meta.get(k)
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            if isinstance(v, str) and not v.strip():  # skip empty strings
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
    stable IDs.
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
                starting_chunk_id=0,  # local id (discarded later)
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

    Returns IngestResult with counts etc.
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

    # Concurrent chunking (INGEST_THREADS, or half the CPU count)
    max_workers = cfg.resolved_ingest_threads()
    chunks = _concurrent_chunk_pages(
        pages,
        chunk_size=int(cfg.chunk_size),
        chunk_overlap=int(cfg.chunk_overlap),
        max_workers=max_workers,
    )

    # Optional near-duplicate filtering (Jaccard on shingles, tuned by env)
    dedup_on = bool(cfg.dedup_chunks)
    dedup_thr = float(cfg.dedup_threshold)
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
    from rag.embeddings import E5MultilingualEmbedder

    base_embedder = E5MultilingualEmbedder(model_name=cfg.embedding_model_name)
    embedder = CachingEmbedder(base_embedder)
    vec_store = ChromaVectorStore.from_config()
    bm25_store = BM25Store.load_or_create()

    # ---- Subject resolution (routing) -------------------------------------
    # Priority: explicit doc_meta.subject > folder hint > auto-classify (only
    # when routing is enabled). When routing is disabled we still accept an
    # explicit subject so future re-ingests aren't lossy, but we don't pay
    # for auto-classification.
    resolved_subject: Optional[str] = None
    if doc_meta.subject:
        resolved_subject = doc_meta.subject
    else:
        hint = _folder_subject_hint(p)
        if hint:
            resolved_subject = hint
        elif cfg.enable_routing:
            classifier = _get_subject_classifier(embedder=base_embedder)
            sample_texts = [t for (_pg, _cid, t) in chunks]
            cls = classifier.classify_chunks(sample_texts)
            resolved_subject = cls.subject

    # Language policy: chunk-level detection when metadata requests "auto"
    base_lang = doc_meta.language.value if doc_meta.language else "auto"

    for page, chunk_id, text in chunks:
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
            "tags": doc_meta.tags,  # will be expanded to tag_* booleans
            "source_path": str(p),
            "page": int(page),
            "chunk_id": int(chunk_id),
            "created_at": created_at,
            "subject": resolved_subject,
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

    Tunables: ENABLE_NEIGHBOR_EXPANSION, NEIGHBOR_RADIUS, DOC_DIVERSITY_CAP,
    all read through Config.
    """
    cfg = load_config()
    enable_expand = bool(cfg.enable_neighbor_expansion)
    radius = int(cfg.neighbor_radius)
    cap = int(cfg.doc_diversity_cap)

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
    We only translate between EN/IT; other languages default to EN.
    """
    if not answer.strip():
        return False
    det = detect_lang_tag(answer)
    return det in {"en", "it"} and det != target_lang


def _translate_text(text: str, target_lang: str, *, chat: Callable[[List[Dict[str, str]]], str]) -> str:
    """
    Translate to `target_lang`, asking the model to preserve bracketed
    citations like [1], [2] exactly.

    `chat` takes the message list and returns the reply. The caller supplies
    it so the routed path can translate through its single resident model.
    Building a runner here would load a second model alongside the one the
    sticky loader already holds, which is the thing that loader exists to
    prevent.
    """
    if not text.strip():
        return text

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
    return chat(msgs).strip() or text


def ask_question(
    *,
    question: str,
    filters: DocumentMetadata,
    top_k: int = 8,
    hybrid: bool = True,
    forced_subject: Optional[str] = None,
) -> AskResult:
    """
    Full RAG query path:
      1) Build retriever (vector + BM25) with e5 embedder (cached).
      2) Retrieve with metadata filters; then expand neighbors + doc diversity.
      3) Choose answer language (forced by filters.language if en/it, otherwise config).
      4) Build grounded messages (context blocks with [n]) and run local LLM.
      5) Optional: translate-on-miss (if LLM answered in the wrong language).
      6) Optional: strict citations (enforce [n] usage and append sources list).
    """
    cfg = load_config()

    # Components
    vec_store = ChromaVectorStore.from_config()
    bm25_store = BM25Store.load_or_create()
    from rag.embeddings import E5MultilingualEmbedder

    base_embedder = E5MultilingualEmbedder(model_name=cfg.embedding_model_name)
    embedder = CachingEmbedder(base_embedder)

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

    if forced_lang in ("en", "it"):
        target_lang = forced_lang
    else:
        _dl = str(cfg.default_language)
        target_lang = _dl if _dl in ("en", "it") else detect_lang_tag(question)

    # We also pre-compute the provenance list aligned with [n] blocks
    _context_text, prov = format_context_blocks(results, max_total_chars=3500)

    # ---- Routed path -----------------------------------------------------
    # When routing is enabled, the hybrid router picks a route from the
    # query + retrieved-chunk metadata, the sticky loader serves the
    # matching GGUF, and we use route-specific system prompts. The legacy
    # single-runner path below is kept untouched for cfg.enable_routing=False.
    if cfg.enable_routing:
        forced_route: Optional[Route] = None
        candidate = forced_subject or (filters.subject if hasattr(filters, "subject") else None)
        if isinstance(candidate, str) and candidate in ROUTES:
            forced_route = candidate  # type: ignore[assignment]

        retrieved_metas = [r.get("metadata") or {} for r in results]
        decision: RouteDecision = _get_hybrid_router().decide(
            question,
            retrieved_metas=retrieved_metas,
            forced_subject=forced_route,
        )

        # Build route-aware messages: route-specific system prompt + the
        # standard numbered-context user message.
        from rag.routing import system_prompt_for

        sys_prompt = system_prompt_for(decision.route, language=target_lang)
        user_msg = f"Context:\n{_context_text}\n\nQuestion:\n{question}\n\nAnswer:"
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_msg},
        ]

        loader = _get_model_loader()
        answer = loader.chat(
            route=decision.route,
            messages=messages,
            max_tokens=int(cfg.route_max_tokens),
            temperature=float(cfg.route_temperature),
            top_p=float(cfg.route_top_p),
        ).strip()

        # Fall back to a general (non-routed) answer if the model bailed out
        # with "I don't know". Use a context-free system prompt — the route
        # prompt forbids answering without context, so reusing it would just
        # produce another "I don't know".
        from_fallback = False
        if _looks_unknown(answer, target_lang):
            general_msgs = build_general_messages(question)
            if target_lang == "it":
                general_msgs[0] = {
                    "role": "system",
                    "content": "Sei un assistente generico. Rispondi alla domanda dell'utente.",
                }
            answer = loader.chat(
                route=decision.route,
                messages=general_msgs,
                max_tokens=int(cfg.route_max_tokens),
                temperature=float(cfg.route_temperature),
                top_p=float(cfg.route_top_p),
            ).strip()
            from_fallback = True

        # Translate-on-miss, through the resident model rather than a second one.
        if bool(cfg.translate_on_miss) and _needs_translation(answer, target_lang):
            answer = _translate_text(
                answer,
                target_lang,
                chat=lambda msgs: loader.chat(
                    route=decision.route,
                    messages=msgs,
                    max_tokens=int(cfg.route_max_tokens),
                    temperature=0.0,
                    top_p=1.0,
                ),
            )

        strict_flag = bool(cfg.strict_citations)
        # Skip citation enforcement when the answer came from the no-context
        # fallback: the model never saw `prov`, so attaching it would be a lie.
        if strict_flag and not from_fallback:
            add_sources = bool(cfg.append_sources_block)
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
            sources=[] if from_fallback else prov,
            retrieved=results,
            filters_applied=where,
            hybrid=bool(hybrid),
            route=decision.route,
            route_reason=decision.reason,
        )

    # ---- Legacy single-model path (routing disabled) ---------------------
    # Build the grounded prompt (with compact, numbered context blocks)
    messages = build_grounded_messages(
        question=question,
        context_text=_context_text,
        citations_required=True,
    )

    # Run local LLM. Imported here rather than at module scope so the
    # pipeline can be imported (and tested) without a compiled llama.cpp.
    from rag.generation import LlamaCppRunner

    runner = LlamaCppRunner()
    answer = runner.chat(messages).strip()

    # If the model essentially said “I don’t know”, fall back to a short general answer (no citations).
    from_fallback = False
    if _looks_unknown(answer, target_lang):
        gm = build_general_messages(question)
        answer = runner.chat(gm).strip()
        from_fallback = True

    # Translate-on-miss (if enabled in cfg/env) — but preserve [n]
    translate_flag = bool(cfg.translate_on_miss)
    if translate_flag and _needs_translation(answer, target_lang):
        answer = _translate_text(
            answer,
            target_lang,
            chat=lambda msgs: runner.chat(msgs, temperature=0.0, top_p=1.0, repeat_penalty=1.0, max_tokens=2048),
        )

    # Strict citation enforcement (post-process). Skip when the answer came
    # from the no-context fallback: `prov` describes context the model never saw.
    strict_flag = bool(cfg.strict_citations)
    if strict_flag and not from_fallback:
        add_sources = bool(cfg.append_sources_block)
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
        sources=[] if from_fallback else prov,
        retrieved=results,
        filters_applied=where,
        hybrid=bool(hybrid),
    )
