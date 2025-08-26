"""
Prompt construction for grounded Q&A (EN/IT) using Llama 3.1.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from rag.utils.lang_detect import detect_lang_tag


def choose_answer_language(
    *,
    question: str,
    forced_language: Optional[str] = None,
    default_language: str = "auto",
) -> str:
    if forced_language in {"en", "it"}:
        return forced_language
    if default_language in {"en", "it"}:
        return default_language
    lang = detect_lang_tag(question or "")
    return "it" if lang == "it" else "en"


def _format_provenance(meta: Dict) -> str:
    sp = meta.get("source_path") or meta.get("path") or "source"
    page = meta.get("page")
    cid = meta.get("chunk_id")
    if page is not None and cid is not None:
        return f"{sp}#p{page}:c{cid}"
    if page is not None:
        return f"{sp}#p{page}"
    return str(sp)


def format_context_blocks(
    results: Sequence[Dict[str, object]],
    *,
    max_chars_per_block: int = 800,
    max_total_chars: Optional[int] = 3500,
) -> Tuple[str, List[str]]:
    """
    Formats retrieval results into numbered blocks and enforces a total context budget.
    Returns: (context_text, provenance_list)
    """
    blocks: List[str] = []
    prov: List[str] = []
    used = 0
    for i, r in enumerate(results, start=1):
        doc = (r.get("document") or "")
        if max_chars_per_block:
            doc = doc[: max_chars_per_block]
        meta = r.get("metadata") or {}
        pv = _format_provenance(meta)
        blk = f"[{i}] {pv}\n{doc}".strip()
        blk_len = len(blk) + (2 if blocks else 0)  # account for joins
        if max_total_chars is not None and used + blk_len > max_total_chars:
            break
        blocks.append(blk)
        prov.append(pv)
        used += blk_len
    return ("\n\n".join(blocks)).strip(), prov


def build_grounded_messages(
    *,
    question: str,
    contexts: Sequence[Dict[str, object]],
    forced_language: Optional[str] = None,
    default_language: str = "auto",
    style: str = "concise",
    max_context_chars: Optional[int] = 3500,
) -> List[Dict[str, str]]:
    lang = choose_answer_language(
        question=question,
        forced_language=forced_language,
        default_language=default_language,
    )
    context_text, _prov = format_context_blocks(contexts, max_total_chars=max_context_chars)

    now = datetime.utcnow().strftime("%Y-%m-%d")
    system_en = (
        "You are CLASSMATE-RAG, a helpful study assistant. "
        "Answer strictly using the provided context. "
        "Cite sources inline using [n]. If the answer is not in the context, say you don't know."
    )
    system_it = (
        "Sei CLASSMATE-RAG, un assistente allo studio. "
        "Rispondi solo utilizzando il contesto fornito. "
        "Cita le fonti nel testo usando [n]. Se la risposta non è nel contesto, dillo chiaramente."
    )
    system_msg = system_it if lang == "it" else system_en

    user_intro_en = (
        f"Date: {now}\n"
        "Use only these context blocks to answer. Cite as [n].\n\n"
        f"{context_text}\n\n"
        f"Question: {question}\n"
        f"Answer in English. Keep it {style}."
    )
    user_intro_it = (
        f"Data: {now}\n"
        "Usa solo questi blocchi di contesto per rispondere. Cita come [n].\n\n"
        f"{context_text}\n\n"
        f"Domanda: {question}\n"
        f"Rispondi in italiano. Mantieni uno stile {style}."
    )
    user_msg = user_intro_it if lang == "it" else user_intro_en

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]
