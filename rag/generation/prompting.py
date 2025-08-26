"""
Prompt construction for grounded Q&A (EN/IT) using Llama 3.1.

We build 'messages' suitable for llama.cpp's chat completion API:
  - system: role and instructional style
  - user: includes the question and a set of context snippets with citations

Utilities:
- choose_answer_language(): determine answer language from query or override ('en'/'it')
- format_context_blocks(): format retrieved chunks with human-friendly provenance
- build_grounded_messages(): assemble messages for chat completion
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
    """
    Decide the answer language:
      - if forced_language in {'en','it'} -> use it
      - else if default_language is 'en'/'it' -> use it
      - else detect from question -> 'en' or 'it' (fallback 'en')
    """
    if forced_language in {"en", "it"}:
        return forced_language
    if default_language in {"en", "it"}:
        return default_language
    # auto
    lang = detect_lang_tag(question or "")
    return "it" if lang == "it" else "en"


def _format_provenance(meta: Dict) -> str:
    """
    Render a concise provenance string like:
      path/to/file.pdf#p3:c12
    """
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
) -> Tuple[str, List[str]]:
    """
    Formats retrieval results into numbered blocks suitable for prompting.
    Returns:
      (context_text, provenance_list)
    where provenance_list aligns with block indices [1..N].
    """
    blocks: List[str] = []
    prov: List[str] = []
    for i, r in enumerate(results, start=1):
        doc = (r.get("document") or "")[: max_chars_per_block]
        meta = r.get("metadata") or {}
        pv = _format_provenance(meta)
        block = f"[{i}] {pv}\n{doc}".strip()
        blocks.append(block)
        prov.append(pv)
    return ("\n\n".join(blocks)).strip(), prov


def build_grounded_messages(
    *,
    question: str,
    contexts: Sequence[Dict[str, object]],
    forced_language: Optional[str] = None,
    default_language: str = "auto",
    style: str = "concise",
) -> List[Dict[str, str]]:
    """
    Build chat messages for grounded Q&A.
    - 'contexts' is a list of retrieval dicts (id, document, metadata, scores...)
    """
    lang = choose_answer_language(
        question=question,
        forced_language=forced_language,
        default_language=default_language,
    )
    context_text, _prov = format_context_blocks(contexts)

    now = datetime.utcnow().strftime("%Y-%m-%d")
    system_en = (
        "You are CLASSMATE-RAG, a helpful study assistant. "
        "Answer strictly using the provided context. "
        "Cite sources inline using [n] where n is the context block number. "
        "If the answer is not in the context, say you don't know."
    )
    system_it = (
        "Sei CLASSMATE-RAG, un assistente allo studio. "
        "Rispondi solo utilizzando il contesto fornito. "
        "Cita le fonti nel testo usando [n] dove n è il numero del blocco di contesto. "
        "Se la risposta non è nel contesto, dillo chiaramente."
    )
    system_msg = system_it if lang == "it" else system_en

    # User message includes the context and the question
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
