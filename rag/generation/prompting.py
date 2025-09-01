"""
Helpers for building prompts for the language model.
Includes:
- format_context_blocks: turn retrieved chunks into context text
- build_grounded_messages: prompt with context + citations
- build_general_messages: prompt without context
"""

from __future__ import annotations

from typing import List, Sequence, Tuple, Dict, Any


def format_context_blocks(
    retrieved: Sequence[Dict[str, Any]],
    *,
    max_total_chars: int | None = 2000,
) -> Tuple[str, List[str]]:
    """
    Turn retrieved results into a single text block and a list of sources.

    Args:
        retrieved: list of dicts with 'document' and 'metadata'
        max_total_chars: optional cutoff for total characters

    Returns:
        context_text: concatenated chunks with [n] labels
        provenance: list of source descriptions in the same order
    """
    blocks: List[str] = []
    prov: List[str] = []
    total_chars = 0
    for i, r in enumerate(retrieved, start=1):
        text = (r.get("document") or "").strip()
        meta = r.get("metadata") or {}
        src = str(meta.get("source_path") or "")
        prov.append(src if src else f"chunk-{i}")
        if not text:
            continue
        block = f"[{i}] {text}"
        if max_total_chars is not None and total_chars + len(block) > max_total_chars:
            break
        blocks.append(block)
        total_chars += len(block)
    return "\n\n".join(blocks), prov


def build_grounded_messages(
    question: str,
    context_text: str,
    *,
    citations_required: bool = True,
) -> List[Dict[str, str]]:
    """
    Build a grounded prompt (with context and citations).

    Args:
        question: the user question
        context_text: retrieved chunks formatted with [n]
        citations_required: tell model to cite sources or not

    Returns:
        A list of role/content dicts for the model
    """
    sys = (
        "You are a helpful assistant that answers questions "
        "using the provided context. "
    )
    if citations_required:
        sys += (
            "Include numeric citations [1], [2], ... "
            "next to the statements you make. "
            "Cite only from the provided context."
        )
    else:
        sys += "You may use the provided context, but citations are optional."

    user = f"Context:\n{context_text}\n\nQuestion:\n{question}\n\nAnswer:"
    return [{"role": "system", "content": sys}, {"role": "user", "content": user}]


def build_general_messages(question: str) -> List[Dict[str, str]]:
    """
    Build a simple prompt without context (general Q&A).
    """
    sys = "You are a helpful assistant that answers general questions."
    return [{"role": "system", "content": sys}, {"role": "user", "content": question}]
