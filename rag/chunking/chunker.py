# rag/chunking/chunker.py
"""
Sentence-aware chunking with overlap for EN/IT documents.
(…docstring unchanged…)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

# Fixed-width lookbehind on sentence-ending punctuation only.
# Paragraph breaks are handled separately.
_SENT_BOUNDARY = re.compile(
    r"""
    (?<=[.!?])         # fixed-width lookbehind: end punctuation
    \s+                # whitespace after boundary
    (?=                # next char indicates sentence start (heuristic)
        [A-ZÀ-ÖØ-Þ]    # Latin uppercase incl. accented (IT)
        | ["“‘(]       # quotes or opening paren
        | \n           # or a newline
    )
    """,
    re.VERBOSE,
)

_ABBREV = {
    "sig.", "sig.ra", "sig.na", "ing.", "dott.", "dr.", "prof.", "ecc.", "etc.", "e.g.", "i.e.",
}

_WHITESPACE = re.compile(r"[ \t]+")
_MULTI_NL = re.compile(r"\n{3,}")   # keep up to double newlines

@dataclass(frozen=True)
class RagChunk:
    page: int
    chunk_id: int
    text: str

def _normalize_for_split(text: str) -> str:
    if not text:
        return ""
    lines = []
    for ln in text.splitlines():
        lines.append(_WHITESPACE.sub(" ", ln).strip())
    out = "\n".join(lines)
    out = _MULTI_NL.sub("\n\n", out)
    return out.strip()

def _split_paragraphs(text: str) -> List[str]:
    return [p for p in re.split(r"\n{2,}", text) if p.strip()]

def _split_sentences_in_paragraph(par: str) -> List[str]:
    parts = _SENT_BOUNDARY.split(par)
    out: List[str] = []
    buf = ""
    for part in parts:
        seg = part.strip()
        if not seg:
            continue
        cand = (buf + " " + seg).strip() if buf else seg
        prev = buf.strip().split()[-1].lower() if buf else ""
        if prev in _ABBREV and (not seg[:1].isupper()):
            buf = cand
            continue
        if buf:
            out.append(buf.strip())
        buf = seg
    if buf:
        out.append(buf.strip())
    out = [s for s in out if s and not all(ch in ".!?,;:()[]{}\"'—–-" for ch in s)]
    return out

def sentence_split(text: str) -> List[str]:
    t = _normalize_for_split(text)
    if not t:
        return []
    sents: List[str] = []
    for par in _split_paragraphs(t):
        sents.extend(_split_sentences_in_paragraph(par))
    return sents

def _pack_sentences(sents: Sequence[str], *, chunk_size: int) -> List[List[str]]:
    chunks: List[List[str]] = []
    cur: List[str] = []
    cur_len = 0
    def flush():
        nonlocal cur, cur_len
        if cur: chunks.append(cur)
        cur, cur_len = [], 0
    for s in sents:
        slen = len(s)
        if slen > chunk_size:
            flush()
            start = 0
            while start < slen:
                end = min(start + chunk_size, slen)
                chunks.append([s[start:end]])
                start = end
            continue
        extra = 1 if cur_len > 0 else 0
        if cur_len + extra + slen <= chunk_size:
            cur.append(s); cur_len += extra + slen
        else:
            flush(); cur.append(s); cur_len = slen
    flush()
    return chunks

def _compute_sentence_overlap(sent_block: List[str], target_overlap_chars: int) -> int:
    if not sent_block or target_overlap_chars <= 0:
        return 0
    total = 0
    n = 0
    for s in reversed(sent_block):
        n += 1
        total += len(s) + (1 if total > 0 else 0)
        if total >= target_overlap_chars:
            break
    return min(n, max(0, len(sent_block) - 1))

def chunk_text(
    text: str,
    *,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    page: int = 1,
    starting_chunk_id: int = 0,
) -> List[RagChunk]:
    sents = sentence_split(text)
    if not sents:
        if text.strip():
            return [RagChunk(page=page, chunk_id=starting_chunk_id, text=text.strip())]
        return []
    packed = _pack_sentences(sents, chunk_size=chunk_size)
    overlapped: List[List[str]] = []
    for i, block in enumerate(packed):
        if i == 0:
            overlapped.append(block); continue
        prev = overlapped[-1]
        n_overlap = _compute_sentence_overlap(prev, chunk_overlap)
        merged = (prev[-n_overlap:] + block) if n_overlap > 0 else block
        overlapped.append(merged)
    chunks: List[RagChunk] = []
    cid = starting_chunk_id
    for block in overlapped:
        txt = " ".join(block).strip()
        if txt:
            chunks.append(RagChunk(page=page, chunk_id=cid, text=txt))
            cid += 1
    return chunks

def chunk_pages(
    pages: Iterable[Tuple[int, str]],
    *,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    starting_chunk_id: int = 0,
) -> List[Tuple[int, int, str]]:
    out: List[Tuple[int, int, str]] = []
    cid = starting_chunk_id
    for page, text in pages:
        chs = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap, page=page, starting_chunk_id=cid)
        for c in chs:
            out.append((c.page, c.chunk_id, c.text))
        if chs:
            cid = chs[-1].chunk_id + 1
    return out
