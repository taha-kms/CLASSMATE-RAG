"""
Sentence-aware text chunking with overlap for English and Italian documents.

Functions provided:
- sentence_split: split text into clean sentences
- chunk_text: split text into overlapping chunks of sentences
- chunk_pages: apply chunking across multiple pages
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

# Regex to detect sentence boundaries (period/question mark/exclamation mark).
# Handles uppercase, quotes, and newlines after punctuation.
_SENT_BOUNDARY = re.compile(
    r"""
    (?<=[.!?])         # sentence-ending punctuation
    \s+                # whitespace after punctuation
    (?=                # next character starts a new sentence
        [A-ZÀ-ÖØ-Þ]    # uppercase letters (English + accented)
        | ["“‘(]       # quotes or parentheses
        | \n           # or a newline
    )
    """,
    re.VERBOSE,
)

# Common abbreviations to avoid splitting incorrectly
_ABBREV = {
    "sig.", "sig.ra", "sig.na", "ing.", "dott.", "dr.", "prof.", "ecc.",
    "etc.", "e.g.", "i.e.",
}

# Regex helpers for cleaning
_WHITESPACE = re.compile(r"[ \t]+")
_MULTI_NL = re.compile(r"\n{3,}")   # shrink 3+ newlines into 2


@dataclass(frozen=True)
class RagChunk:
    """Represents one chunk of text, with page number and chunk ID."""
    page: int
    chunk_id: int
    text: str


# ------------------------------
# Normalization and splitting
# ------------------------------

def _normalize_for_split(text: str) -> str:
    """Clean up whitespace and reduce multiple newlines before splitting."""
    if not text:
        return ""
    lines = []
    for ln in text.splitlines():
        lines.append(_WHITESPACE.sub(" ", ln).strip())
    out = "\n".join(lines)
    out = _MULTI_NL.sub("\n\n", out)
    return out.strip()

def _split_paragraphs(text: str) -> List[str]:
    """Split text into paragraphs (separated by 2+ newlines)."""
    return [p for p in re.split(r"\n{2,}", text) if p.strip()]

def _split_sentences_in_paragraph(par: str) -> List[str]:
    """
    Split one paragraph into sentences.
    Avoid splitting after known abbreviations.
    """
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
    # Drop sentences that are only punctuation
    out = [s for s in out if s and not all(ch in ".!?,;:()[]{}\"'—–-" for ch in s)]
    return out

def sentence_split(text: str) -> List[str]:
    """Split text into a clean list of sentences."""
    t = _normalize_for_split(text)
    if not t:
        return []
    sents: List[str] = []
    for par in _split_paragraphs(t):
        sents.extend(_split_sentences_in_paragraph(par))
    return sents


# ------------------------------
# Chunking
# ------------------------------

def _pack_sentences(sents: Sequence[str], *, chunk_size: int) -> List[List[str]]:
    """
    Pack sentences into groups that fit within chunk_size characters.
    If a sentence is too long, split it into smaller pieces.
    """
    chunks: List[List[str]] = []
    cur: List[str] = []
    cur_len = 0

    def flush():
        nonlocal cur, cur_len
        if cur:
            chunks.append(cur)
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
            cur.append(s)
            cur_len += extra + slen
        else:
            flush()
            cur.append(s)
            cur_len = slen
    flush()
    return chunks

def _compute_sentence_overlap(sent_block: List[str], target_overlap_chars: int) -> int:
    """
    Decide how many sentences from the previous chunk should overlap
    with the next one, based on target overlap in characters.
    """
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
    """
    Split text into overlapping chunks of sentences.
    Returns a list of RagChunk objects.
    """
    sents = sentence_split(text)
    if not sents:
        if text.strip():
            return [RagChunk(page=page, chunk_id=starting_chunk_id, text=text.strip())]
        return []
    packed = _pack_sentences(sents, chunk_size=chunk_size)
    overlapped: List[List[str]] = []
    for i, block in enumerate(packed):
        if i == 0:
            overlapped.append(block)
            continue
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
    """
    Apply chunking across multiple pages.
    Returns list of tuples: (page number, chunk ID, chunk text).
    """
    out: List[Tuple[int, int, str]] = []
    cid = starting_chunk_id
    for page, text in pages:
        chs = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap, page=page, starting_chunk_id=cid)
        for c in chs:
            out.append((c.page, c.chunk_id, c.text))
        if chs:
            cid = chs[-1].chunk_id + 1
    return out
