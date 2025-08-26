"""
Sentence-aware chunking with overlap for EN/IT documents.

Goals:
- Prefer sentence boundaries (better retrieval quality for e5).
- Greedy packing up to ~chunk_size characters.
- Overlap controlled by ~chunk_overlap characters, implemented via sentence overlap.
- Robust fallback for very long sentences (char slicing).
- Works on single large texts or per-page lists.

Public API:
- sentence_split(text) -> list[str]
- chunk_text(text, chunk_size=1000, chunk_overlap=150) -> list[RagChunk]
- chunk_pages(pages, chunk_size=1000, chunk_overlap=150) -> list[(page, chunk_id, text)]

Notes:
- We keep it lightweight (no external tokenizers) and tuned for EN/IT punctuation.
- If you later add a tokenizer (e.g., tiktoken), you can swap packing logic without changing call sites.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple


# Uppercase ranges include accented Latin letters common in Italian.
_SENT_BOUNDARY = re.compile(
    r"""
    (?<=            # assert position follows one of:
    [.!?]           #   end punctuation
    |               #   OR
    \n{2,}          #   paragraph break
    )
    \s+             # whitespace after boundary
    (?=             # next char indicates sentence start (heuristic)
    [A-ZÀ-ÖØ-Þ]     # Latin uppercase incl. accented
    | \" | \“ | \‘ | \(
    | \n            # or a new line
    )
    """,
    re.VERBOSE,
)

# Heuristic: periods used in abbreviations we *don't* want to split on
_ABBREV = {
    "sig.", "sig.ra", "sig.na", "ing.", "dott.", "dr.", "prof.", "ecc.", "etc.", "e.g.", "i.e.",
}

_WHITESPACE = re.compile(r"[ \t]+")
_MULTI_NL = re.compile(r"\n{3,}")


@dataclass(frozen=True)
class RagChunk:
    page: int
    chunk_id: int
    text: str


def _normalize_for_split(text: str) -> str:
    if not text:
        return ""
    # Collapse spaces/tabs inside lines and trim lines
    lines = []
    for ln in text.splitlines():
        lines.append(_WHITESPACE.sub(" ", ln).strip())
    out = "\n".join(lines)
    # Collapse excessive blank lines
    out = _MULTI_NL.sub("\n\n", out)
    return out.strip()


def sentence_split(text: str) -> List[str]:
    """
    Lightweight EN/IT sentence splitter:
    - Splits on ., !, ?, and paragraph breaks.
    - Tries to avoid common abbreviations.
    - Keeps punctuation attached to sentences.
    """
    t = _normalize_for_split(text)
    if not t:
        return []
    # First, split on our heuristic boundaries
    parts = _SENT_BOUNDARY.split(t)

    # Re-stitch splits that break on common abbreviations
    out: List[str] = []
    buf = ""
    for part in parts:
        seg = part.strip()
        if not seg:
            continue
        if buf:
            cand = (buf + " " + seg).strip()
        else:
            cand = seg

        # If previous piece ends with a known abbreviation, don't commit a split yet
        prev = buf.strip().split()[-1].lower() if buf else ""
        if prev in _ABBREV and not seg[:1].isupper():
            buf = cand
            continue

        # Otherwise, commit previous buffer and start a new one
        if buf:
            out.append(buf.strip())
        buf = seg

    if buf:
        out.append(buf.strip())

    # Final cleanup: drop tiny fragments that are just punctuation
    out = [s for s in out if s and not all(ch in ".!?,;:()[]{}\"'—–-" for ch in s)]
    return out


def _pack_sentences(
    sents: Sequence[str],
    *,
    chunk_size: int,
) -> List[List[str]]:
    """
    Greedy packing of sentences into chunks up to ~chunk_size chars.
    If a single sentence > chunk_size, we char-slice it.
    """
    chunks: List[List[str]] = []
    cur: List[str] = []
    cur_len = 0

    def flush():
        nonlocal cur, cur_len
        if cur:
            chunks.append(cur)
        cur = []
        cur_len = 0

    for s in sents:
        slen = len(s)
        # Oversized single sentence: slice into char windows
        if slen > chunk_size:
            # flush current
            flush()
            start = 0
            while start < slen:
                end = min(start + chunk_size, slen)
                chunks.append([s[start:end]])
                start = end
            continue

        # Greedy add
        if cur_len + (1 if cur else 0) + slen <= chunk_size:
            cur.append(s)
            cur_len += (1 if cur_len > 0 else 0) + slen
        else:
            # flush and start new
            flush()
            cur.append(s)
            cur_len = slen

    flush()
    return chunks


def _compute_sentence_overlap(sent_block: List[str], target_overlap_chars: int) -> int:
    """
    Determine how many trailing sentences to overlap with the next chunk
    to reach roughly target_overlap_chars (not exceeding block length).
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
    # Do not overlap the entire block
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
    Chunk a single large text into RagChunk items using sentence-aware packing.
    Returns list of RagChunk with (page, chunk_id, text).
    """
    sents = sentence_split(text)
    if not sents:
        if text.strip():
            return [RagChunk(page=page, chunk_id=starting_chunk_id, text=text.strip())]
        return []

    packed = _pack_sentences(sents, chunk_size=chunk_size)
    # Apply sentence-level overlap between consecutive packs
    overlapped: List[List[str]] = []
    for i, block in enumerate(packed):
        if i == 0:
            overlapped.append(block)
            continue
        # Determine how many sentences from previous block to prepend
        prev = overlapped[-1]
        n_overlap = _compute_sentence_overlap(prev, chunk_overlap)
        if n_overlap > 0:
            merged = prev[-n_overlap:] + block
        else:
            merged = block
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
    Chunk a sequence of (page_number, text) items.
    Returns a list of (page, chunk_id, text). Page numbering is preserved.

    Note: We keep chunks within their original pages for provenance.
    """
    out: List[Tuple[int, int, str]] = []
    cid = starting_chunk_id
    for page, text in pages:
        chs = chunk_text(
            text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            page=page,
            starting_chunk_id=cid,
        )
        for c in chs:
            out.append((c.page, c.chunk_id, c.text))
        if chs:
            cid = chs[-1].chunk_id + 1
    return out
