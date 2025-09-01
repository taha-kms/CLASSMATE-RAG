"""
Clean up and enforce inline numeric citations like [1], [2].

What this does:
- Remove citations that point outside the available range.
- Merge adjacent citations (e.g., "[1] [2]" → "[1][2]").
- Optionally append a plain "Sources" list with the cited items.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Set


# --- Patterns for citation tokens and fixing spacing between them ---

_CIT_RE = re.compile(r"\[(\d+)\]")               # matches [number]
_ADJ_RE = re.compile(r"\]\s*(?:,?\s*)\[")        # matches "] [", "], [", "]   [", etc.


def _extract_citation_indices(text: str) -> List[int]:
    """Return all citation numbers found in the text (as ints)."""
    return [int(m.group(1)) for m in _CIT_RE.finditer(text or "")]


def _dedupe_preserve_order(nums: Iterable[int]) -> List[int]:
    """Remove duplicates from a sequence while preserving the original order."""
    seen: Set[int] = set()
    out: List[int] = []
    for n in nums:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def _remove_out_of_range(text: str, *, max_idx: int) -> str:
    """
    Drop any [n] where n < 1 or n > max_idx, then compact adjacent citations.
    """
    def _repl(m):
        n = int(m.group(1))
        if n < 1 or n > max_idx:
            return ""  # remove invalid reference
        return m.group(0)

    cleaned = _CIT_RE.sub(_repl, text or "")      # remove invalid [n]
    cleaned = _ADJ_RE.sub("][", cleaned)          # join adjacent citations
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()  # normalize extra spaces
    return cleaned


def _format_sources_block(cited_idxs: List[int], provenance: List[str], title: str) -> str:
    """
    Build a human-readable list of the cited sources in order of citation.
    Only indices that exist are included.
    """
    if not cited_idxs:
        return ""
    lines = [title]
    for i in cited_idxs:
        if 1 <= i <= len(provenance):
            lines.append(f"[{i}] {provenance[i - 1]}")
    return "\n" + "\n".join(lines)


def enforce_citations(
    answer: str,
    provenance: List[str],
    *,
    add_sources_block: bool = False,
    sources_title: str = "Sources",
) -> str:
    """
    Clean an LLM answer's in-text citations and (optionally) append a sources list.

    Args:
        answer: the raw model output that may contain [n] citations
        provenance: list of source strings; valid indices are 1..len(provenance)
        add_sources_block: if True, append a "Sources" section at the end
        sources_title: custom title for the sources section

    Returns:
        The cleaned answer (and possibly an appended sources list).
    """
    if not (answer or "").strip():
        return ""

    # 1) Remove invalid citations and compact adjacent ones.
    max_idx = len(provenance)
    cleaned = _remove_out_of_range(answer, max_idx=max_idx)

    if not add_sources_block:
        return cleaned

    # 2) Gather the unique, in-order citations that remain after cleaning.
    cited = _dedupe_preserve_order(_extract_citation_indices(cleaned))

    # 3) Append a simple "Sources" block if any valid citations exist.
    return cleaned + _format_sources_block(cited, provenance, sources_title)
