"""
Citation post-processing for CLASSMATE-RAG.

Goals
- Remove out-of-range [n] citations (e.g., hallucinated [9] when only [1..K] exist).
- Normalize/compact adjacent citations: "[1] [2]" -> "[1][2]" and " [1], [2] " -> "[1][2]".
- Optionally append a human-readable "Sources" block listing the cited items.

Public API:
    enforce_citations(
        answer: str,
        provenance: list[str],
        *,
        add_sources_block: bool = False,
        sources_title: str = "Sources"
    ) -> str
"""

from __future__ import annotations

import re
from typing import Iterable, List, Set


# --- Regexes for [n] tokens and adjacency cleanup ---

_CIT_RE = re.compile(r"\[(\d+)\]")
# normalize separators between adjacent citations: "] [", "], [", "]  [", etc.
_ADJ_RE = re.compile(r"\]\s*(?:,?\s*)\[")


def _extract_citation_indices(text: str) -> List[int]:
    return [int(m.group(1)) for m in _CIT_RE.finditer(text or "")]


def _dedupe_preserve_order(nums: Iterable[int]) -> List[int]:
    seen: Set[int] = set()
    out: List[int] = []
    for n in nums:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def _remove_out_of_range(text: str, *, max_idx: int) -> str:
    """
    Removes any [n] where n < 1 or n > max_idx.
    """
    def _repl(m):
        n = int(m.group(1))
        if n < 1 or n > max_idx:
            return ""  # drop hallucinated reference
        return m.group(0)
    # pass 1: drop bad refs
    cleaned = _CIT_RE.sub(_repl, text or "")
    # pass 2: collapse leftover whitespace between adjacent citations
    cleaned = _ADJ_RE.sub("][", cleaned)
    # also collapse any double spaces created
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned


def _format_sources_block(cited_idxs: List[int], provenance: List[str], title: str) -> str:
    """
    Build a human-readable list of cited sources in ascending citation order.
    Only include indices that actually appeared in the (cleaned) answer.
    """
    if not cited_idxs:
        return ""
    lines = [title]
    for i in cited_idxs:
        # guard against out-of-range even after cleaning
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
    Sanitize inline citations in 'answer' based on provided 'provenance' list.
    - Drops any [n] with n outside 1..len(provenance).
    - Compacts adjacent citations to "[1][2]".
    - Optionally appends a Sources block listing only the cited ones.

    Returns the modified answer (may be identical if nothing to fix).
    """
    if not answer:
        return answer or ""

    max_idx = len(provenance or [])
    cleaned = _remove_out_of_range(answer, max_idx=max_idx)

    # collect only citations that survived cleaning
    cited = _extract_citation_indices(cleaned)
    cited = _dedupe_preserve_order(cited)

    if add_sources_block and cited:
        cleaned += _format_sources_block(cited, provenance or [], sources_title)

    return cleaned
