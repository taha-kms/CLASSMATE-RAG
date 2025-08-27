"""
HTML loader with readability-style cleaning.

- Uses readability-lxml to extract the main article/content.
- Falls back to BeautifulSoup text extraction if readability fails.
- Returns a list of (page_number, text) tuples; for HTML we treat the whole
  document as a single "page" (page=1).

Dependencies:
    readability-lxml, beautifulsoup4, lxml
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from bs4 import BeautifulSoup
from readability import Document  # type: ignore


def _read_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _fallback_bs(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    # normalize excess blank lines
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines)


def load_html_readable(path: str | Path) -> List[Tuple[int, str]]:
    p = Path(path).expanduser().resolve()
    html = _read_file(p)

    try:
        doc = Document(html)
        main_html = doc.summary(html_partial=True)  # type: ignore
        soup = BeautifulSoup(main_html, "lxml")
        text = soup.get_text(separator="\n")
    except Exception:
        text = _fallback_bs(html)

    text = (text or "").strip()
    return [(1, text)] if text else []
