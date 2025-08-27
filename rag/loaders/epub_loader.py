"""
EPUB loader.

- Uses ebooklib to parse EPUB documents.
- Extracts textual content from each DOCUMENT item via BeautifulSoup.
- Each CONTENT DOCUMENT becomes a "page" in the returned list.

Returns: list[(page_number, text)]
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from bs4 import BeautifulSoup
from ebooklib import epub  # type: ignore


def _extract_text_from_html(html: bytes | str) -> str:
    if isinstance(html, bytes):
        html = html.decode("utf-8", errors="ignore")
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines)


def load_epub(path: str | Path) -> List[Tuple[int, str]]:
    p = Path(path).expanduser().resolve()
    book = epub.read_epub(str(p))  # type: ignore

    pages: List[Tuple[int, str]] = []
    page = 1
    for item in book.get_items():  # type: ignore
        if item.get_type() == epub.ITEM_DOCUMENT:  # type: ignore
            txt = _extract_text_from_html(item.get_content())
            if txt:
                pages.append((page, txt))
                page += 1
    return pages
