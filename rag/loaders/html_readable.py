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

import logging
from pathlib import Path

from bs4 import BeautifulSoup
from readability import Document  # type: ignore

log = logging.getLogger(__name__)


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


def load_html_readable(path: str | Path) -> list[tuple[int, str]]:
    p = Path(path).expanduser().resolve()
    html = _read_file(p)

    try:
        doc = Document(html)
        main_html = doc.summary(html_partial=True)  # type: ignore
        soup = BeautifulSoup(main_html, "lxml")
        text = soup.get_text(separator="\n")
    except Exception:  # noqa: BLE001 - readability/lxml on arbitrary HTML
        # Real-world HTML breaks these parsers in open-ended ways, and the
        # whole point of _fallback_bs is to cope. Log so a systematic failure
        # is visible rather than silently degrading every page.
        log.debug("readability extraction failed, using the plain fallback", exc_info=True)
        text = _fallback_bs(html)

    text = (text or "").strip()
    return [(1, text)] if text else []
