"""
HTML loader.

Strips scripts/styles and extracts visible text via BeautifulSoup.
"""

from __future__ import annotations

from pathlib import Path

from bs4 import BeautifulSoup


def load_html_text(path: str | Path) -> str:
    p = Path(path)
    if not p.exists() or p.suffix.lower() not in (".html", ".htm"):
        raise ValueError(f"Expected an existing .html/.htm file, got: {path}")

    html = p.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text("\n")
    # Simple cleanup: collapse extra blank lines
    lines = [line.strip() for line in text.splitlines()]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines)
