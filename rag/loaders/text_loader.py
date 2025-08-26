"""
Plain text and Markdown loaders.

- TXT: read as UTF-8 with ignore errors.
- MD: strip YAML front matter and fenced code blocks, render to HTML with
      Python-Markdown, then extract readable text via BeautifulSoup.
"""

from __future__ import annotations

import re
from pathlib import Path

from bs4 import BeautifulSoup
import markdown as md

from rag.utils.text import normalize_text


_FRONT_MATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?", re.DOTALL)
# Match fenced code blocks using ``` or ~~~
_FENCED_CODE_RE = re.compile(
    r"(^|\n)```.*?\n.*?\n```|(^|\n)~~~.*?\n.*?\n~~~",
    re.DOTALL,
)


def load_txt_text(path: str | Path) -> str:
    p = Path(path)
    if not p.exists():
        raise ValueError(f"File not found: {path}")
    return p.read_text(encoding="utf-8", errors="ignore")


def _strip_front_matter(text: str) -> str:
    # Remove leading YAML front matter if present
    m = _FRONT_MATTER_RE.match(text)
    if m:
        return text[m.end() :]
    return text


def _strip_fenced_code(text: str) -> str:
    # Remove fenced code blocks to avoid indexing large code blobs
    return _FENCED_CODE_RE.sub("\n", text)


def load_md_text(path: str | Path) -> str:
    p = Path(path)
    if not p.exists() or p.suffix.lower() not in (".md", ".markdown"):
        raise ValueError(f"Expected an existing .md/.markdown file, got: {path}")

    raw = p.read_text(encoding="utf-8", errors="ignore")

    # 1) Cleanup raw Markdown
    cleaned = _strip_front_matter(raw)
    cleaned = _strip_fenced_code(cleaned)

    # 2) Render to HTML (tables, sane lists)
    html = md.markdown(
        cleaned,
        extensions=["tables", "sane_lists", "toc"],
        output_format="html5",
    )

    # 3) Extract readable text
    soup = BeautifulSoup(html, "lxml")
    # Drop non-content elements
    for tag in soup(["script", "style"]):
        tag.decompose()
    # Code/pre may remain from inline snippets; drop to reduce noise
    for tag in soup(["code", "pre"]):
        tag.decompose()
    # Convert images to their alt text if present
    for img in soup.find_all("img"):
        alt = (img.get("alt") or "").strip()
        img.replace_with(alt)

    text = soup.get_text("\n")

    # 4) Normalize whitespace
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    return normalize_text("\n".join(lines))
