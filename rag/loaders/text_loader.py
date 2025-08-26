"""
Plain text and Markdown loaders.
"""

from __future__ import annotations

from pathlib import Path


def load_txt_text(path: str | Path) -> str:
    p = Path(path)
    if not p.exists():
        raise ValueError(f"File not found: {path}")
    return p.read_text(encoding="utf-8", errors="ignore")


def load_md_text(path: str | Path) -> str:
    # For now, treat Markdown as plain text; optional stripping can be added later
    return load_txt_text(path)
