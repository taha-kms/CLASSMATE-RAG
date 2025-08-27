"""
CSV -> bullet text loader.

- Reads CSV via Python stdlib.
- Converts each row to a one-line bullet with "col: value" pairs.
- Splits output into multiple "pages" every N rows to avoid giant pages.

Returns: list[(page_number, text)]
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, List, Tuple


def _iter_rows(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield {k or "": (v or "").strip() for k, v in row.items()}


def _row_to_bullet(row: dict) -> str:
    parts = []
    for k, v in row.items():
        k2 = (k or "").strip()
        v2 = (v or "").strip()
        if not k2 and not v2:
            continue
        if k2 and v2:
            parts.append(f"{k2}: {v2}")
        elif k2:
            parts.append(f"{k2}:")
        else:
            parts.append(v2)
    return "- " + "; ".join(parts) if parts else ""


def load_csv_bullets(path: str | Path, *, rows_per_page: int = 80) -> List[Tuple[int, str]]:
    p = Path(path).expanduser().resolve()
    bullets: List[str] = []
    for row in _iter_rows(p):
        b = _row_to_bullet(row)
        if b:
            bullets.append(b)

    if not bullets:
        return []

    pages: List[Tuple[int, str]] = []
    page = 1
    for i in range(0, len(bullets), rows_per_page):
        chunk = "\n".join(bullets[i : i + rows_per_page])
        pages.append((page, chunk))
        page += 1
    return pages
