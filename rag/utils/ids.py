"""
Deterministic ID helpers for documents/chunks.

We generate stable IDs from:
  (source_path, page, chunk_index, course, unit)

This ensures upserts are idempotent and re-ingesting the same file does not create duplicates.
"""

from __future__ import annotations

from hashlib import blake2b
from pathlib import Path
from typing import Optional


def stable_chunk_id(
    *,
    source_path: str | Path,
    page: int,
    chunk_index: int,
    course: Optional[str] = None,
    unit: Optional[str] = None,
    prefix: str = "cm_",
) -> str:
    sp = str(Path(source_path).resolve())
    key = f"{sp}|{page}|{chunk_index}|{course or ''}|{unit or ''}"
    h = blake2b(key.encode("utf-8"), digest_size=16).hexdigest()
    return f"{prefix}{h}"
