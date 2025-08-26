"""
CSV loader.

Reads up to a capped number of rows and flattens them to line text.
Keeps header row if present. Designed for quick-and-dirty indexing of tabular notes.
"""

from __future__ import annotations

import csv
from pathlib import Path


def load_csv_text(path: str | Path, *, max_rows: int = 5000) -> str:
    p = Path(path)
    if not p.exists() or p.suffix.lower() != ".csv":
        raise ValueError(f"Expected an existing .csv file, got: {path}")

    lines = []
    with p.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i == 0:
                header = [c.strip() for c in row]
                if any(header):
                    lines.append(" | ".join(header))
            else:
                vals = [c.strip() for c in row]
                if any(vals):
                    lines.append(" | ".join(vals))
            if i >= max_rows:
                lines.append(f"... (truncated at {max_rows} rows)")
                break
    return "\n".join(lines)
