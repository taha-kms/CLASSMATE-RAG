"""
Lightweight text normalization helpers.
"""

from __future__ import annotations

import re


_WS_RE = re.compile(r"[ \t]+")
_NL_RE = re.compile(r"\n{3,}")


def normalize_text(text: str) -> str:
    """
    Normalize whitespace:
      - collapse consecutive spaces/tabs
      - trim lines
      - collapse 3+ blank lines to 2
      - strip leading/trailing whitespace
    """
    if not text:
        return ""
    # Replace tabs/multiple spaces inside lines
    lines = []
    for line in text.splitlines():
        line = _WS_RE.sub(" ", line).strip()
        lines.append(line)
    out = "\n".join(lines)
    # Collapse excessive blank lines
    out = _NL_RE.sub("\n\n", out)
    return out.strip()
