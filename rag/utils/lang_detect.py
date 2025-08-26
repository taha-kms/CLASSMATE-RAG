"""
Language detection utility focused on EN/IT.

Returns two-letter tags 'en' or 'it'.
Falls back to 'en' when confidence is low or ambiguous.
"""

from __future__ import annotations

from langdetect import detect, DetectorFactory

# Make language detection deterministic across runs
DetectorFactory.seed = 42


def detect_lang_tag(text: str) -> str:
    try:
        lang = detect(text or "")
        if lang in ("en", "it"):
            return lang
        # Common fallbacks: if detection says something else or empty, default to English
        return "en"
    except Exception:
        return "en"
