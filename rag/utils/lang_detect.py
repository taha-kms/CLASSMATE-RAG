"""
Language detection utility focused on EN/IT.

Returns two-letter tags 'en' or 'it'.
Falls back to 'en' when confidence is low or ambiguous.
"""

from __future__ import annotations

from langdetect import DetectorFactory, detect
from langdetect.lang_detect_exception import LangDetectException

# Make language detection deterministic across runs
DetectorFactory.seed = 42


def detect_lang_tag(text: str) -> str:
    try:
        lang = detect(text or "")
        if lang in ("en", "it"):
            return lang
        # Common fallbacks: if detection says something else or empty, default to English
        return "en"
    except LangDetectException:
        # Too short, or no recognisable script. English is the documented
        # default; an unexpected failure type should still surface.
        return "en"
