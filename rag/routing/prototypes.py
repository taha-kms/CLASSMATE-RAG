"""
Seed phrases per route. The classifier embeds each list, averages to a
single prototype vector, and scores queries/documents by cosine similarity.

Edit these lists to tune routing without code changes. Keep prototypes
short and topic-pure: noisy seeds drag a route's score down everywhere.

The "default" route deliberately has no prototype — it's the fallback
when no other route is confident.
"""

from __future__ import annotations

from typing import Dict, List

from .types import Route

SUBJECT_PROTOTYPES: Dict[Route, List[str]] = {
    "math": [
        "solve this equation step by step",
        "calculus derivative integral limit",
        "linear algebra matrix vector eigenvalue",
        "prove this theorem using induction",
        "probability statistics distribution variance",
        "algebra polynomial factor quadratic",
        "geometry trigonometry sine cosine",
        "differential equation solution",
    ],
    "code": [
        "debug this Python function",
        "implement an algorithm in C++",
        "write unit tests for this code",
        "refactor this class for readability",
        "explain this stack trace",
        "data structure linked list binary tree",
        "time complexity big O analysis",
        "regex parse string javascript typescript",
    ],
    "translation": [
        "translate this from English to Italian",
        "traduci questo testo in inglese",
        "what does this Italian phrase mean in English",
        "come si dice in italiano",
        "translation of this paragraph",
        "translate the following sentence",
        "italian grammar conjugation tense",
    ],
    # No prototype: chosen by elimination when nothing else is confident.
    "default": [],
}


# Explicit translate-intent triggers — used as a hard requirement when
# Config.route_translation_requires_intent is True (default), because
# SalamandraTA is a translation-only model and must not be used for
# general Q&A even if the prototype score is high.
TRANSLATION_INTENT_KEYWORDS: tuple[str, ...] = (
    "translate",
    "translation",
    "traduci",
    "traduce",
    "traduzione",
    "traduco",
    "translator",
    "in english",
    "in italian",
    "in italiano",
    "in inglese",
    "how do you say",
    "how to say",
    "come si dice",
    "what does",  # combined with another lang signal upstream
)
