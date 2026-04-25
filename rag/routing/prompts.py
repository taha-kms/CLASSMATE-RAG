"""
Per-route system prompts.

Each prompt is a thin wrapper that:
- Names the model's specialty so behavior matches the chosen route.
- Reinforces the RAG citation contract ([1], [2], cite only the context).
- Picks the language tone matching the answer language ('en' or 'it').

The translation route is intentionally different: SalamandraTA is
translation-only and ignores chat-style instructions, so we hand it the
text to translate directly and skip the citation contract.
"""

from __future__ import annotations

from typing import Optional

from .types import Route


_BASE_CITATION_RULES_EN = (
    "Answer using ONLY the numbered context blocks provided by the user. "
    "Cite each statement with the matching [n] from the context. "
    "If the context does not contain the answer, reply exactly: I don't know."
)

_BASE_CITATION_RULES_IT = (
    "Rispondi UTILIZZANDO SOLO i blocchi di contesto numerati forniti dall'utente. "
    "Cita ogni affermazione con il [n] corrispondente. "
    "Se il contesto non contiene la risposta, rispondi esattamente: Non lo so."
)


def _base_rules(language: str) -> str:
    return _BASE_CITATION_RULES_IT if language == "it" else _BASE_CITATION_RULES_EN


def _math_prompt(language: str) -> str:
    role = (
        "Sei un tutor di matematica per studenti universitari. "
        "Risolvi problemi passo-passo, mostra i passaggi e giustifica brevemente."
        if language == "it"
        else "You are a mathematics tutor for university students. "
             "Solve problems step-by-step, show the work, and briefly justify each step."
    )
    return f"{role}\n\n{_base_rules(language)}"


def _code_prompt(language: str) -> str:
    role = (
        "Sei un assistente di programmazione. Fornisci codice corretto e idiomatico, "
        "spiega le scelte chiave e indica la complessità quando rilevante."
        if language == "it"
        else "You are a programming assistant. Provide correct, idiomatic code, "
             "explain key design choices, and note time/space complexity when relevant."
    )
    return f"{role}\n\n{_base_rules(language)}"


def _translation_prompt(language: str) -> str:
    """
    SalamandraTA is translation-only. Keep the prompt minimal and direct;
    do not impose RAG citation rules. The user message contains the source
    text and the desired target language.
    """
    if language == "it":
        return (
            "Sei un traduttore. Traduci fedelmente il testo dell'utente nella "
            "lingua di destinazione richiesta. Non aggiungere commenti."
        )
    return (
        "You are a translator. Faithfully translate the user's text into the "
        "requested target language. Do not add commentary."
    )


def _default_prompt(language: str) -> str:
    role = (
        "Sei un assistente di studio per studenti universitari. "
        "Rispondi in modo chiaro, ordinato e basato sulle fonti."
        if language == "it"
        else "You are a study assistant for university students. "
             "Answer clearly and concisely, grounded in the provided sources."
    )
    return f"{role}\n\n{_base_rules(language)}"


def system_prompt_for(route: Route, *, language: Optional[str] = None) -> str:
    """
    Return the system prompt for `route`. `language` is "en" or "it"
    (anything else is treated as "en").
    """
    lang = language if language in {"en", "it"} else "en"
    if route == "math":
        return _math_prompt(lang)
    if route == "code":
        return _code_prompt(lang)
    if route == "translation":
        return _translation_prompt(lang)
    return _default_prompt(lang)
