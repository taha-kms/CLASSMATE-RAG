"""
Generation through Google's Gemini API.

Selected with LLM_PROVIDER=gemini. The message format differs more from
the others than they do from each other, so most of this file is the
adapter rather than the call.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import TYPE_CHECKING

from rag.config import load_config
from rag.generation.backend import register_backend

if TYPE_CHECKING:  # pragma: no cover - annotation only
    from rag.routing.types import Route

log = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-2.0-flash"


class GeminiBackend:
    """Chat completions through the google-genai SDK."""

    name = "gemini"

    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        cfg = load_config()
        self.model = model or cfg.gemini_model or DEFAULT_MODEL
        self._api_key = api_key or cfg.gemini_api_key

    def _client(self):
        try:
            from google import genai
        except ImportError as e:
            raise RuntimeError(
                "The google-genai package is not installed.\n"
                "Install it with `pip install 'classmate-rag[gemini]'`, or set "
                "LLM_PROVIDER=llama_cpp to generate locally."
            ) from e

        if not self._api_key:
            raise RuntimeError(
                "No Gemini API key.\n"
                "Set one with `rag config set GEMINI_API_KEY ...`, or export "
                "GEMINI_API_KEY. Keys come from https://aistudio.google.com/apikey."
            )

        return genai.Client(api_key=self._api_key)

    @staticmethod
    def _adapt(messages: list[dict[str, str]]) -> tuple[str | None, list[dict[str, object]]]:
        """
        Convert our messages into Gemini's shape.

        Three differences, and getting any of them wrong fails quietly
        rather than loudly:

        - the system prompt is a separate config field, not a message
        - the assistant role is called "model"
        - content is a list of parts rather than a string
        """
        system: list[str] = []
        contents: list[dict[str, object]] = []

        for message in messages:
            role = message.get("role")
            text = message.get("content", "")
            if role == "system":
                if text:
                    system.append(text)
                continue
            contents.append(
                {
                    "role": "model" if role == "assistant" else "user",
                    "parts": [{"text": text}],
                }
            )

        return ("\n\n".join(system) or None), contents

    def _config(self, system, max_tokens, temperature, top_p, stop):
        from google.genai import types

        return types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=int(max_tokens),
            temperature=float(temperature),
            top_p=float(top_p),
            stop_sequences=stop or None,
        )

    def _explain(self, error: Exception) -> str:
        text = str(error)
        if "429" in text or "RESOURCE_EXHAUSTED" in text or "quota" in text.lower():
            # The free tier is rate limited aggressively, which for a student
            # project is more feature than problem, but it should read as a
            # limit rather than a crash.
            return (
                "Gemini rate limit reached. Wait a moment and ask again, or set "
                "LLM_PROVIDER=llama_cpp to generate locally."
            )
        return f"Gemini request failed: {text}"

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        route: Route | None = None,
        max_tokens: int = 768,
        temperature: float = 0.2,
        top_p: float = 0.95,
        stop: list[str] | None = None,
    ) -> str:
        client = self._client()
        system, contents = self._adapt(messages)

        try:
            response = client.models.generate_content(
                model=self.model,
                contents=contents,
                config=self._config(system, max_tokens, temperature, top_p, stop),
            )
        except Exception as e:
            raise RuntimeError(self._explain(e)) from e

        return (getattr(response, "text", "") or "").strip()

    def chat_stream(
        self,
        messages: list[dict[str, str]],
        *,
        route: Route | None = None,
        max_tokens: int = 768,
        temperature: float = 0.2,
        top_p: float = 0.95,
        stop: list[str] | None = None,
    ) -> Iterator[str]:
        client = self._client()
        system, contents = self._adapt(messages)

        try:
            stream = client.models.generate_content_stream(
                model=self.model,
                contents=contents,
                config=self._config(system, max_tokens, temperature, top_p, stop),
            )
            for chunk in stream:
                piece = getattr(chunk, "text", None)
                if piece:
                    yield piece
        except Exception as e:
            raise RuntimeError(self._explain(e)) from e


register_backend("gemini", GeminiBackend)
