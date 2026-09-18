"""
Generation through the OpenAI API, or anything that speaks its protocol.

Selected with LLM_PROVIDER=openai. OPENAI_BASE_URL points it somewhere
else, which is most of the value here: llama.cpp's own server, Ollama,
vLLM, LM Studio, OpenRouter, Together and several other vendors all expose
this shape. One adapter covers considerably more than one provider.
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

DEFAULT_MODEL = "gpt-4o-mini"


class OpenAIBackend:
    """Chat completions through the OpenAI SDK."""

    name = "openai"

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        cfg = load_config()
        self.model = model or cfg.openai_model or DEFAULT_MODEL
        self.base_url = base_url or cfg.openai_base_url
        self._api_key = api_key or cfg.openai_api_key

    def _client(self):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError(
                "The openai package is not installed.\n"
                "Install it with `pip install 'classmate-rag[openai]'`, or set "
                "LLM_PROVIDER=llama_cpp to generate locally."
            ) from e

        if not self._api_key:
            if self.base_url:
                # Local servers commonly ignore the key but the SDK still
                # requires one, so supply a placeholder rather than making
                # people invent it.
                self._api_key = "not-needed"
            else:
                raise RuntimeError(
                    "No OpenAI API key.\n"
                    "Set one with `rag config set OPENAI_API_KEY sk-...`, export "
                    "OPENAI_API_KEY, or point OPENAI_BASE_URL at a local server."
                )

        kwargs: dict[str, object] = {"api_key": self._api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        return OpenAI(**kwargs)

    def _params(self, messages, max_tokens, temperature, top_p, stop) -> dict[str, object]:
        # The message list passes through unchanged: this protocol takes the
        # system prompt as a message, which is the shape the pipeline already
        # builds.
        params: dict[str, object] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "top_p": float(top_p),
        }
        if stop:
            params["stop"] = stop
        return params

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
        response = client.chat.completions.create(**self._params(messages, max_tokens, temperature, top_p, stop))
        try:
            return (response.choices[0].message.content or "").strip()
        except (AttributeError, IndexError, TypeError):
            # An OpenAI-compatible server that returns a slightly different
            # shape should give an empty answer, not a traceback from the
            # middle of the pipeline.
            log.warning("Unexpected completion shape from %s", self.base_url or "openai")
            return ""

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
        stream = client.chat.completions.create(
            stream=True, **self._params(messages, max_tokens, temperature, top_p, stop)
        )

        for chunk in stream:
            try:
                piece = chunk.choices[0].delta.content
            except (AttributeError, IndexError, TypeError):
                # Role-only first chunks, final chunks carrying only a finish
                # reason, and keepalives all land here.
                continue
            if piece:
                yield piece


register_backend("openai", OpenAIBackend)
