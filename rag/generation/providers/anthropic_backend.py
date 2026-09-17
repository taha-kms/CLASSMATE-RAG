"""
Generation through the Anthropic API.

Selected with LLM_PROVIDER=anthropic. Needs ANTHROPIC_API_KEY, which can
live in the credential store rather than the environment (#95), and the
`anthropic` package, which is an optional extra rather than a base
dependency: someone running a local GGUF should not be made to install
three provider SDKs.
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

#: Anthropic has no concept of swapping a local model per subject, so a
#: route selects a model name instead. Everything maps to one model by
#: default; ANTHROPIC_MODEL overrides it.
DEFAULT_MODEL = "claude-opus-5"


class AnthropicBackend:
    """Chat completions through the Anthropic SDK."""

    name = "anthropic"

    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        cfg = load_config()
        self.model = model or cfg.anthropic_model or DEFAULT_MODEL
        self._api_key = api_key or cfg.anthropic_api_key

    def _client(self):
        try:
            import anthropic
        except ImportError as e:
            raise RuntimeError(
                "The anthropic package is not installed.\n"
                "Install it with `pip install 'classmate-rag[anthropic]'`, or set "
                "LLM_PROVIDER=llama_cpp to generate locally."
            ) from e

        if not self._api_key:
            raise RuntimeError(
                "No Anthropic API key.\n"
                "Set one with `rag config set ANTHROPIC_API_KEY sk-ant-...`, or "
                "export ANTHROPIC_API_KEY."
            )

        return anthropic.Anthropic(api_key=self._api_key)

    @staticmethod
    def _split(messages: list[dict[str, str]]) -> tuple[str | None, list[dict[str, str]]]:
        """
        Separate the system prompt from the conversation.

        Anthropic takes the system prompt as its own parameter rather than a
        message with role "system", so passing our message list through
        unchanged would silently drop the route prompt.
        """
        system: list[str] = []
        rest: list[dict[str, str]] = []
        for message in messages:
            if message.get("role") == "system":
                system.append(message.get("content", ""))
            else:
                rest.append(message)
        return ("\n\n".join(s for s in system if s) or None), rest

    def _params(self, messages, max_tokens, temperature, top_p, stop):
        system, conversation = self._split(messages)
        params: dict[str, object] = {
            "model": self.model,
            "max_tokens": int(max_tokens),
            "messages": conversation,
        }
        if system:
            params["system"] = system
        if stop:
            params["stop_sequences"] = stop
        # temperature and top_p are rejected on current models, which use
        # adaptive thinking instead; they are accepted here and dropped so
        # the pipeline does not need to know which provider it is talking to.
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
        params = self._params(messages, max_tokens, temperature, top_p, stop)

        # Streaming even for the non-streaming call: a long answer on a
        # non-streamed request can exceed the SDK's HTTP timeout.
        with client.messages.stream(**params) as stream:
            message = stream.get_final_message()

        if getattr(message, "stop_reason", None) == "refusal":
            log.warning("Anthropic declined the request")
            return ""

        return "".join(block.text for block in message.content if getattr(block, "type", None) == "text").strip()

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
        params = self._params(messages, max_tokens, temperature, top_p, stop)

        with client.messages.stream(**params) as stream:
            yield from stream.text_stream


register_backend("anthropic", AnthropicBackend)
