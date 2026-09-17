"""
A provider-agnostic seam for generation.

Generation was hardwired to llama.cpp: ask_question_stream built a
LlamaCppRunner or reached for StickyModelLoader directly, so adding a
hosted provider would have meant teaching every one of those call sites
about it.

What a backend owns is narrow on purpose: turn messages into text, either
all at once or as it arrives. Everything else stays in the pipeline, where
it is provider-independent and already tested. The no-context fallback,
citation attribution, translate-on-miss and language selection are not
things a provider should have opinions about.

Routes are passed through rather than resolved here. For llama.cpp a route
selects which GGUF is resident; for a hosted provider it would select a
model name. Both are the backend's business, and neither is the
pipeline's.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from rag.config import load_config

if TYPE_CHECKING:  # pragma: no cover - annotation only
    from rag.routing.types import Route


@runtime_checkable
class ChatBackend(Protocol):
    """What generation needs from a provider, and nothing more."""

    #: Identifier used by LLM_PROVIDER.
    name: str

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
        """Generate a complete answer."""
        ...

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
        """Generate an answer, yielding it as it is produced."""
        ...


class LlamaCppBackend:
    """
    Local generation through llama.cpp.

    Handles both shapes the pipeline uses. With a route, the sticky loader
    keeps one GGUF resident and swaps when the route changes. Without one,
    a single configured model answers everything.
    """

    name = "llama_cpp"

    def _runner(self):
        from rag.generation import LlamaCppRunner

        # Built per call rather than cached: the non-routed path constructs
        # it once per question today, and caching it here would change when
        # a model is loaded without anyone asking for that.
        return LlamaCppRunner()

    def _loader(self):
        from rag.pipeline.rag import _get_model_loader

        return _get_model_loader()

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
        if route is not None:
            return self._loader().chat(
                route=route,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
            )
        return self._runner().chat(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
        )

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
        if route is not None:
            return self._loader().chat_stream(
                route=route,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
            )
        return self._runner().chat_stream(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
        )


#: provider name -> a callable returning a backend. Providers register here
#: rather than being imported eagerly, so an SDK that is not installed costs
#: nothing until someone selects it.
_BACKENDS: dict[str, callable] = {
    "llama_cpp": LlamaCppBackend,
}


def register_backend(name: str, factory) -> None:
    """Make a provider selectable as LLM_PROVIDER=<name>."""
    _BACKENDS[name] = factory


def available_backends() -> list[str]:
    return sorted(_BACKENDS)


def get_backend(name: str | None = None) -> ChatBackend:
    """
    The configured backend, or the one named.

    Defaults to llama.cpp, so nothing changes for anyone who has not opted
    into a provider.
    """
    name = (name or load_config().llm_provider or "llama_cpp").strip().lower()
    factory = _BACKENDS.get(name)
    if factory is None:
        raise ValueError(f"Unknown LLM_PROVIDER '{name}'. Available: {', '.join(available_backends())}.")
    return factory()
