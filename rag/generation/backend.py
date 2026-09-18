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

import threading
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

    #: The unrouted runner, built once for the process.
    #:
    #: It used to be built per call, which was harmless while the only caller
    #: was a CLI that asks one question and exits. The API asks concurrently,
    #: and LlamaCppRunner.__init__ calls load_llama eagerly, so every request
    #: was loading its own copy of the GGUF -- hundreds of megabytes each,
    #: several at a time. Callers still only load a model by asking a
    #: question; they just stop re-loading it.
    _shared_runner = None
    _runner_lock = threading.Lock()

    def _runner(self):
        from rag.generation import LlamaCppRunner

        # Callers hold the generation lock by the time they get here, so this
        # second lock only guards against a non-generating caller racing in.
        with LlamaCppBackend._runner_lock:
            if LlamaCppBackend._shared_runner is None:
                LlamaCppBackend._shared_runner = LlamaCppRunner()
            return LlamaCppBackend._shared_runner

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
        # Same single-generation guarantee as chat_stream below. Simpler here
        # because there is no generator to hand back: the whole call runs
        # inside the lock.
        from rag.routing.loader import generation_lock

        with generation_lock():
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

        # The unrouted path takes the same lock the sticky loader does, and for
        # the same reason: one generation at a time, one model resident. It was
        # missing here, and since ENABLE_ROUTING defaults to false this is the
        # path almost everyone runs -- so concurrent callers were generating
        # simultaneously, each against its own freshly loaded model.
        #
        # Acquired before the generator is returned, exactly as the loader
        # does: a generator body does not run until the first next(), so
        # locking inside would let two callers hold un-started generators and
        # then both load a model.
        from rag.routing.loader import generation_lock

        lock = generation_lock()
        lock.acquire()
        try:
            stream = self._runner().chat_stream(
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
            )
        except BaseException:
            lock.release()
            raise

        def _guarded() -> Iterator[str]:
            try:
                yield from stream
            finally:
                lock.release()

        return _guarded()


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


#: Providers already warned about in this process, so the notice appears
#: once rather than before every question.
_ANNOUNCED: set[str] = set()


def _announce_if_hosted(name: str) -> None:
    """Say what leaves the machine, the first time a hosted provider is used."""
    import sys

    from rag.generation.privacy import data_sent_notice

    if name in _ANNOUNCED:
        return
    _ANNOUNCED.add(name)

    notice = data_sent_notice(name)
    if notice:
        # stderr, so it never lands in the JSON on stdout.
        print(f"\n{notice}\n", file=sys.stderr)


def get_backend(name: str | None = None) -> ChatBackend:
    """
    The configured backend, or the one named.

    Defaults to llama.cpp, so nothing changes for anyone who has not opted
    into a provider.
    """
    name = (name or load_config().llm_provider or "llama_cpp").strip().lower()

    # Providers register on import. Done here rather than at module scope so
    # importing this module never pulls in a provider SDK.
    if name not in _BACKENDS:
        from rag.generation.providers import register_all

        register_all()
    factory = _BACKENDS.get(name)
    if factory is None:
        raise ValueError(f"Unknown LLM_PROVIDER '{name}'. Available: {', '.join(available_backends())}.")
    _announce_if_hosted(name)
    return factory()
