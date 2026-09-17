"""
Sticky single-model loader.

A 7B Q4_K_M GGUF is roughly 4.4 GB, so only one fits in memory at a time
on the hardware this project targets, and on a 4 GB card not even one fits
whole. This loader therefore keeps exactly one llama_cpp.Llama instance
resident. When a query asks for a
different route, the previous instance is freed and the new one is loaded.

Model construction and completion unpacking are shared with
LlamaCppRunner through rag.generation.llama_backend. What stays here is the
part that is actually specific to routing: holding one model resident and
swapping it when the route changes.
"""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass, field

from rag.generation.llama_backend import chat_completion, load_llama, require_llama

from .registry import ModelSpec, get_model_spec
from .types import Route

log = logging.getLogger(__name__)


@dataclass
class _ResidentModel:
    """The single currently-loaded model, plus the spec it was loaded from."""

    spec: ModelSpec
    llm: object  # llama_cpp.Llama, kept as object to avoid a hard import dep


@dataclass
class StickyModelLoader:
    """
    Holds at most one llama.cpp model in memory.

    Usage:
        loader = StickyModelLoader()
        text = loader.chat(route="math", messages=[...], max_tokens=512)
    """

    fallback_to_default: bool = True
    _resident: _ResidentModel | None = field(default=None, init=False, repr=False)

    # ------------------------------------------------------------------
    # Loading / swapping
    # ------------------------------------------------------------------

    def ensure_loaded(self, route: Route) -> ModelSpec:
        """
        Make sure the model for `route` is the resident one. Returns the
        ModelSpec actually in use (which may have been demoted to the
        default route if `route`'s file is missing).
        """
        require_llama()

        target = get_model_spec(route, fallback_to_default=self.fallback_to_default)

        if (
            self._resident is not None
            and self._resident.spec.route == target.route
            and self._resident.spec.model_path == target.model_path
        ):
            return self._resident.spec  # already loaded

        self._evict()

        log.info(
            "Loading route=%s model=%s n_ctx=%d gpu_layers=%d",
            target.route,
            target.model_path,
            target.n_ctx,
            target.n_gpu_layers,
        )
        llm = load_llama(
            target.model_path,
            n_ctx=target.n_ctx,
            n_gpu_layers=target.n_gpu_layers,
            seed=target.seed,
            verbose=target.verbose,
        )
        self._resident = _ResidentModel(spec=target, llm=llm)
        return target

    def _evict(self) -> None:
        """Drop the resident model so the OS can reclaim its memory."""
        if self._resident is None:
            return
        log.info("Evicting route=%s model=%s", self._resident.spec.route, self._resident.spec.model_path)
        try:
            # llama_cpp.Llama frees the underlying context on __del__.
            self._resident.llm = None  # type: ignore[assignment]
        finally:
            self._resident = None
        gc.collect()

    def unload(self) -> None:
        """Public alias for _evict(). Useful in tests."""
        self._evict()

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def chat(
        self,
        *,
        route: Route,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.2,
        top_p: float = 0.95,
        repeat_penalty: float = 1.0,
        stop: list[str] | None = None,
    ) -> str:
        """
        Run a chat completion on the route's model. Loads/swaps as needed.
        Returns the assistant message text (stripped).
        """
        spec = self.ensure_loaded(route)
        if self._resident is None or self._resident.llm is None:
            raise RuntimeError(f"Model for route '{spec.route}' failed to load.")

        return chat_completion(
            self._resident.llm,
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
            stop=stop,
        )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def current_route(self) -> Route | None:
        return self._resident.spec.route if self._resident else None

    @property
    def current_spec(self) -> ModelSpec | None:
        return self._resident.spec if self._resident else None
