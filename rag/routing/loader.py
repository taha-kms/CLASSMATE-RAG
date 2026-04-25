"""
Sticky single-model loader.

On 8 GB VRAM only one ~7B Q4 model fits at a time, so this loader keeps
exactly one llama_cpp.Llama instance resident. When a query asks for a
different route, the previous instance is freed and the new one is loaded.

Why we don't reuse rag.generation.LlamaCppRunner:
- Its __init__ requires a model_path and exposes generate(), not chat().
- The pipeline already calls a chat()-style API. Building a clean wrapper
  here lets the routing path work without touching the existing runner.
"""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

try:
    from llama_cpp import Llama
except Exception:  # pragma: no cover - llama_cpp may not be installed in test envs
    Llama = None  # type: ignore[assignment]

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
    _resident: Optional[_ResidentModel] = field(default=None, init=False, repr=False)

    # ------------------------------------------------------------------
    # Loading / swapping
    # ------------------------------------------------------------------

    def ensure_loaded(self, route: Route) -> ModelSpec:
        """
        Make sure the model for `route` is the resident one. Returns the
        ModelSpec actually in use (which may have been demoted to the
        default route if `route`'s file is missing).
        """
        if Llama is None:
            raise RuntimeError(
                "llama-cpp-python is not installed; cannot load any route model."
            )

        target = get_model_spec(route, fallback_to_default=self.fallback_to_default)

        if self._resident is not None and self._resident.spec.route == target.route \
                and self._resident.spec.model_path == target.model_path:
            return self._resident.spec  # already loaded

        self._evict()

        log.info(
            "Loading route=%s model=%s n_ctx=%d gpu_layers=%d",
            target.route, target.model_path, target.n_ctx, target.n_gpu_layers,
        )
        llm = Llama(
            model_path=str(target.model_path),
            n_ctx=int(target.n_ctx),
            n_gpu_layers=int(target.n_gpu_layers),
            seed=int(target.seed),
            verbose=bool(target.verbose),
        )
        self._resident = _ResidentModel(spec=target, llm=llm)
        return target

    def _evict(self) -> None:
        """Drop the resident model so the OS can reclaim its memory."""
        if self._resident is None:
            return
        log.info("Evicting route=%s model=%s",
                 self._resident.spec.route, self._resident.spec.model_path)
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
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.2,
        top_p: float = 0.95,
        repeat_penalty: float = 1.0,
        stop: Optional[List[str]] = None,
    ) -> str:
        """
        Run a chat completion on the route's model. Loads/swaps as needed.
        Returns the assistant message text (stripped).
        """
        spec = self.ensure_loaded(route)
        if self._resident is None or self._resident.llm is None:
            raise RuntimeError(f"Model for route '{spec.route}' failed to load.")

        llm = self._resident.llm
        # llama_cpp.Llama.create_chat_completion takes OpenAI-style messages.
        result = llm.create_chat_completion(  # type: ignore[attr-defined]
            messages=messages,
            max_tokens=int(max_tokens),
            temperature=float(temperature),
            top_p=float(top_p),
            repeat_penalty=float(repeat_penalty),
            stop=stop,
        )
        try:
            content = result["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            content = ""
        return (content or "").strip()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def current_route(self) -> Optional[Route]:
        return self._resident.spec.route if self._resident else None

    @property
    def current_spec(self) -> Optional[ModelSpec]:
        return self._resident.spec if self._resident else None
