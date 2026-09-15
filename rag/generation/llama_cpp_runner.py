"""
Wrapper for llama-cpp-python to load and run LLaMA models.

Exposes both:
- chat(messages, ...): OpenAI-style chat completion (used by the RAG pipeline)
- generate(prompt, ...): single-string completion (kept for backward compatibility)

The runner is no-arg constructible: when `model_path` is omitted it resolves the
configured model via `rag.config.load_config()` and downloads it on demand via
`rag.model_fetch.ensure_llama_model_available()`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Union

from rag.config import load_config
from rag.generation.llama_backend import chat_completion, load_llama, require_llama
from rag.model_fetch import ensure_llama_model_available


class LlamaCppRunner:
    """Thin wrapper around `llama_cpp.Llama` with chat + generate helpers."""

    def __init__(
        self,
        *,
        model_path: Optional[Union[str, Path]] = None,
        n_ctx: int = 4096,
        n_gpu_layers: Optional[int] = None,
        seed: int = 42,
        verbose: bool = False,
    ) -> None:
        if model_path is None:
            cfg = load_config()
            model_path = cfg.llm_model_path
            try:
                # If the file is missing, this downloads it from HF and returns the path.
                model_path = ensure_llama_model_available()
            except Exception:
                # Fall back to the configured path; existence is enforced below.
                pass

        require_llama()

        if n_gpu_layers is None:
            n_gpu_layers = int(os.getenv("LLAMA_GPU_LAYERS", "0"))

        self.model = load_llama(
            model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            seed=seed,
            verbose=verbose,
        )

    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        max_tokens: int = 768,
        temperature: float = 0.2,
        top_p: float = 0.95,
        repeat_penalty: float = 1.0,
        stop: Optional[List[str]] = None,
    ) -> str:
        """Run an OpenAI-style chat completion and return the assistant text."""
        return chat_completion(
            self.model,
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
            stop=stop,
        )

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 40,
        stop: Optional[List[str]] = None,
    ) -> str:
        """Single-string completion (kept for backward compatibility)."""
        res = self.model(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop=stop,
        )
        return res["choices"][0]["text"].strip()
