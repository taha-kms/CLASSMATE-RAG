"""
Wrapper for llama-cpp-python to load and run LLaMA models.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from llama_cpp import Llama

from rag.config import load_config
from rag.model_fetch import ensure_llama_model_available


@dataclass
class LlamaCppRunner:
    """
    Simple wrapper around llama_cpp.Llama.
    Provides:
    - model loading
    - basic text generation
    """

    def __init__(
        self,
        model_path: str,
        *,
        n_ctx: int = 2048,
        n_gpu_layers: Optional[int] = None,
        seed: int = 42,
        verbose: bool = False,
    ) -> None:
        """
        Load a LLaMA model.

        Args:
            model_path: path to the model file
            n_ctx: context window size
            n_gpu_layers: number of layers on GPU (None → auto/env)
            seed: random seed
            verbose: show llama-cpp logs
        """
        p = Path(model_path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Model file not found: {p}")

        gpu_layers = n_gpu_layers
        if gpu_layers is None:
            gpu_layers = int(os.getenv("LLAMA_GPU_LAYERS", "0"))

        self.model = Llama(
            model_path=str(p),
            n_ctx=n_ctx,
            n_gpu_layers=gpu_layers,
            seed=seed,
            verbose=verbose,
        )
        return self.model

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
        """
        Generate text.

        Args:
            prompt: input text
            max_tokens: max tokens to generate
            temperature: sampling temperature
            top_p: nucleus sampling
            top_k: top-k filtering
            stop: optional stop tokens

        Returns:
            Generated text.
        """
        res = self.model(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop=stop,
        )
        return res["choices"][0]["text"].strip()
