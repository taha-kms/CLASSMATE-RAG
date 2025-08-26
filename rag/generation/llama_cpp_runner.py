"""
Llama 3.1-8B runner using llama-cpp-python.

- Loads GGUF via model path from env-config (rag.config / classmate.config).
- Ensures the model file exists locally; if missing, triggers auto-download via classmate.model_fetch.
- Provides a simple chat() interface returning the assistant's text.

Notes:
- Defaults are conservative for CPU+low-VRAM laptops. Adjust n_gpu_layers to offload more layers to GPU if available.
- Streaming is not implemented here; we return the full completion for simplicity in the CLI.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from llama_cpp import Llama

from rag.config import load_config
from classmate.model_fetch import ensure_llama_model_available


@dataclass
class LlamaCppRunner:
    model_path: Optional[str] = None
    n_ctx: int = 4096
    n_threads: Optional[int] = None      # default: os.cpu_count()
    n_gpu_layers: int = 0                # 0 = CPU only. Increase to offload to GPU if available.
    seed: int = 42
    temperature: float = 0.2
    top_p: float = 0.95
    repeat_penalty: float = 1.1
    max_tokens: int = 512

    # internal
    _llm: Optional[Llama] = None
    _loaded_path: Optional[Path] = None

    def _resolve_model_path(self) -> Path:
        if self.model_path:
            return Path(self.model_path).expanduser().resolve()

        cfg = load_config()
        # Validate/get path; this does not require file to exist yet
        target = cfg.validate_for_llm()
        # Ensure model is available (auto-download if needed)
        path = ensure_llama_model_available()
        # If ensure_llama_model_available returned a different file than env points to, use it.
        return Path(path)

    def _ensure_llm(self) -> Llama:
        if self._llm is not None:
            return self._llm

        path = self._resolve_model_path()
        self._loaded_path = path

        n_threads = self.n_threads or os.cpu_count() or 4

        # Let llama.cpp auto-detect chat format from GGUF metadata (Llama-3/3.1)
        self._llm = Llama(
            model_path=str(path),
            n_ctx=self.n_ctx,
            n_threads=n_threads,
            n_gpu_layers=self.n_gpu_layers,  # set >0 to offload layers to GPU
            seed=self.seed,
            logits_all=False,
            vocab_only=False,
            use_mlock=False,      # set True if you want to lock pages in RAM (requires privileges)
            use_mmap=True,        # memory-map the model (recommended)
            verbose=False,
        )
        return self._llm

    def chat(
        self,
        messages: List[dict],
        *,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        repeat_penalty: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[list[str]] = None,
    ) -> str:
        """
        Run a chat completion. 'messages' must be a list of dicts with 'role' and 'content',
        e.g., [{"role":"system","content":"..."},{"role":"user","content":"..."}].
        Returns the assistant text.
        """
        llm = self._ensure_llm()
        t = self.temperature if temperature is None else float(temperature)
        p = self.top_p if top_p is None else float(top_p)
        rp = self.repeat_penalty if repeat_penalty is None else float(repeat_penalty)
        mt = self.max_tokens if max_tokens is None else int(max_tokens)

        # llama.cpp uses create_chat_completion for structured chat with templates (e.g., Llama-3)
        out = llm.create_chat_completion(
            messages=messages,
            temperature=t,
            top_p=p,
            repeat_penalty=rp,
            max_tokens=mt,
            stop=stop or [],
        )
        # Extract text
        try:
            return out["choices"][0]["message"]["content"].strip()
        except Exception:
            return ""
