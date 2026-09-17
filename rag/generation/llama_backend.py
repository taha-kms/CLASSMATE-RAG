"""Shared llama.cpp plumbing.

Two places load a model and run a chat completion: LlamaCppRunner, used on
the non-routed path, and StickyModelLoader, which keeps exactly one model
resident and swaps it when the route changes. Both used to build their own
`llama_cpp.Llama` and unpack `create_chat_completion` themselves, so a
change to generation defaults had to be made twice and it was easy to
remember only one.

The two classes keep their distinct jobs. What lives here is the part that
was genuinely the same: constructing the model, and turning a completion
into a string.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from llama_cpp import Llama
except ImportError:  # pragma: no cover - llama.cpp is an optional compile
    Llama = None  # type: ignore[assignment]


def require_llama() -> None:
    """Fail with something readable when the optional runtime is absent."""
    if Llama is None:
        raise RuntimeError(
            "llama-cpp-python is not installed, so no local model can be run. "
            "Install it with `pip install llama-cpp-python`."
        )


def load_llama(
    model_path: str | Path,
    *,
    n_ctx: int = 4096,
    n_gpu_layers: int = 0,
    seed: int = 42,
    verbose: bool = False,
) -> Any:
    """Construct a llama.cpp model, checking the file exists first."""
    require_llama()

    resolved = Path(model_path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Model file not found: {resolved}")

    return Llama(
        model_path=str(resolved),
        n_ctx=int(n_ctx),
        n_gpu_layers=int(n_gpu_layers),
        seed=int(seed),
        verbose=bool(verbose),
    )


def chat_completion(
    llm: Any,
    messages: list[dict[str, str]],
    *,
    max_tokens: int = 768,
    temperature: float = 0.2,
    top_p: float = 0.95,
    repeat_penalty: float = 1.0,
    stop: list[str] | None = None,
) -> str:
    """Run an OpenAI-style chat completion and return the assistant text."""
    result = llm.create_chat_completion(
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
        # A backend that returns an unexpected shape should surface as an
        # empty answer rather than a traceback from deep in the pipeline.
        content = ""
    return (content or "").strip()
