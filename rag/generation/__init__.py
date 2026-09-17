"""
Expose generation-related utilities.

Includes:
- LlamaCppRunner: wrapper to run LLaMA models with llama-cpp-python
- build_grounded_messages / build_general_messages: helpers to build prompts
- format_context_blocks: format retrieved text for the model
- enforce_citations: ensure citations are present in answers
"""

from typing import TYPE_CHECKING

from rag._lazy import lazy_exports

__all__ = [
    "LlamaCppRunner",
    "build_grounded_messages",
    "build_general_messages",
    "format_context_blocks",
    "enforce_citations",
]

# Only the runner needs llama_cpp. prompting and post are pure Python, and
# keeping them reachable without it is what lets them be tested.
__getattr__, __dir__ = lazy_exports(
    __name__,
    {
        "LlamaCppRunner": ".llama_cpp_runner",
        "build_grounded_messages": ".prompting",
        "build_general_messages": ".prompting",
        "format_context_blocks": ".prompting",
        "enforce_citations": ".post",
    },
)

if TYPE_CHECKING:  # pragma: no cover
    from .llama_cpp_runner import LlamaCppRunner
    from .post import enforce_citations
    from .prompting import build_general_messages, build_grounded_messages, format_context_blocks
