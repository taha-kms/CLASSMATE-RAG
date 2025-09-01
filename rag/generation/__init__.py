"""
Expose generation-related utilities.

Includes:
- LlamaCppRunner: wrapper to run LLaMA models with llama-cpp-python
- build_grounded_messages / build_general_messages: helpers to build prompts
- format_context_blocks: format retrieved text for the model
- enforce_citations: ensure citations are present in answers
"""

from .llama_cpp_runner import LlamaCppRunner
from .prompting import (
    build_grounded_messages,
    build_general_messages,
    format_context_blocks,
)
from .post import (
    enforce_citations,
)

__all__ = [
    "LlamaCppRunner",
    "build_grounded_messages",
    "build_general_messages",
    "format_context_blocks",
    "enforce_citations",
]
