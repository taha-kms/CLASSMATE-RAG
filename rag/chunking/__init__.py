"""
Expose chunking utilities for splitting text into smaller pieces.

Includes:
- RagChunk: data class for a chunk of text
- sentence_split: split text into sentences
- chunk_text: split text into overlapping chunks
- chunk_pages: split multiple pages into chunks
"""

from .chunker import (
    RagChunk,
    sentence_split,
    chunk_text,
    chunk_pages,
)

__all__ = [
    "RagChunk",
    "sentence_split",
    "chunk_text",
    "chunk_pages",
]
