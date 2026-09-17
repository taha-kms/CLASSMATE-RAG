"""
Event types for streaming a generated answer.

Generation blocks for tens of seconds on a laptop, and a UI with nothing to
show during that reads as a hang. Streaming turns one long wait into
something that visibly progresses.

The contract is a sequence of events rather than a stream of strings,
because tokens are not the only thing worth reporting. Retrieval, routing
and loading a multi-gigabyte model all happen before the first token and
each can take longer than the generation itself.

Consumers should treat FinalEvent as authoritative. Post-processing, the
citation cleanup and the no-context fallback, needs the whole answer and
therefore cannot run until the tokens have stopped. With STRICT_CITATIONS
off, which is the default, the final text matches what was streamed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:  # pragma: no cover - annotation only
    from rag.pipeline.rag import AskResult

#: Work that happens before, or instead of, token generation.
Stage = Literal["retrieving", "routing", "loading_model", "generating"]


@dataclass(frozen=True)
class StageEvent:
    """Progress through the pipeline. Emitted before the stage runs."""

    stage: Stage

    @property
    def message(self) -> str:
        return {
            "retrieving": "Searching your documents",
            "routing": "Choosing a model",
            "loading_model": "Loading the model",
            "generating": "Writing the answer",
        }[self.stage]


@dataclass(frozen=True)
class TokenEvent:
    """A piece of the answer. Append it to whatever is on screen."""

    text: str


@dataclass(frozen=True)
class ReplaceEvent:
    """
    Discard everything shown so far; what follows supersedes it.

    Emitted when the model answers "I don't know" against the retrieved
    context and the pipeline retries without it. The first answer was real
    output, not a partial one, so it has to be withdrawn explicitly rather
    than appended to.
    """

    reason: str


@dataclass(frozen=True)
class FinalEvent:
    """
    The authoritative result, after post-processing.

    Always the last event. Its answer may differ from the concatenated
    tokens when citation enforcement or translation rewrote it.
    """

    result: AskResult


StreamEvent = StageEvent | TokenEvent | ReplaceEvent | FinalEvent


def collect_text(events) -> str:
    """
    Fold an event stream into the text a viewer would have ended up with.

    Used by the non-streaming path and by tests: it is the definition of
    what the events mean, in one place, rather than reimplemented per
    consumer.
    """
    buffer: list[str] = []
    for event in events:
        if isinstance(event, TokenEvent):
            buffer.append(event.text)
        elif isinstance(event, ReplaceEvent):
            buffer.clear()
        elif isinstance(event, FinalEvent):
            return event.result.answer
    return "".join(buffer)
