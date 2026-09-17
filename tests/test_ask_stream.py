"""ask_question_stream, and that ask_question is genuinely a drain over it.

The two must not become separate implementations: the whole point of the
drain is that there is one code path and the non-streaming answer cannot
drift from the streamed one.
"""

import inspect
from unittest.mock import MagicMock

from rag.generation.stream import (
    FinalEvent,
    ReplaceEvent,
    StageEvent,
    TokenEvent,
    collect_text,
)
from rag.pipeline import rag as pipeline


def test_ask_question_is_a_drain_over_the_stream():
    # If this grows its own retrieval or generation, the two paths can
    # disagree and only one of them is tested.
    source = inspect.getsource(pipeline.ask_question)
    assert "ask_question_stream" in source
    assert "retriever" not in source
    assert "LlamaCppRunner" not in source


def test_the_stream_is_a_generator():
    assert inspect.isgeneratorfunction(pipeline.ask_question_stream)


def test_the_drain_returns_the_final_result():
    expected = MagicMock()

    def fake_stream(**_kwargs):
        yield StageEvent("retrieving")
        yield TokenEvent("partial")
        yield FinalEvent(expected)

    original = pipeline.ask_question_stream
    pipeline.ask_question_stream = fake_stream
    try:
        got = pipeline.ask_question(question="q", filters=MagicMock())
    finally:
        pipeline.ask_question_stream = original

    assert got is expected


def test_the_drain_complains_if_the_stream_ends_without_a_result():
    def truncated(**_kwargs):
        yield TokenEvent("half an answer")

    original = pipeline.ask_question_stream
    pipeline.ask_question_stream = truncated
    try:
        try:
            pipeline.ask_question(question="q", filters=MagicMock())
        except RuntimeError as e:
            assert "without a result" in str(e)
        else:  # pragma: no cover
            raise AssertionError("a truncated stream should not pass silently")
    finally:
        pipeline.ask_question_stream = original


def test_a_replace_means_the_earlier_tokens_are_withdrawn():
    # The unknown-answer fallback re-answers without context. What was shown
    # was a complete answer, not a partial one.
    events = [
        StageEvent("generating"),
        TokenEvent("I don't know."),
        ReplaceEvent("unknown_fallback"),
        TokenEvent("The chain rule is..."),
    ]
    assert collect_text(events) == "The chain rule is..."


def test_both_generation_paths_stream():
    # Routed and legacy both have to yield tokens, or turning routing on
    # silently turns streaming off.
    source = inspect.getsource(pipeline.ask_question_stream)

    # Four: each path generates once and re-answers once on the fallback.
    assert source.count("chat_stream(") >= 4, "both paths should stream, including the fallback"

    # The answer itself must never come from a blocking call. Translation
    # still uses chat(), which is correct: it needs the whole answer and is
    # not shown as it is produced.
    assert "answer = loader.chat(" not in source
    assert "answer = runner.chat(" not in source
