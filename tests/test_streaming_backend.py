"""Streaming chat completions, and the chunk shapes llama.cpp actually emits."""

from unittest.mock import MagicMock

from rag.generation import llama_backend
from rag.generation.stream import (
    FinalEvent,
    ReplaceEvent,
    StageEvent,
    TokenEvent,
    collect_text,
)


def _streaming_llm(chunks):
    llm = MagicMock()
    llm.create_chat_completion.return_value = iter(chunks)
    return llm


def _delta(content):
    return {"choices": [{"delta": {"content": content}}]}


def test_the_pieces_arrive_in_order():
    llm = _streaming_llm([_delta("The "), _delta("chain "), _delta("rule.")])
    out = list(llama_backend.chat_completion_stream(llm, [{"role": "user", "content": "hi"}]))
    assert out == ["The ", "chain ", "rule."]


def test_stream_is_requested():
    llm = _streaming_llm([_delta("x")])
    list(llama_backend.chat_completion_stream(llm, [{"role": "user", "content": "hi"}]))
    assert llm.create_chat_completion.call_args.kwargs["stream"] is True


def test_the_role_only_first_chunk_is_skipped():
    # llama.cpp opens with a chunk carrying the role and no content.
    chunks = [{"choices": [{"delta": {"role": "assistant"}}]}, _delta("hello")]
    llm = _streaming_llm(chunks)
    assert list(llama_backend.chat_completion_stream(llm, [])) == ["hello"]


def test_the_final_chunk_with_no_content_is_skipped():
    chunks = [_delta("done"), {"choices": [{"delta": {}, "finish_reason": "stop"}]}]
    llm = _streaming_llm(chunks)
    assert list(llama_backend.chat_completion_stream(llm, [])) == ["done"]


def test_empty_pieces_do_not_become_blank_tokens():
    llm = _streaming_llm([_delta(""), _delta("real"), _delta(None)])
    assert list(llama_backend.chat_completion_stream(llm, [])) == ["real"]


def test_a_malformed_chunk_does_not_end_the_answer():
    # Skipping is better than stopping: the rest of the stream is usable.
    chunks = [_delta("before"), {"choices": []}, {}, _delta("after")]
    llm = _streaming_llm(chunks)
    assert list(llama_backend.chat_completion_stream(llm, [])) == ["before", "after"]


def test_generation_parameters_reach_the_model():
    llm = _streaming_llm([_delta("x")])
    list(llama_backend.chat_completion_stream(llm, [], max_tokens=64, temperature=0.9, top_p=0.5, stop=["</s>"]))
    kwargs = llm.create_chat_completion.call_args.kwargs
    assert (kwargs["max_tokens"], kwargs["temperature"], kwargs["top_p"]) == (64, 0.9, 0.5)
    assert kwargs["stop"] == ["</s>"]


# ---- the event contract ----------------------------------------------------


def test_tokens_concatenate():
    events = [StageEvent("generating"), TokenEvent("The "), TokenEvent("answer.")]
    assert collect_text(events) == "The answer."


def test_a_replace_discards_what_came_before():
    events = [TokenEvent("I don't know."), ReplaceEvent("unknown_fallback"), TokenEvent("Actually...")]
    assert collect_text(events) == "Actually..."


def test_the_final_event_wins_over_the_tokens():
    # Citation cleanup rewrites the answer after the tokens have stopped.
    result = MagicMock()
    result.answer = "Cleaned up [1]."
    events = [TokenEvent("raw [9] text"), FinalEvent(result)]
    assert collect_text(events) == "Cleaned up [1]."


def test_stages_have_something_to_show_a_user():
    for stage in ("retrieving", "routing", "loading_model", "generating"):
        assert StageEvent(stage).message
