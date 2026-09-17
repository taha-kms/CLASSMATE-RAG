"""The provider-agnostic generation seam."""

from unittest.mock import MagicMock, patch

import pytest

from rag.config import load_config
from rag.generation.backend import (
    ChatBackend,
    LlamaCppBackend,
    available_backends,
    get_backend,
    register_backend,
)


def test_llama_cpp_is_the_default(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    load_config(reload=True)
    assert get_backend().name == "llama_cpp"


def test_the_local_backend_satisfies_the_protocol():
    # runtime_checkable, so this is a real structural check rather than a
    # comment claiming the shapes match.
    assert isinstance(LlamaCppBackend(), ChatBackend)


def test_an_unknown_provider_names_the_ones_that_exist():
    with pytest.raises(ValueError, match="llama_cpp"):
        get_backend("telepathy")


def test_a_provider_can_register_itself():
    class Fake:
        name = "fake"

        def chat(self, messages, **kw):
            return "hello"

        def chat_stream(self, messages, **kw):
            yield "hel"
            yield "lo"

    register_backend("fake", Fake)
    try:
        assert "fake" in available_backends()
        backend = get_backend("fake")
        assert backend.chat([]) == "hello"
        assert "".join(backend.chat_stream([])) == "hello"
    finally:
        from rag.generation import backend as module

        module._BACKENDS.pop("fake", None)


def test_a_route_sends_generation_through_the_sticky_loader():
    backend = LlamaCppBackend()
    loader = MagicMock()
    loader.chat.return_value = "routed answer"

    with patch.object(backend, "_loader", return_value=loader):
        assert backend.chat([{"role": "user", "content": "hi"}], route="math") == "routed answer"

    assert loader.chat.call_args.kwargs["route"] == "math"


def test_no_route_uses_the_single_model_runner():
    backend = LlamaCppBackend()
    runner = MagicMock()
    runner.chat.return_value = "plain answer"

    with patch.object(backend, "_runner", return_value=runner):
        assert backend.chat([{"role": "user", "content": "hi"}]) == "plain answer"

    runner.chat.assert_called_once()


def test_streaming_follows_the_same_split():
    backend = LlamaCppBackend()
    loader, runner = MagicMock(), MagicMock()
    loader.chat_stream.return_value = iter(["a"])
    runner.chat_stream.return_value = iter(["b"])

    with patch.object(backend, "_loader", return_value=loader):
        assert list(backend.chat_stream([], route="code")) == ["a"]
    with patch.object(backend, "_runner", return_value=runner):
        assert list(backend.chat_stream([])) == ["b"]


def test_generation_parameters_are_passed_through():
    backend = LlamaCppBackend()
    runner = MagicMock()
    runner.chat.return_value = ""

    with patch.object(backend, "_runner", return_value=runner):
        backend.chat([], max_tokens=42, temperature=0.9, top_p=0.5, stop=["</s>"])

    kwargs = runner.chat.call_args.kwargs
    assert (kwargs["max_tokens"], kwargs["temperature"], kwargs["top_p"]) == (42, 0.9, 0.5)
    assert kwargs["stop"] == ["</s>"]


def test_the_pipeline_generates_only_through_the_backend():
    # The point of the seam is that adding a provider does not mean finding
    # every place that built a runner.
    import inspect

    from rag.pipeline import rag as pipeline

    source = inspect.getsource(pipeline.ask_question_stream)
    assert "get_backend()" in source
    assert "LlamaCppRunner()" not in source
    assert "_get_model_loader()" not in source
