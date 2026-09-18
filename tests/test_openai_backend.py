"""The OpenAI backend, and the base-url case that makes it cover more."""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from rag.generation.backend import ChatBackend, get_backend
from rag.generation.providers.openai_backend import OpenAIBackend


def test_it_registers_itself():
    assert get_backend("openai").name == "openai"


def test_it_satisfies_the_protocol():
    assert isinstance(OpenAIBackend(api_key="k"), ChatBackend)


def test_a_missing_sdk_says_how_to_fix_it(monkeypatch):
    monkeypatch.setitem(sys.modules, "openai", None)
    with pytest.raises(RuntimeError, match="not installed"):
        OpenAIBackend(api_key="k")._client()


def test_a_missing_key_says_how_to_set_one():
    backend = OpenAIBackend(api_key=None, base_url=None)
    with patch.dict(sys.modules, {"openai": MagicMock()}):
        with pytest.raises(RuntimeError, match="rag config set OPENAI_API_KEY"):
            backend._client()


def test_a_local_server_does_not_need_a_key():
    # llama.cpp's server and Ollama ignore the key, but the SDK insists on
    # one. Making people invent a placeholder is a poor first experience.
    backend = OpenAIBackend(api_key=None, base_url="http://localhost:8080/v1")
    fake = MagicMock()
    with patch.dict(sys.modules, {"openai": fake}):
        backend._client()
    kwargs = fake.OpenAI.call_args.kwargs
    assert kwargs["base_url"] == "http://localhost:8080/v1"
    assert kwargs["api_key"]


def test_no_base_url_means_the_real_api():
    backend = OpenAIBackend(api_key="k")
    fake = MagicMock()
    with patch.dict(sys.modules, {"openai": fake}):
        backend._client()
    assert "base_url" not in fake.OpenAI.call_args.kwargs


# ---- message shape ---------------------------------------------------------


def test_the_messages_pass_through_unchanged():
    # This protocol takes the system prompt as a message, which is already
    # the shape the pipeline builds. Rewriting it would be a chance to break
    # the route prompts for nothing.
    messages = [
        {"role": "system", "content": "You answer from context."},
        {"role": "user", "content": "q"},
    ]
    params = OpenAIBackend(api_key="k")._params(messages, 256, 0.2, 0.95, None)
    assert params["messages"] is messages


def test_sampling_parameters_are_forwarded():
    # Unlike Anthropic, this API accepts them.
    params = OpenAIBackend(api_key="k")._params([], 256, 0.7, 0.5, ["</s>"])
    assert (params["max_tokens"], params["temperature"], params["top_p"]) == (256, 0.7, 0.5)
    assert params["stop"] == ["</s>"]


# ---- generation ------------------------------------------------------------


def _client_returning(text):
    message = SimpleNamespace(content=text)
    choice = SimpleNamespace(message=message)
    client = MagicMock()
    client.chat.completions.create.return_value = SimpleNamespace(choices=[choice])
    return client


def test_chat_returns_the_text():
    backend = OpenAIBackend(api_key="k")
    with patch.object(backend, "_client", return_value=_client_returning("  answer  ")):
        assert backend.chat([{"role": "user", "content": "q"}]) == "answer"


def test_an_unexpected_shape_gives_an_empty_answer_not_a_traceback():
    # An OpenAI-compatible server is not always faithful. The existing
    # unknown-answer fallback can handle an empty string; it cannot handle
    # an exception from the middle of the pipeline.
    backend = OpenAIBackend(api_key="k")
    client = MagicMock()
    client.chat.completions.create.return_value = SimpleNamespace(choices=[])
    with patch.object(backend, "_client", return_value=client):
        assert backend.chat([{"role": "user", "content": "q"}]) == ""


def _chunk(content):
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=content))])


def test_streaming_yields_the_pieces():
    backend = OpenAIBackend(api_key="k")
    client = MagicMock()
    client.chat.completions.create.return_value = iter(
        [_chunk("The "), _chunk("chain "), _chunk("rule.")]
    )
    with patch.object(backend, "_client", return_value=client):
        assert list(backend.chat_stream([])) == ["The ", "chain ", "rule."]


def test_streaming_asks_for_a_stream():
    backend = OpenAIBackend(api_key="k")
    client = MagicMock()
    client.chat.completions.create.return_value = iter([_chunk("x")])
    with patch.object(backend, "_client", return_value=client):
        list(backend.chat_stream([]))
    assert client.chat.completions.create.call_args.kwargs["stream"] is True


def test_empty_and_malformed_chunks_are_skipped():
    # Role-only first chunks, final chunks with only a finish reason, and
    # keepalives from proxies all arrive looking like this.
    backend = OpenAIBackend(api_key="k")
    client = MagicMock()
    client.chat.completions.create.return_value = iter(
        [_chunk(None), _chunk(""), SimpleNamespace(choices=[]), _chunk("real")]
    )
    with patch.object(backend, "_client", return_value=client):
        assert list(backend.chat_stream([])) == ["real"]
