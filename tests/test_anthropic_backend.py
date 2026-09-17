"""The Anthropic backend."""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from rag.generation.backend import ChatBackend, get_backend
from rag.generation.providers.anthropic_backend import AnthropicBackend


def test_it_registers_itself():
    assert get_backend("anthropic").name == "anthropic"


def test_it_satisfies_the_protocol():
    assert isinstance(AnthropicBackend(api_key="k"), ChatBackend)


def test_a_missing_sdk_says_how_to_fix_it(monkeypatch):
    monkeypatch.setitem(sys.modules, "anthropic", None)
    backend = AnthropicBackend(api_key="k")
    with pytest.raises(RuntimeError, match="not installed"):
        backend._client()


def test_a_missing_key_says_how_to_set_one():
    backend = AnthropicBackend(api_key=None)
    fake_sdk = MagicMock()
    with patch.dict(sys.modules, {"anthropic": fake_sdk}):
        with pytest.raises(RuntimeError, match="rag config set ANTHROPIC_API_KEY"):
            backend._client()


# ---- the message shape -----------------------------------------------------


def test_the_system_prompt_is_lifted_out_of_the_messages():
    # Anthropic takes `system` as its own parameter. Passing our list
    # through unchanged would silently drop the route prompt.
    system, rest = AnthropicBackend._split(
        [
            {"role": "system", "content": "You answer from context."},
            {"role": "user", "content": "What is the chain rule?"},
        ]
    )
    assert system == "You answer from context."
    assert rest == [{"role": "user", "content": "What is the chain rule?"}]
    assert all(m["role"] != "system" for m in rest)


def test_several_system_messages_are_joined():
    system, rest = AnthropicBackend._split(
        [
            {"role": "system", "content": "first"},
            {"role": "system", "content": "second"},
            {"role": "user", "content": "q"},
        ]
    )
    assert "first" in system and "second" in system
    assert len(rest) == 1


def test_no_system_message_means_no_system_parameter():
    system, _ = AnthropicBackend._split([{"role": "user", "content": "q"}])
    assert system is None


def test_sampling_parameters_are_dropped_not_forwarded():
    # Current models reject temperature and top_p. Accepting and dropping
    # them keeps the pipeline provider-agnostic.
    backend = AnthropicBackend(api_key="k")
    params = backend._params([{"role": "user", "content": "q"}], 256, 0.7, 0.9, None)
    assert "temperature" not in params
    assert "top_p" not in params
    assert params["max_tokens"] == 256


def test_stop_sequences_use_the_provider_spelling():
    backend = AnthropicBackend(api_key="k")
    params = backend._params([{"role": "user", "content": "q"}], 256, 0.2, 0.95, ["</s>"])
    assert params["stop_sequences"] == ["</s>"]
    assert "stop" not in params


# ---- generation ------------------------------------------------------------


def _client_returning(text, stop_reason="end_turn"):
    message = SimpleNamespace(content=[SimpleNamespace(type="text", text=text)], stop_reason=stop_reason)
    stream = MagicMock()
    stream.__enter__ = MagicMock(return_value=stream)
    stream.__exit__ = MagicMock(return_value=False)
    stream.get_final_message.return_value = message
    client = MagicMock()
    client.messages.stream.return_value = stream
    return client, stream


def test_chat_returns_the_text():
    backend = AnthropicBackend(api_key="k")
    client, _ = _client_returning("  The chain rule.  ")
    with patch.object(backend, "_client", return_value=client):
        assert backend.chat([{"role": "user", "content": "q"}]) == "The chain rule."


def test_chat_streams_even_when_not_streaming_to_the_caller():
    # A long answer on a non-streamed request can exceed the SDK timeout.
    backend = AnthropicBackend(api_key="k")
    client, _ = _client_returning("x")
    with patch.object(backend, "_client", return_value=client):
        backend.chat([{"role": "user", "content": "q"}])
    client.messages.stream.assert_called_once()


def test_a_refusal_comes_back_as_an_empty_answer():
    # stop_reason "refusal" is an HTTP 200 with no usable content, not an
    # exception. Reading .content blindly would produce nonsense.
    backend = AnthropicBackend(api_key="k")
    client, _ = _client_returning("", stop_reason="refusal")
    with patch.object(backend, "_client", return_value=client):
        assert backend.chat([{"role": "user", "content": "q"}]) == ""


def test_streaming_yields_the_text_stream():
    backend = AnthropicBackend(api_key="k")
    client, stream = _client_returning("unused")
    stream.text_stream = iter(["The ", "chain ", "rule."])
    with patch.object(backend, "_client", return_value=client):
        assert list(backend.chat_stream([{"role": "user", "content": "q"}])) == [
            "The ",
            "chain ",
            "rule.",
        ]
