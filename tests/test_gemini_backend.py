"""The Gemini backend, whose message format differs most from the others."""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from rag.generation.backend import ChatBackend, get_backend
from rag.generation.providers.gemini_backend import GeminiBackend


def test_it_registers_itself():
    assert get_backend("gemini").name == "gemini"


def test_it_satisfies_the_protocol():
    assert isinstance(GeminiBackend(api_key="k"), ChatBackend)


def test_a_missing_sdk_says_how_to_fix_it(monkeypatch):
    monkeypatch.setitem(sys.modules, "google.genai", None)
    monkeypatch.setitem(sys.modules, "google", SimpleNamespace())
    with pytest.raises(RuntimeError, match="not installed"):
        GeminiBackend(api_key="k")._client()


def test_a_missing_key_points_at_where_keys_come_from():
    backend = GeminiBackend(api_key=None)
    fake = SimpleNamespace(genai=MagicMock())
    with patch.dict(sys.modules, {"google": fake, "google.genai": fake.genai}):
        with pytest.raises(RuntimeError, match="aistudio.google.com"):
            backend._client()


# ---- the message adapter, which is most of this backend --------------------


def test_the_system_prompt_becomes_a_separate_instruction():
    system, contents = GeminiBackend._adapt(
        [
            {"role": "system", "content": "Answer from context."},
            {"role": "user", "content": "What is the chain rule?"},
        ]
    )
    assert system == "Answer from context."
    assert all(c["role"] != "system" for c in contents)


def test_the_assistant_role_is_renamed_to_model():
    # Gemini calls it "model". Sending "assistant" is rejected or ignored
    # depending on the endpoint, neither of which is obvious from the answer.
    _, contents = GeminiBackend._adapt(
        [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
        ]
    )
    assert [c["role"] for c in contents] == ["user", "model"]


def test_content_becomes_a_list_of_parts():
    _, contents = GeminiBackend._adapt([{"role": "user", "content": "hello"}])
    assert contents[0]["parts"] == [{"text": "hello"}]


def test_several_system_messages_are_joined():
    system, _ = GeminiBackend._adapt(
        [
            {"role": "system", "content": "first"},
            {"role": "system", "content": "second"},
            {"role": "user", "content": "q"},
        ]
    )
    assert "first" in system and "second" in system


def test_no_system_message_means_no_instruction():
    system, contents = GeminiBackend._adapt([{"role": "user", "content": "q"}])
    assert system is None
    assert len(contents) == 1


def test_an_empty_system_message_is_not_carried_through():
    system, _ = GeminiBackend._adapt([{"role": "system", "content": ""}, {"role": "user", "content": "q"}])
    assert system is None


# ---- generation ------------------------------------------------------------


def _backend_with_client(client):
    backend = GeminiBackend(api_key="k")
    backend._client = MagicMock(return_value=client)
    backend._config = MagicMock(return_value=object())
    return backend


def test_chat_returns_the_text():
    client = MagicMock()
    client.models.generate_content.return_value = SimpleNamespace(text="  answer  ")
    assert _backend_with_client(client).chat([{"role": "user", "content": "q"}]) == "answer"


def test_a_response_with_no_text_is_an_empty_answer():
    client = MagicMock()
    client.models.generate_content.return_value = SimpleNamespace(text=None)
    assert _backend_with_client(client).chat([{"role": "user", "content": "q"}]) == ""


def test_streaming_yields_the_chunks():
    client = MagicMock()
    client.models.generate_content_stream.return_value = iter(
        [SimpleNamespace(text="The "), SimpleNamespace(text="rule.")]
    )
    assert list(_backend_with_client(client).chat_stream([])) == ["The ", "rule."]


def test_empty_stream_chunks_are_skipped():
    client = MagicMock()
    client.models.generate_content_stream.return_value = iter(
        [SimpleNamespace(text=None), SimpleNamespace(text=""), SimpleNamespace(text="real")]
    )
    assert list(_backend_with_client(client).chat_stream([])) == ["real"]


# ---- rate limits -----------------------------------------------------------


@pytest.mark.parametrize("message", ["429 Too Many Requests", "RESOURCE_EXHAUSTED", "Quota exceeded for model"])
def test_a_rate_limit_reads_as_a_limit_not_a_crash(message):
    # The free tier is limited aggressively, which is fine, but it should
    # not look like the tool is broken.
    client = MagicMock()
    client.models.generate_content.side_effect = RuntimeError(message)
    backend = _backend_with_client(client)

    with pytest.raises(RuntimeError, match="rate limit") as excinfo:
        backend.chat([{"role": "user", "content": "q"}])

    assert "llama_cpp" in str(excinfo.value), "always offer the way back to local"


def test_other_failures_keep_their_detail():
    client = MagicMock()
    client.models.generate_content.side_effect = RuntimeError("the network is on fire")
    with pytest.raises(RuntimeError, match="the network is on fire"):
        _backend_with_client(client).chat([{"role": "user", "content": "q"}])
