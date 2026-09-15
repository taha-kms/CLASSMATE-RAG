"""Translate-on-miss.

The routed path returned before ever reaching the translation step, so
TRANSLATE_ON_MISS silently did nothing once ENABLE_ROUTING was on.
"""

import pytest

from rag.pipeline.rag import _needs_translation, _translate_text


def test_an_answer_already_in_the_target_language_is_left_alone():
    assert _needs_translation("The chain rule differentiates a composition.", "en") is False


def test_an_answer_in_the_other_language_is_flagged():
    assert _needs_translation("La regola della catena deriva una composizione di funzioni.", "en") is True


def test_an_empty_answer_is_never_translated():
    assert _needs_translation("", "en") is False
    assert _needs_translation("   ", "en") is False


def test_translation_goes_through_the_callers_chat_function():
    seen = []

    def fake_chat(messages):
        seen.append(messages)
        return "Translated."

    out = _translate_text("Testo italiano.", "en", chat=fake_chat)

    assert out == "Translated."
    # One call, so the routed path cannot end up loading a second model.
    assert len(seen) == 1


def test_the_prompt_asks_for_citations_to_be_preserved():
    captured = {}

    def fake_chat(messages):
        captured["system"] = messages[0]["content"]
        return "ok"

    _translate_text("Some text [1].", "en", chat=fake_chat)
    assert "[1]" in captured["system"]


@pytest.mark.parametrize("target,expected_marker", [("it", "italiano"), ("en", "English")])
def test_the_instruction_is_written_for_the_target_language(target, expected_marker):
    captured = {}

    def fake_chat(messages):
        captured["system"] = messages[0]["content"]
        return "ok"

    _translate_text("text", target, chat=fake_chat)
    assert expected_marker in captured["system"]


def test_an_empty_reply_falls_back_to_the_original():
    out = _translate_text("original text", "en", chat=lambda msgs: "   ")
    assert out == "original text"
