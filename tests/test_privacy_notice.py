"""Saying what leaves the machine."""

import pytest

from rag.generation.privacy import data_sent_notice, is_local


def test_the_local_backend_is_local():
    assert is_local("llama_cpp")
    assert data_sent_notice("llama_cpp") is None, "nothing leaves, so there is nothing to say"


@pytest.mark.parametrize("backend", ["anthropic", "openai", "gemini"])
def test_a_hosted_backend_is_not_local(backend):
    assert not is_local(backend)


@pytest.mark.parametrize("backend", ["anthropic", "openai", "gemini"])
def test_the_notice_says_what_is_sent_and_what_is_not(backend):
    notice = data_sent_notice(backend)
    assert notice

    # The specific fear is that the whole corpus is uploaded. Say otherwise.
    assert "not uploaded" in notice
    # And be honest about what does go.
    assert "question" in notice and "passages" in notice
    # Cost is part of the trade.
    assert "billed" in notice
    # Always offer the way back.
    assert "llama_cpp" in notice
    assert backend in notice


def test_an_unknown_backend_is_treated_as_hosted():
    # Safer default: a provider nobody has classified is assumed to be
    # somebody else's computer.
    assert data_sent_notice("some-new-provider") is not None


def test_a_local_backend_announces_nothing(capsys, monkeypatch):
    from rag.generation import backend as module

    monkeypatch.setattr(module, "_ANNOUNCED", set())
    module.get_backend("llama_cpp")
    assert capsys.readouterr().err == ""


def test_a_hosted_backend_announces_once(capsys, monkeypatch):
    from rag.generation import backend as module

    monkeypatch.setattr(module, "_ANNOUNCED", set())
    module.register_backend("pretend_hosted", lambda: object())
    try:
        module.get_backend("pretend_hosted")
        first = capsys.readouterr().err
        module.get_backend("pretend_hosted")
        second = capsys.readouterr().err
    finally:
        module._BACKENDS.pop("pretend_hosted", None)

    assert "not uploaded" in first
    # Once per process, not before every question.
    assert second == ""


def test_the_notice_goes_to_stderr_so_json_stays_parseable(capsys, monkeypatch):
    from rag.generation import backend as module

    monkeypatch.setattr(module, "_ANNOUNCED", set())
    module.register_backend("pretend_hosted2", lambda: object())
    try:
        module.get_backend("pretend_hosted2")
    finally:
        module._BACKENDS.pop("pretend_hosted2", None)

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err
