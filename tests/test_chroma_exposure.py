"""Chroma is only tolerable on loopback, and the app says so when it is not.

chromadb 1.x has unpatched vulnerabilities that need access to its HTTP port
(#130). Loopback is the whole mitigation, so leaving it must not be silent.
"""

import logging

import pytest

from rag.retrieval import vector_chroma
from rag.retrieval.vector_chroma import chroma_exposure_warnings


@pytest.mark.parametrize("bind", ["", "127.0.0.1", "localhost", "::1"])
@pytest.mark.parametrize("url", ["", "http://localhost:8000", "http://127.0.0.1:8000", "http://[::1]:8000"])
def test_loopback_is_not_a_problem(bind, url):
    assert chroma_exposure_warnings(url, bind) == []


@pytest.mark.parametrize("bind", ["0.0.0.0", "::", "192.168.1.20"])
def test_publishing_beyond_loopback_is_reported(bind):
    (message,) = chroma_exposure_warnings("http://localhost:8000", bind)
    assert message.startswith(f"CHROMA_BIND_HOST={bind} publishes the Chroma port beyond this machine.")


def test_a_remote_server_is_reported():
    (message,) = chroma_exposure_warnings("http://chroma.example.org:8000", "")
    assert message.startswith("CHROMA_HTTP_URL points at chroma.example.org, not this machine.")


def test_both_problems_are_both_reported():
    assert len(chroma_exposure_warnings("http://10.0.0.5:8000", "0.0.0.0")) == 2


def test_an_unparseable_url_is_left_to_the_caller():
    assert chroma_exposure_warnings("http://[bad", "") == []


def test_each_exposure_is_logged_once_not_per_query(monkeypatch, caplog):
    monkeypatch.setattr(vector_chroma, "_already_warned", set())
    monkeypatch.setenv("CHROMA_BIND_HOST", "0.0.0.0")
    with caplog.at_level(logging.WARNING, logger=vector_chroma.log.name):
        for _ in range(5):
            vector_chroma._warn_if_chroma_exposed("http://localhost:8000")
    assert len([r for r in caplog.records if "CHROMA_BIND_HOST" in r.getMessage()]) == 1
