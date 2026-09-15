"""LOG_LEVEL actually configures logging.

Config carried a log_level for a long time while nothing called
basicConfig, so every record the project emitted was discarded.
"""

import logging

import pytest

from rag.config import _NOISY_LOGGERS, _quiet_noisy_libraries, configure_logging, load_config


@pytest.fixture(autouse=True)
def restore_logging():
    root = logging.getLogger()
    before = root.level
    noisy_before = {name: logging.getLogger(name).level for name in _NOISY_LOGGERS}
    yield
    root.setLevel(before)
    for name, level in noisy_before.items():
        logging.getLogger(name).setLevel(level)


def test_log_level_is_applied_to_the_root_logger(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    load_config(reload=True)
    configure_logging()
    assert logging.getLogger().level == logging.WARNING


def test_an_unrecognised_level_falls_back_to_info(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "NOT_A_LEVEL")
    load_config(reload=True)
    configure_logging()
    assert logging.getLogger().level == logging.INFO


def test_an_explicit_argument_beats_the_environment(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    load_config(reload=True)
    configure_logging("ERROR")
    assert logging.getLogger().level == logging.ERROR


def test_chatty_libraries_are_held_at_warning_by_default():
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.NOTSET)

    _quiet_noisy_libraries(logging.INFO)

    # httpx logs a line per request; at INFO it would bury our own messages.
    assert logging.getLogger("httpx").level == logging.WARNING
    assert logging.getLogger("chromadb").level == logging.WARNING


def test_debug_leaves_the_chatty_libraries_alone():
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.NOTSET)

    _quiet_noisy_libraries(logging.DEBUG)

    # Asking for DEBUG means you want the HTTP traffic too.
    assert logging.getLogger("httpx").level == logging.NOTSET
