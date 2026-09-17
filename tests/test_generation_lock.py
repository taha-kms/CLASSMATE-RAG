"""The lock that stops two callers racing the resident model.

StickyModelLoader frees the current llama_cpp.Llama before loading the
next, so overlapping generations can have one thread reading a context
another has released. That is a segfault, not an exception.
"""

import threading
import time
from unittest.mock import MagicMock

import pytest

from rag.routing.loader import StickyModelLoader, generation_lock


def _loader_with(chunks, route="default"):
    loader = StickyModelLoader()
    llm = MagicMock()
    llm.create_chat_completion.return_value = iter(chunks)
    resident = MagicMock()
    resident.llm = llm
    resident.spec.route = route
    loader._resident = resident
    loader.ensure_loaded = MagicMock(return_value=resident.spec)
    return loader


def _delta(text):
    return {"choices": [{"delta": {"content": text}}]}


def test_the_lock_is_held_before_the_first_next():
    # A generator body does not run until the first next(), so taking the
    # lock inside it would let two callers both hold un-started generators.
    loader = _loader_with([_delta("a")])
    stream = loader.chat_stream(route="default", messages=[])

    assert generation_lock().locked(), "lock must be held from the moment the stream exists"
    list(stream)


def test_the_lock_is_released_when_the_stream_is_exhausted():
    loader = _loader_with([_delta("a"), _delta("b")])
    assert list(loader.chat_stream(route="default", messages=[])) == ["a", "b"]
    assert not generation_lock().locked()


def test_the_lock_is_released_when_the_stream_is_abandoned():
    # A disconnected client leaves the generator unconsumed. Closing it must
    # give the lock back, or the next question blocks forever.
    loader = _loader_with([_delta("a"), _delta("b"), _delta("c")])
    stream = loader.chat_stream(route="default", messages=[])
    next(stream)
    stream.close()
    assert not generation_lock().locked()


def test_the_lock_is_released_when_loading_fails():
    loader = StickyModelLoader()
    loader.ensure_loaded = MagicMock(side_effect=RuntimeError("no model"))

    with pytest.raises(RuntimeError):
        loader.chat_stream(route="default", messages=[])

    assert not generation_lock().locked(), "a failure before the generator must not strand the lock"


def test_a_second_caller_waits_rather_than_racing():
    loader = _loader_with([_delta("a")])
    first = loader.chat_stream(route="default", messages=[])

    started = threading.Event()
    acquired = threading.Event()

    def second():
        started.set()
        with generation_lock():
            acquired.set()

    t = threading.Thread(target=second, daemon=True)
    t.start()
    started.wait(timeout=2)
    time.sleep(0.05)

    assert not acquired.is_set(), "the second caller got in while the first still held the model"

    list(first)
    t.join(timeout=2)
    assert acquired.is_set(), "the second caller should proceed once the first is done"


def test_ensure_loaded_does_not_take_the_lock_itself():
    # The lock is a plain Lock, not an RLock, so the entry points must take
    # it exactly once. If ensure_loaded ever starts taking it too, chat()
    # deadlocks rather than failing a test, which is far harder to trace.
    import inspect

    from rag.routing import loader as loader_module

    source = inspect.getsource(loader_module.StickyModelLoader.ensure_loaded)
    assert "_GENERATION_LOCK" not in source


def test_non_streaming_chat_also_holds_it():
    loader = StickyModelLoader()
    llm = MagicMock()
    llm.create_chat_completion.return_value = {"choices": [{"message": {"content": "hi"}}]}
    resident = MagicMock()
    resident.llm = llm
    resident.spec.route = "default"
    loader._resident = resident

    observed = {}

    def ensure(route):
        observed["locked"] = generation_lock().locked()
        return resident.spec

    loader.ensure_loaded = ensure
    assert loader.chat(route="default", messages=[]) == "hi"
    assert observed["locked"] is True
