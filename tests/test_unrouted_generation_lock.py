"""The unrouted path serializes generation, same as the routed one.

LlamaCppBackend has two branches. With a route it delegates to
StickyModelLoader, which holds a single lock so two routes cannot race to swap
the resident model. Without one it used to call a freshly built LlamaCppRunner
directly -- no lock, and LlamaCppRunner.__init__ loads the GGUF eagerly.

ENABLE_ROUTING defaults to false, so that unlocked branch is the one almost
everyone runs. It was invisible while the only caller was a CLI that asks one
question and exits. Two concurrent API requests each loaded their own copy of
the model and generated at the same time.

No llama.cpp here: the runner is patched, so what is under test is the locking
and the caching.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from rag.generation.backend import LlamaCppBackend
from rag.routing.loader import generation_lock


@pytest.fixture(autouse=True)
def _reset_runner():
    LlamaCppBackend._shared_runner = None
    yield
    LlamaCppBackend._shared_runner = None


class FakeRunner:
    """Records overlap the way a real model would suffer it."""

    built = 0
    live = 0
    max_live = 0
    _counter_lock = threading.Lock()

    def __init__(self):
        with FakeRunner._counter_lock:
            FakeRunner.built += 1

    def _work(self):
        with FakeRunner._counter_lock:
            FakeRunner.live += 1
            FakeRunner.max_live = max(FakeRunner.max_live, FakeRunner.live)
        threading.Event().wait(0.03)
        with FakeRunner._counter_lock:
            FakeRunner.live -= 1

    def chat(self, messages, **kwargs):
        self._work()
        return "answer"

    def chat_stream(self, messages, **kwargs):
        def gen():
            self._work()
            yield "answer"

        return gen()


@pytest.fixture(autouse=True)
def _reset_counters():
    FakeRunner.built = FakeRunner.live = FakeRunner.max_live = 0
    yield


@pytest.fixture
def backend(monkeypatch):
    b = LlamaCppBackend()
    monkeypatch.setattr(LlamaCppBackend, "_runner", lambda self: FakeRunner())
    return b


def _drain(backend, n=6):
    def one(_i):
        return list(backend.chat_stream([{"role": "user", "content": "q"}]))

    with ThreadPoolExecutor(max_workers=n) as pool:
        return [f.result() for f in [pool.submit(one, i) for i in range(n)]]


def test_concurrent_streams_never_overlap(backend):
    _drain(backend)
    assert FakeRunner.max_live == 1, f"{FakeRunner.max_live} generations ran at once; the lock is not held"


def test_concurrent_chats_never_overlap(backend):
    with ThreadPoolExecutor(max_workers=6) as pool:
        [f.result() for f in [pool.submit(backend.chat, [{"role": "user", "content": "q"}]) for _ in range(6)]]
    assert FakeRunner.max_live == 1


def test_the_lock_is_released_when_the_stream_is_exhausted(backend):
    list(backend.chat_stream([{"role": "user", "content": "q"}]))
    assert not generation_lock().locked(), "the lock outlived the stream"


def test_the_lock_is_released_when_the_stream_is_abandoned(backend):
    """A client that disconnects mid-answer must not wedge the process."""
    stream = backend.chat_stream([{"role": "user", "content": "q"}])
    next(stream)
    stream.close()
    assert not generation_lock().locked(), "closing the stream left the lock held"


def test_the_lock_is_released_if_starting_the_stream_raises(monkeypatch):
    b = LlamaCppBackend()

    def boom(self):
        raise RuntimeError("no model")

    monkeypatch.setattr(LlamaCppBackend, "_runner", boom)
    with pytest.raises(RuntimeError):
        b.chat_stream([{"role": "user", "content": "q"}])
    assert not generation_lock().locked(), "a failed start left the lock held"


def test_the_runner_is_built_once_per_process(monkeypatch):
    """__init__ loads the GGUF, so per-call construction reloaded it each time."""
    monkeypatch.setattr("rag.generation.LlamaCppRunner", FakeRunner, raising=False)
    import rag.generation as gen

    monkeypatch.setattr(gen, "LlamaCppRunner", FakeRunner, raising=False)

    b = LlamaCppBackend()
    first = b._runner()
    second = b._runner()
    assert first is second
    assert FakeRunner.built == 1, f"the model was loaded {FakeRunner.built} times"
