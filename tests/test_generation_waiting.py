"""A queued generation announces itself rather than going quiet.

Only one generation runs at a time: StickyModelLoader holds a single lock so
two routes cannot race to swap the resident model. A second caller blocks
waiting for it, and over HTTP that is bytes not arriving -- indistinguishable
from a hang.

The first attempt at this checked the lock at the top of the HTTP response.
That reported it free every time: retrieval and embedding run first and take
seconds, so the lock is not yet held when the response begins. Driving two
real requests through the running server is what showed it, so the check moved
next to the generation call and these pin it there.
"""

from __future__ import annotations

import threading
from unittest.mock import patch

from rag.generation.stream import StageEvent
from rag.pipeline.rag import _waiting_if_busy


class _Lock:
    def __init__(self, held: bool) -> None:
        self._held = held

    def locked(self) -> bool:
        return self._held


def test_nothing_is_emitted_when_the_model_is_free():
    with patch("rag.routing.loader.generation_lock", lambda: _Lock(False)):
        assert list(_waiting_if_busy()) == []


def test_a_held_lock_produces_a_waiting_stage():
    with patch("rag.routing.loader.generation_lock", lambda: _Lock(True)):
        assert list(_waiting_if_busy()) == [StageEvent("waiting")]


def test_the_stage_carries_a_message_a_ui_can_show():
    assert StageEvent("waiting").message == "Waiting for another answer to finish"


def test_it_reads_the_real_lock_not_a_copy():
    """Patching aside, the helper must consult the process-wide lock."""
    from rag.routing.loader import generation_lock

    real = generation_lock()
    assert isinstance(real, type(threading.Lock()))

    assert list(_waiting_if_busy()) == [], "the lock should be free in a test run"
    real.acquire()
    try:
        assert list(_waiting_if_busy()) == [StageEvent("waiting")]
    finally:
        real.release()


def test_the_check_sits_before_generation_not_at_the_start_of_the_stream():
    """Guards against moving it back to where it reports free every time."""
    import inspect

    from rag.pipeline import rag as pipeline

    source = inspect.getsource(pipeline.ask_question_stream)
    waiting_at = source.index("_waiting_if_busy()")
    retrieving_at = source.index('StageEvent("retrieving")')
    generating_at = source.index('StageEvent("generating")')
    assert retrieving_at < waiting_at < generating_at, (
        "the waiting check must sit after retrieval and before generation; "
        "earlier than that and the lock is not yet held, so it never fires"
    )
