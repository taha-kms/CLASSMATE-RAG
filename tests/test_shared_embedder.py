"""The embedding model is built once per process, under a lock.

Four concurrent /ask requests each constructed their own SentenceTransformer
and two of them died:

    NotImplementedError: Cannot copy out of meta tensor; no data!

torch was moving the same weights onto a device from several threads at once.
Building it once behind a lock fixes that, and incidentally stops every request
paying for a 1.1 GB model load.

No torch here: the construction is patched, so what is under test is the
caching and the locking rather than sentence-transformers itself.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pytest

import rag.embeddings as emb


@pytest.fixture(autouse=True)
def _clear_cache():
    emb._EMBEDDER_CACHE.clear()
    yield
    emb._EMBEDDER_CACHE.clear()


class _FakeEmbedder:
    def __init__(self, model_name="m", device=None, normalize=True):
        self.model_name = model_name
        self.device = device
        self.normalize = normalize


def test_the_same_settings_return_the_same_instance():
    with patch.object(emb, "E5MultilingualEmbedder", _FakeEmbedder):
        assert emb.shared_embedder("a") is emb.shared_embedder("a")


def test_different_models_get_different_instances():
    """rebuild_embeddings asks for a second model on purpose."""
    with patch.object(emb, "E5MultilingualEmbedder", _FakeEmbedder):
        assert emb.shared_embedder("a") is not emb.shared_embedder("b")


def test_concurrent_callers_construct_it_exactly_once():
    """The crash came from two threads entering the constructor together."""
    calls = []
    entered = threading.Event()

    class SlowEmbedder(_FakeEmbedder):
        def __init__(self, model_name="m", device=None, normalize=True):
            calls.append(model_name)
            entered.set()
            # Wide enough that an unlocked implementation reliably overlaps.
            threading.Event().wait(0.05)
            super().__init__(model_name, device, normalize)

    with patch.object(emb, "E5MultilingualEmbedder", SlowEmbedder):
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = [f.result() for f in [pool.submit(emb.shared_embedder, "same") for _ in range(8)]]

    assert len(calls) == 1, f"the model was constructed {len(calls)} times, not once"
    assert len({id(r) for r in results}) == 1, "callers got different instances"


def test_the_lock_is_held_across_construction_not_just_the_write():
    """Locking only the dict write still lets both threads build a model."""
    import inspect

    source = inspect.getsource(emb.shared_embedder)
    with_at = source.index("with _EMBEDDER_LOCK")
    build_at = source.index("E5MultilingualEmbedder(")
    assert with_at < build_at, "construction must happen inside the lock, not before or after it"
