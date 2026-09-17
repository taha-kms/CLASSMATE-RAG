"""Deleting must remove from both stores, and must not lie about it.

`rag delete` reported success while removing nothing from the vector store:
ChromaVectorStore had no delete() at all, and delete_by_ids caught the
AttributeError and reported len(ids) as the number deleted.
"""

from unittest.mock import MagicMock

import pytest

from rag.retrieval import BM25Store, ChromaVectorStore

META = {"course": "Calculus101", "unit": "3", "language": "en", "doc_type": "md"}


def test_the_vector_store_exposes_a_delete():
    # The bug was not a broken delete, it was a missing one.
    assert hasattr(ChromaVectorStore, "delete")
    assert callable(ChromaVectorStore.delete)


def test_delete_passes_the_ids_to_the_collection():
    store = ChromaVectorStore.__new__(ChromaVectorStore)
    collection = MagicMock()
    collection.count.side_effect = [10, 7]
    store._ensure_collection = MagicMock(return_value=collection)

    removed = store.delete(ids=["a", "b", "c"])

    collection.delete.assert_called_once_with(ids=["a", "b", "c"])
    # Reports what actually went, measured, not the length of the request.
    assert removed == 3


def test_delete_reports_what_was_removed_not_what_was_asked():
    store = ChromaVectorStore.__new__(ChromaVectorStore)
    collection = MagicMock()
    # Two of the three ids were not in the store.
    collection.count.side_effect = [10, 9]
    store._ensure_collection = MagicMock(return_value=collection)

    assert store.delete(ids=["a", "b", "c"]) == 1


def test_deleting_nothing_is_not_an_error():
    store = ChromaVectorStore.__new__(ChromaVectorStore)
    store._ensure_collection = MagicMock()
    assert store.delete(ids=[]) == 0
    store._ensure_collection.assert_not_called()


def test_a_failing_delete_raises_instead_of_reporting_success():
    # The original swallowed everything and returned len(ids), so a total
    # failure looked like a complete success.
    store = ChromaVectorStore.__new__(ChromaVectorStore)
    collection = MagicMock()
    collection.delete.side_effect = RuntimeError("backend is down")
    store._ensure_collection = MagicMock(return_value=collection)

    with pytest.raises(RuntimeError):
        store.delete(ids=["a"])


def test_bm25_delete_many_reports_a_count(tmp_path):
    store = BM25Store.load_or_create(tmp_path / "bm25")
    store.upsert_many(ids=["a", "b"], texts=["one", "two"], metadatas=[META, META])

    assert store.delete_many(["a"]) == 1
    # Deleting something absent removes nothing, and says so.
    assert store.delete_many(["nope"]) == 0
    assert store.count() == 1
