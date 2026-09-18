"""Finding rows stranded in one store.

BM25 is the de facto catalog: every filtered command resolves ids by
reading it. So a vector row without a BM25 entry is invisible to `list`,
`show`, `delete` and `reingest` while still being returned by search,
which is how deleted material kept coming back in #100.
"""

from unittest.mock import MagicMock, patch

from rag.admin.manage import find_orphans, reconcile_stores


def _stores(bm25_ids, vector_ids):
    bm = MagicMock()
    bm._entries = dict.fromkeys(bm25_ids)
    vec = MagicMock()
    vec.all_ids.return_value = list(vector_ids)
    vec.delete.return_value = 0
    return bm, vec


def _patched(bm, vec):
    return patch.multiple(
        "rag.retrieval",
        BM25Store=MagicMock(load_or_create=MagicMock(return_value=bm)),
        ChromaVectorStore=MagicMock(from_config=MagicMock(return_value=vec)),
    )


def test_stores_that_agree_have_no_orphans():
    bm, vec = _stores(["a", "b"], ["a", "b"])
    with _patched(bm, vec):
        assert find_orphans() == {"vector_only": [], "bm25_only": []}


def test_a_vector_row_with_no_catalog_entry_is_found():
    bm, vec = _stores(["a"], ["a", "stranded"])
    with _patched(bm, vec):
        assert find_orphans()["vector_only"] == ["stranded"]


def test_a_catalog_entry_with_no_vector_is_found_too():
    # Not retrievable by vector search, but still listed and deletable, so
    # it is a different problem and reported separately.
    bm, vec = _stores(["a", "unembedded"], ["a"])
    with _patched(bm, vec):
        assert find_orphans()["bm25_only"] == ["unembedded"]


def test_reporting_does_not_delete_anything():
    bm, vec = _stores(["a"], ["a", "stranded"])
    with _patched(bm, vec):
        result = reconcile_stores(dry_run=True)

    assert result["vector_only"] == 1
    assert result["removed"] == 0
    assert result["dry_run"] is True
    vec.delete.assert_not_called()


def test_applying_removes_only_the_vector_orphans():
    # A BM25 entry without a vector can be repaired by re-embedding. A
    # vector row without metadata cannot be repaired at all, and is
    # unreachable through every command that filters.
    bm, vec = _stores(["a", "unembedded"], ["a", "stranded"])
    vec.delete.return_value = 1
    with _patched(bm, vec):
        result = reconcile_stores(dry_run=False)

    vec.delete.assert_called_once_with(ids=["stranded"])
    assert result["removed"] == 1
    assert result["bm25_only"] == 1, "reported, and left alone"


def test_nothing_is_deleted_when_there_is_nothing_stranded():
    bm, vec = _stores(["a"], ["a"])
    with _patched(bm, vec):
        reconcile_stores(dry_run=False)
    vec.delete.assert_not_called()


def test_a_sample_is_reported_for_checking_by_hand():
    many = [f"orphan_{i}" for i in range(50)]
    bm, vec = _stores([], many)
    with _patched(bm, vec):
        result = reconcile_stores(dry_run=True)

    assert result["vector_only"] == 50
    # Enough to spot-check, not enough to bury the summary.
    assert len(result["sample"]) == 10


def test_stats_flags_a_disagreement():
    # Two counts side by side that disagree is the signal that would have
    # surfaced #100 far earlier.
    from rag.admin import inspect as inspect_module

    vec = MagicMock()
    vec.count.return_value = 14
    bm = MagicMock()
    bm.count.return_value = 2

    with (
        patch.object(inspect_module, "ChromaVectorStore", MagicMock(from_config=MagicMock(return_value=vec))),
        patch.object(inspect_module, "BM25Store", MagicMock(load_or_create=MagicMock(return_value=bm))),
    ):
        stats = inspect_module.index_stats()

    assert stats["consistent"] is False
    assert "14" in stats["warning"] and "2" in stats["warning"]
    assert "rag reconcile" in stats["warning"]


def test_stats_says_nothing_when_the_stores_agree():
    from rag.admin import inspect as inspect_module

    vec = MagicMock()
    vec.count.return_value = 7
    bm = MagicMock()
    bm.count.return_value = 7

    with (
        patch.object(inspect_module, "ChromaVectorStore", MagicMock(from_config=MagicMock(return_value=vec))),
        patch.object(inspect_module, "BM25Store", MagicMock(load_or_create=MagicMock(return_value=bm))),
    ):
        stats = inspect_module.index_stats()

    assert stats["consistent"] is True
    assert "warning" not in stats
