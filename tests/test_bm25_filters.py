"""Metadata filtering in the BM25 store.

These cover the lexical half of hybrid retrieval, which has to agree with
the Chroma half about what a filter value of None means.
"""


from rag.retrieval.bm25 import BM25Store, _matches_filter

CHUNK_META = {
    "course": "Calculus101",
    "unit": "3",
    "language": "en",
    "doc_type": "md",
}


def test_unspecified_filter_fields_do_not_exclude_anything():
    # DocumentMetadata.to_dict() emits a key for every filter field, using
    # None for the ones the caller left out. None means "don't care", not
    # "this field must be null".
    where = {
        "course": "Calculus101",
        "unit": None,
        "author": None,
        "semester": None,
        "source_path": None,
        "created_at": None,
    }
    assert _matches_filter(CHUNK_META, where) is True


def test_a_populated_field_still_has_to_match_when_it_is_specified():
    assert _matches_filter(CHUNK_META, {"course": "Calculus101"}) is True
    assert _matches_filter(CHUNK_META, {"course": "Physics202"}) is False
    assert _matches_filter(CHUNK_META, {"course": "Calculus101", "unit": "4"}) is False


def test_search_returns_hits_when_only_some_filters_are_given(tmp_path):
    store = BM25Store.load_or_create(tmp_path / "bm25")
    store.upsert_many(
        ids=["a", "b"],
        texts=[
            "The chain rule differentiates a composition of two functions.",
            "The product rule handles a product of two functions.",
        ],
        metadatas=[CHUNK_META, CHUNK_META],
    )

    unfiltered = store.search(query="chain rule", top_k=5)
    assert len(unfiltered) == 2

    # The shape the CLI actually passes: one real filter, the rest None.
    filtered = store.search(
        query="chain rule",
        where={"course": "Calculus101", "unit": None, "author": None, "semester": None},
        top_k=5,
    )
    assert len(filtered) == len(unfiltered)


def test_a_non_matching_filter_still_excludes(tmp_path):
    store = BM25Store.load_or_create(tmp_path / "bm25")
    store.upsert_many(ids=["a"], texts=["The chain rule."], metadatas=[CHUNK_META])
    assert store.search(query="chain", where={"course": "Physics202"}, top_k=5) == []


def test_count_reflects_upserts_and_deletes(tmp_path):
    store = BM25Store.load_or_create(tmp_path / "bm25")
    assert store.count() == 0

    store.upsert_many(
        ids=["a", "b"],
        texts=["The chain rule.", "The product rule."],
        metadatas=[CHUNK_META, CHUNK_META],
    )
    assert store.count() == 2

    # Upserting the same id replaces rather than appends.
    store.upsert_many(ids=["a"], texts=["The chain rule, again."], metadatas=[CHUNK_META])
    assert store.count() == 2

    store.delete_many(["a"])
    assert store.count() == 1
