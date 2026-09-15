"""Reciprocal rank fusion.

This is the core of hybrid retrieval and had no coverage, because importing
rag.retrieval.fusion used to require sentence-transformers.
"""

import pytest

from rag.retrieval import rrf_fuse


def test_an_item_ranked_by_both_retrievers_beats_one_ranked_by_only_one():
    scores = rrf_fuse(rank_lists=[["a", "b"], ["b", "c"]])
    # b appears in both lists, so it should win even though a and c are each first somewhere.
    assert scores["b"] > scores["a"]
    assert scores["b"] > scores["c"]


def test_rank_matters_but_later_ranks_still_contribute():
    scores = rrf_fuse(rank_lists=[["first", "second", "third"]])
    assert scores["first"] > scores["second"] > scores["third"] > 0


def test_weights_shift_the_balance_between_retrievers():
    balanced = rrf_fuse(rank_lists=[["a"], ["b"]])
    assert balanced["a"] == pytest.approx(balanced["b"])

    lexical_heavy = rrf_fuse(rank_lists=[["a"], ["b"]], weights=[0.2, 1.0])
    assert lexical_heavy["b"] > lexical_heavy["a"]


def test_rrf_k_controls_how_sharply_rank_is_discounted():
    flat = rrf_fuse(rank_lists=[["a", "b"]], rrf_k=1000)
    steep = rrf_fuse(rank_lists=[["a", "b"]], rrf_k=1)
    # A large k flattens the curve, so the gap between ranks shrinks.
    assert (flat["a"] - flat["b"]) < (steep["a"] - steep["b"])


def test_no_rank_lists_is_not_an_error():
    assert rrf_fuse(rank_lists=[]) == {}


def test_mismatched_weights_are_rejected():
    with pytest.raises(ValueError):
        rrf_fuse(rank_lists=[["a"], ["b"]], weights=[1.0])
