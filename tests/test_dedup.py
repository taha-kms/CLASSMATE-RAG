"""Near-duplicate chunk filtering."""

from rag.utils.dedup import dedup_block_indices, dedup_text_blocks

UNIQUE = [
    "The chain rule differentiates a composition of two functions.",
    "The product rule handles a product of two differentiable functions.",
    "The quotient rule handles a ratio where the denominator is non-zero.",
]


def test_distinct_blocks_are_all_kept():
    assert dedup_block_indices(UNIQUE) == [0, 1, 2]


def test_an_exact_repeat_is_dropped_and_the_first_wins():
    blocks = [UNIQUE[0], UNIQUE[1], UNIQUE[0]]
    assert dedup_block_indices(blocks) == [0, 1]


def test_indices_line_up_with_the_input_positions():
    blocks = [UNIQUE[0], UNIQUE[0], UNIQUE[1]]
    keep = dedup_block_indices(blocks)
    assert [blocks[i] for i in keep] == [UNIQUE[0], UNIQUE[1]]


def test_a_document_repeating_the_same_text_keeps_one_copy():
    # The old implementation matched blocks back by string equality after a
    # second chunking pass, which mishandled exactly this shape.
    blocks = [UNIQUE[0]] * 4
    assert dedup_block_indices(blocks) == [0]


def test_a_low_threshold_collapses_more_aggressively():
    similar = [
        "The chain rule differentiates a composition of two functions.",
        "The chain rule differentiates a composition of two mappings.",
    ]
    assert len(dedup_block_indices(similar, jaccard_threshold=0.99)) == 2
    assert len(dedup_block_indices(similar, jaccard_threshold=0.20)) == 1


def test_empty_input():
    assert dedup_block_indices([]) == []


def test_the_text_returning_wrapper_still_behaves():
    blocks = [UNIQUE[0], UNIQUE[0], UNIQUE[1]]
    assert dedup_text_blocks(blocks) == [UNIQUE[0], UNIQUE[1]]
