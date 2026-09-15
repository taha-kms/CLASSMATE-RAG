"""Sources reflect what the answer cited, not what retrieval returned.

Previously `sources` carried the full provenance list whichever way the
answer went, so a UI would render them as if they backed text that might
never have referenced them.
"""

import pytest

from rag.generation.post import cited_indices
from rag.pipeline.rag import _attribute_sources

PROV = ["notes.md", "slides.pdf", "handout.md"]


def test_cited_indices_are_deduplicated_and_ordered_by_first_use():
    assert cited_indices("Start [2], then [1], back to [2] again.") == [2, 1]


def test_no_citations_means_no_indices():
    assert cited_indices("No markers at all.") == []


def test_only_cited_sources_are_reported():
    sources, grounded, notice = _attribute_sources("Only the first one [1].", PROV, "en")

    assert [s.ref for s in sources] == ["notes.md"]
    assert grounded is True
    assert notice is None


def test_source_numbers_match_the_markers_in_the_answer():
    # The CLI used to renumber with enumerate(), which would have relabelled
    # [3] as [1] and broken the link back to the text.
    sources, _, _ = _attribute_sources("Per the handout [3].", PROV, "en")
    assert [(s.n, s.ref) for s in sources] == [(3, "handout.md")]


def test_an_uncited_answer_is_reported_as_ungrounded_with_no_sources():
    sources, grounded, notice = _attribute_sources("The chain rule is a formula.", PROV, "en")

    assert sources == []
    assert grounded is False
    assert notice and "not your documents" in notice


def test_markers_outside_the_available_range_are_ignored():
    sources, grounded, _ = _attribute_sources("Claim [9].", PROV, "en")
    assert sources == []
    assert grounded is False


def test_a_partially_valid_citation_set_keeps_the_valid_ones():
    sources, grounded, _ = _attribute_sources("One [1] and nine [9].", PROV, "en")
    assert [s.n for s in sources] == [1]
    assert grounded is True


@pytest.mark.parametrize("language", ["en", "it"])
def test_the_notice_follows_the_answer_language(language):
    _, _, notice = _attribute_sources("No citations.", PROV, language)
    assert notice


def test_an_empty_provenance_list_cannot_be_grounded():
    sources, grounded, notice = _attribute_sources("Answer [1].", [], "en")
    assert sources == []
    assert grounded is False
    assert notice
