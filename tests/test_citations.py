"""Citation cleanup applied to generated answers.

Reachable now that rag.generation resolves its exports lazily; it used to
need llama_cpp just to import enforce_citations.
"""

from rag.generation import enforce_citations

PROV = ["notes.md", "slides.pdf"]


def test_citations_pointing_past_the_available_sources_are_dropped():
    out = enforce_citations("The rule is X [1] and also Y [7].", PROV)
    assert "[1]" in out
    assert "[7]" not in out


def test_adjacent_citations_are_compacted():
    assert "[1][2]" in enforce_citations("Both apply [1], [2].", PROV)


def test_an_answer_with_no_citations_passes_through_unchanged():
    # Worth pinning: "strict" citations do not currently require any (#44).
    text = "The chain rule differentiates a composition."
    assert enforce_citations(text, PROV) == text


def test_an_empty_answer_stays_empty():
    assert enforce_citations("", PROV) == ""
    assert enforce_citations("   ", PROV) == ""


def test_the_sources_block_lists_only_what_was_cited():
    out = enforce_citations("Only the first [1].", PROV, add_sources_block=True)
    assert "notes.md" in out
    assert "slides.pdf" not in out


def test_the_sources_block_is_omitted_when_nothing_was_cited():
    out = enforce_citations("No citations here.", PROV, add_sources_block=True)
    assert "notes.md" not in out


def test_the_sources_title_can_be_localised():
    out = enforce_citations("Vale [1].", PROV, add_sources_block=True, sources_title="Fonti")
    assert "Fonti" in out
