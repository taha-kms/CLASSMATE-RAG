"""Building the numbered context blocks the model is asked to cite."""

from rag.generation import build_general_messages, build_grounded_messages, format_context_blocks


def _hit(doc, path):
    return {"document": doc, "metadata": {"source_path": path}}


def test_blocks_are_numbered_from_one_and_carry_provenance():
    text, prov = format_context_blocks(
        [_hit("chain rule", "notes.md"), _hit("product rule", "slides.pdf")]
    )
    assert "[1] chain rule" in text
    assert "[2] product rule" in text
    assert prov == ["notes.md", "slides.pdf"]


def test_the_character_budget_truncates_instead_of_growing_the_prompt():
    hits = [_hit("x" * 500, f"doc{i}.md") for i in range(10)]
    text, _ = format_context_blocks(hits, max_total_chars=1200)
    assert len(text) < 2000


def test_a_hit_with_no_source_path_still_gets_a_provenance_entry():
    _, prov = format_context_blocks([{"document": "text", "metadata": {}}])
    assert prov == ["chunk-1"]


def test_provenance_stays_aligned_when_a_hit_has_empty_text():
    # Empty documents are skipped in the text but must not shift the numbering.
    _, prov = format_context_blocks([_hit("", "empty.md"), _hit("real", "real.md")])
    assert prov == ["empty.md", "real.md"]


def test_the_grounded_prompt_asks_for_citations():
    messages = build_grounded_messages(question="What?", context_text="[1] ctx")
    system = messages[0]["content"].lower()
    assert "citation" in system or "cite" in system
    assert "[1] ctx" in messages[1]["content"]


def test_citations_can_be_made_optional():
    messages = build_grounded_messages(question="What?", context_text="[1] ctx", citations_required=False)
    assert "optional" in messages[0]["content"].lower()


def test_the_general_prompt_carries_no_context():
    messages = build_general_messages("What is the chain rule?")
    assert len(messages) == 2
    assert "[1]" not in messages[1]["content"]
