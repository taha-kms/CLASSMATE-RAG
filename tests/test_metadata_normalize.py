from rag.metadata import normalize_cli_metadata, LanguageEnum, DocTypeEnum


def test_to_dict_strips_auto_and_other_sentinels():
    # `language=auto` and `doc_type=other` are sentinels meaning "no preference".
    # They must NOT leak into to_dict() — otherwise they get applied as filters
    # by the retrieval where-builder and silently narrow results to nothing.
    m = normalize_cli_metadata(language="auto", doc_type=None)
    d = m.to_dict()
    assert "language" not in d
    assert "doc_type" not in d

    # Real values pass through unchanged.
    m2 = normalize_cli_metadata(language="en", doc_type="pdf", course="cs50")
    d2 = m2.to_dict()
    assert d2["language"] == "en"
    assert d2["doc_type"] == "pdf"
    assert d2["course"] == "cs50"


def test_normalize_cli_metadata_enums_and_tags():
    meta = normalize_cli_metadata(
        course="  cs50 ",
        unit="  2 ",
        language="ita",
        doc_type="ppt",
        author=" Alice ",
        semester=" Fall ",
        tags="Exam,exam,Week1,week1,READING",
    )
    assert meta.language == LanguageEnum.it
    assert meta.doc_type == DocTypeEnum.pptx  # ppt maps to pptx
    # tags lowercased and deduped, order preserved first-seen
    assert meta.tags == ["exam", "week1", "reading"]
    assert meta.course == "cs50"
    assert meta.unit == "2"
    assert meta.author == "Alice"
    assert meta.semester == "Fall"
