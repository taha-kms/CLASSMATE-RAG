from rag.metadata import normalize_cli_metadata, LanguageEnum, DocTypeEnum

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
