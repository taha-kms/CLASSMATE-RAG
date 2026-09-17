import pytest

from rag.metadata.validation import validate_cli_metadata


def test_validate_cli_metadata_fixup_false_rejects_bad_tag():
    raw = {"language": "en", "doc_type": "pdf", "tags": "bad tag with spaces"}
    with pytest.raises(ValueError):
        validate_cli_metadata(raw, fixup=False)


def test_validate_cli_metadata_fixup_true_slugifies_and_infers_doc_type():
    raw = {
        "course": "  Math101 ",
        "unit": "  1 ",
        "language": "English",
        "doc_type": None,  # will be inferred when provided
        "author": "  Prof. X ",
        "semester": " 2025S ",
        "tags": "Exam Week 1,  tricky-Tag  ",
    }
    cleaned = validate_cli_metadata(
        raw,
        fixup=True,
        inferred_doc_type="pdf",
        explicit_doc_type=False,
    )
    # language coerced to 'auto' during permissive path then normalized later
    assert cleaned["doc_type"] == "pdf"
    # slugged and deduped tags (lower/snake-ish)
    assert cleaned["tags"] and all(t.replace("_", "").isalnum() for t in cleaned["tags"])


def test_validate_cli_metadata_bad_doc_type_behavior():
    raw = {"doc_type": "pptzzz"}

    # STRICT path: should error on unknown doc_type
    import pytest

    with pytest.raises(ValueError):
        validate_cli_metadata(raw, fixup=False, explicit_doc_type=True)

    # FIXUP path: should coerce/normalize instead of raising
    cleaned = validate_cli_metadata(
        raw,
        fixup=True,
        explicit_doc_type=True,
        inferred_doc_type="pdf",  # repo uses this when it needs to fall back
    )
    assert isinstance(cleaned["doc_type"], str)
    assert cleaned["doc_type"] != "pptzzz"  # must not keep the invalid value


def test_validation_runs_on_pydantic_v2_without_deprecation_shims():
    import pydantic

    assert pydantic.VERSION.startswith("2."), "the validator is written against pydantic v2"

    from rag.metadata import validation

    # The dual-version shim is gone; these are the v2 spellings.
    assert hasattr(validation._MetaInput, "model_config")
    assert not hasattr(validation._MetaInput, "Config")


def test_whitespace_only_values_become_none():
    out = validate_cli_metadata({"course": "   ", "author": "\t"}, fixup=False)
    assert out["course"] is None
    assert out["author"] is None


def test_tags_accept_a_comma_separated_string():
    out = validate_cli_metadata({"tags": "exam, week1 ,, "}, fixup=False)
    assert out["tags"] == ["exam", "week1"]
