from rag.admin.manage import _matches_simple

def test_matches_simple_with_tags_and_scalars():
    meta = {
        "course": "Math101",
        "unit": "1",
        "language": "en",
        "author": "Prof",
        # tags are represented as tag_<slug>: True in metadata
        "tag_exam": True,
        "tag_week1": True,
        "tag_hard": False,
    }
    # Must have both exam and week1, regardless of order or comma string
    where = {"course": "Math101", "tags": "exam,week1"}
    assert _matches_simple(meta, where) is True

    where = {"course": "Math101", "tags": ["exam", "hard"]}
    assert _matches_simple(meta, where) is False  # hard not present

    where = {"course": "Other"}
    assert _matches_simple(meta, where) is False
