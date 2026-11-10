"""The values behind the filter dropdowns."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from rag.admin.manage import FACET_LIMIT, CatalogEntry, distinct_values

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from rag.api.app import create_app  # noqa: E402


def _entries(*metas: dict) -> list[CatalogEntry]:
    return [CatalogEntry(id=f"c{i}", text="t", metadata=m) for i, m in enumerate(metas)]


def test_values_are_distinct_sorted_and_blanks_dropped():
    cat = _entries(
        {"course": "Maths", "unit": "2"},
        {"course": "Chemistry", "unit": "10"},
        {"course": "Maths", "unit": 2},  # a number from an older ingest
        {"course": "  ", "unit": None},
        {"author": "Rossi"},
    )
    with patch("rag.admin.manage._read_bm25_catalog", return_value=cat):
        facets = distinct_values()

    assert facets["course"] == ["Chemistry", "Maths"]
    assert facets["unit"] == ["2", "10"]  # numeric order, not "10" before "2"
    assert facets["author"] == ["Rossi"]
    assert facets["semester"] == []


def test_a_huge_field_is_capped():
    cat = _entries(*({"author": f"a{i:04d}"} for i in range(FACET_LIMIT + 50)))
    with patch("rag.admin.manage._read_bm25_catalog", return_value=cat):
        assert len(distinct_values()["author"]) == FACET_LIMIT


def test_an_empty_corpus_gives_empty_lists():
    with patch("rag.admin.manage._read_bm25_catalog", return_value=[]):
        assert distinct_values() == {"course": [], "unit": [], "author": [], "semester": []}


def test_the_endpoint_returns_them():
    cat = _entries({"course": "Maths", "semester": "2025-1"})
    with patch("rag.admin.manage._read_bm25_catalog", return_value=cat):
        r = TestClient(create_app()).get("/facets")
    assert r.status_code == 200
    assert r.json()["course"] == ["Maths"]
    assert r.json()["semester"] == ["2025-1"]
