"""Handlers must not turn a bug into a plausible-looking value.

#41 and #100 were the same mistake twice: a bare `except` caught an
AttributeError from a method that did not exist, and the handler went on to
produce a number that looked like a documented result -- `-1` for "unavailable"
and `len(ids)` for "deleted". Both were invisible for months.

These tests pin the distinction the audit introduced. A store that cannot be
reached is a condition callers may degrade on; anything else is a bug and has
to reach the caller.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from unittest.mock import patch

import pytest

import rag.admin.backup as backup
import rag.admin.inspect as inspect_mod
from rag.pipeline.rag import _concurrent_chunk_pages
from rag.retrieval.vector_chroma import VectorStoreUnavailable

ROOT = Path(__file__).resolve().parents[1]


class TestIndexStats:
    """`-1` is the documented 'cannot reach it', not a catch-all."""

    def test_unreachable_store_still_reports_minus_one(self):
        with patch.object(
            inspect_mod.ChromaVectorStore,
            "count",
            side_effect=VectorStoreUnavailable("server is down"),
        ):
            assert inspect_mod.index_stats()["vector_count"] == -1

    @pytest.mark.parametrize(
        "boom",
        [
            AttributeError("'ChromaVectorStore' object has no attribute 'count'"),
            TypeError("count() takes 1 positional argument but 2 were given"),
        ],
        ids=["missing-method", "wrong-signature"],
    )
    def test_a_bug_is_not_dressed_up_as_minus_one(self, boom):
        """This is #41 exactly: the method did not exist and -1 hid it."""
        with patch.object(inspect_mod.ChromaVectorStore, "count", side_effect=boom):
            with pytest.raises(type(boom)):
                inspect_mod.index_stats()


class TestVectorStoreCount:
    def test_count_does_not_swallow(self):
        """Returning 0 on failure reads as 'the corpus is empty'."""
        from rag.retrieval import ChromaVectorStore

        store = ChromaVectorStore(persist_dir=Path("/nonexistent"))
        with patch.object(
            ChromaVectorStore,
            "_ensure_collection",
            side_effect=RuntimeError("chroma exploded"),
        ):
            with pytest.raises(RuntimeError, match="chroma exploded"):
                store.count()


class TestVacuum:
    def test_unreachable_store_becomes_a_status_string(self):
        with patch.object(backup, "ChromaVectorStore") as cls:
            cls.from_config.return_value.compact.side_effect = VectorStoreUnavailable("down")
            with patch.object(backup, "BM25Store"):
                assert "error" in str(backup.vacuum_indexes()["chroma"])

    def test_a_bug_is_not_reported_as_routine_output(self):
        """hasattr said the method exists, so an AttributeError means a bug."""
        with patch.object(backup, "ChromaVectorStore") as cls:
            cls.from_config.return_value.compact.side_effect = AttributeError("gone")
            with patch.object(backup, "BM25Store"):
                with pytest.raises(AttributeError):
                    backup.vacuum_indexes()


class TestChunkingIsNotSilentlyLossy:
    def test_a_failed_page_names_itself_instead_of_vanishing(self):
        """Dropping the page indexed the document incomplete and said 'done'."""
        pages = [(1, "first page text"), (2, "second page text")]

        def explode(text, **kwargs):
            if kwargs.get("page") == 2:
                raise ValueError("bad page")
            return []

        with patch("rag.pipeline.rag.chunk_text", side_effect=explode):
            with pytest.raises(RuntimeError, match="Failed to chunk page 2"):
                _concurrent_chunk_pages(pages, chunk_size=100, chunk_overlap=0, max_workers=2)


class TestRulesStayOn:
    """Re-enabling the rules is the point; a silent revert undoes the audit."""

    def test_the_three_rules_are_selected(self):
        cfg = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        selected = cfg["tool"]["ruff"]["lint"]["select"]
        for rule in ("BLE001", "S110", "S112"):
            assert rule in selected, f"{rule} was dropped from the ruff config"

    def test_no_per_file_ignore_smuggles_them_back(self):
        cfg = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        ignores = cfg["tool"]["ruff"]["lint"].get("per-file-ignores", {})
        for path, rules in ignores.items():
            overlap = set(rules) & {"BLE001", "S110", "S112"}
            assert not overlap, (
                f"{path} ignores {sorted(overlap)}. The sites that legitimately "
                f"stay broad carry an inline `# noqa` with a reason instead, so "
                f"each one is justified where it sits."
            )
