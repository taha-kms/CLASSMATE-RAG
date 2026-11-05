"""Findings from the first CodeQL run over the new API surface.

Exposing `resolve_ids` and the job registry over HTTP turned two internal
conveniences into request-reachable behaviour:

* `DELETE /chunks {"path": ...}` reached `Path(path).resolve()`, a filesystem
  operation driven by a request body. The value is only ever a lookup key
  compared against `source_path`, so it has no business touching disk.
* A failed job returned `traceback.format_exc()` in its result, so
  `GET /admin/jobs/{id}` served internal paths and frames.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from rag.admin.manage import CatalogEntry, resolve_ids
from rag.api.jobs import JobRegistry


def _await_finish(registry: JobRegistry, job_id: str, timeout: float = 5.0):
    """Wait for a job thread to finish.

    A bare `for _ in range(n)` spin never yields the GIL, so the worker thread
    is not scheduled and the job still reads as "running".
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = registry.get(job_id)
        if state is not None and state.state != "running":
            return state
        time.sleep(0.01)
    raise AssertionError(f"job {job_id} did not finish within {timeout}s")


def _catalog(source_path: str) -> list[CatalogEntry]:
    return [CatalogEntry(id="cm_1", text="t", metadata={"source_path": source_path})]


class TestPathLookupDoesNotTouchTheFilesystem:
    def test_an_absolute_path_still_matches(self):
        target = "/home/someone/notes.pdf"
        with patch("rag.admin.manage._read_bm25_catalog", return_value=_catalog(target)):
            assert resolve_ids(path=target) == ["cm_1"]

    def test_a_relative_path_is_normalised(self):
        target = os.path.abspath("notes.pdf")
        with patch("rag.admin.manage._read_bm25_catalog", return_value=_catalog(target)):
            assert resolve_ids(path="./notes.pdf") == ["cm_1"]

    def test_a_path_that_matches_nothing_returns_nothing(self):
        with patch("rag.admin.manage._read_bm25_catalog", return_value=_catalog("/a/b.pdf")):
            assert resolve_ids(path="/c/d.pdf") == []

    def test_resolve_is_not_called_on_request_controlled_input(self):
        """Path.resolve() stats the filesystem; abspath is pure string work."""
        called = []
        real_resolve = Path.resolve

        def spy(self, *a, **k):
            called.append(str(self))
            return real_resolve(self, *a, **k)

        with patch("rag.admin.manage._read_bm25_catalog", return_value=_catalog("/a/b.pdf")):
            with patch.object(Path, "resolve", spy):
                resolve_ids(path="/etc/../etc/passwd")

        assert called == [], f"the request path reached the filesystem: {called}"

    def test_traversal_segments_are_collapsed_not_followed(self):
        with patch("rag.admin.manage._read_bm25_catalog", return_value=_catalog("/etc/passwd")):
            # Collapses to /etc/passwd, which is a plain string comparison --
            # nothing is opened either way.
            assert resolve_ids(path="/etc/foo/../passwd") == ["cm_1"]


class TestJobsDoNotLeakTracebacks:
    @staticmethod
    def _run_failing_job() -> dict:
        registry = JobRegistry()

        def boom() -> dict:
            raise RuntimeError("something internal went wrong")

        job = registry.submit("rebuild", boom)
        state = _await_finish(registry, job.id)
        return state.as_dict()

    def test_the_failure_is_reported(self):
        body = self._run_failing_job()
        assert body["state"] == "failed"
        assert "something internal went wrong" in body["error"]

    def test_no_traceback_reaches_the_response(self):
        body = self._run_failing_job()
        blob = repr(body)
        assert "Traceback" not in blob, f"a traceback was returned: {blob}"
        assert 'File "' not in blob, f"source file paths were returned: {blob}"
        assert body["result"] is None


@pytest.mark.parametrize("field", ["error", "result"])
def test_a_succeeding_job_carries_no_error(field):
    registry = JobRegistry()
    job = registry.submit("dump", lambda: {"written": 2})
    body = _await_finish(registry, job.id).as_dict()
    assert body["state"] == "succeeded"
    assert (body["error"] is None) if field == "error" else (body["result"] == {"written": 2})
