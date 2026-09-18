"""The local API, driven through TestClient.

The pipeline itself is patched out. What is under test here is the layer the
CLI does not have: request validation, SSE framing, the filter mapping, the
job lifecycle, and the parts of the contract a frontend will depend on.

These run without chromadb or torch, which is the point of keeping every heavy
import inside a handler in rag/api/app.py.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from unittest.mock import patch

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from rag.api.app import _filter_mapping, create_app  # noqa: E402
from rag.api.models import Filters  # noqa: E402


@pytest.fixture
def client():
    return TestClient(create_app())


def sse_events(body: str) -> list[tuple[str, dict]]:
    """Parse an SSE body into (event name, payload) pairs."""
    out = []
    for block in body.strip().split("\n\n"):
        if not block.strip():
            continue
        name, data = None, None
        for line in block.splitlines():
            if line.startswith("event: "):
                name = line[len("event: ") :]
            elif line.startswith("data: "):
                data = json.loads(line[len("data: ") :])
        out.append((name, data))
    return out


class TestFilterMapping:
    """A filter left out must mean "do not filter" -- the #40 mistake."""

    def test_unset_fields_are_dropped_entirely(self):
        assert _filter_mapping(Filters(course="Maths")) == {"course": "Maths"}

    def test_no_none_values_survive(self):
        mapping = _filter_mapping(Filters())
        assert mapping == {}, f"a None leaked through as a filter value: {mapping}"


class TestAskStream:
    def test_events_are_framed_as_sse_in_order(self, client):
        from rag.generation.stream import StageEvent, TokenEvent

        @dataclass
        class FakeResult:
            question: str = "q"
            answer: str = "hello there"
            language: str = "en"

        from rag.generation.stream import FinalEvent

        def fake_stream(**kwargs):
            yield StageEvent("retrieving")
            yield TokenEvent("hello ")
            yield TokenEvent("there")
            yield FinalEvent(FakeResult())

        with patch("rag.pipeline.ask_question_stream", fake_stream):
            r = client.post("/ask", json={"question": "what is x?"})

        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        events = sse_events(r.text)
        assert [name for name, _ in events] == ["stage", "token", "token", "final"]
        assert events[0][1]["stage"] == "retrieving"
        assert "".join(p["text"] for n, p in events if n == "token") == "hello there"
        assert events[-1][1]["answer"] == "hello there"

    def test_a_waiting_stage_reaches_the_client(self):
        """The pipeline emits it; this checks the API does not drop it.

        Where the check lives matters and is tested in
        tests/test_generation_waiting.py: an earlier version asked the lock at
        the top of the response, before retrieval, where it is always free.
        """
        from rag.generation.stream import FinalEvent, StageEvent

        @dataclass
        class FakeResult:
            answer: str = "done"

        def fake_stream(**kwargs):
            yield StageEvent("retrieving")
            yield StageEvent("waiting")
            yield FinalEvent(FakeResult())

        client = TestClient(create_app())
        with patch("rag.pipeline.ask_question_stream", fake_stream):
            r = client.post("/ask", json={"question": "q"})

        events = sse_events(r.text)
        stages = [(p["stage"], p["message"]) for n, p in events if n == "stage"]
        assert ("waiting", "Waiting for another answer to finish") in stages

    def test_a_failure_mid_stream_becomes_an_error_event(self, client):
        """Headers are already sent, so raising would truncate with no reason."""
        from rag.generation.stream import StageEvent

        def fake_stream(**kwargs):
            yield StageEvent("retrieving")
            raise RuntimeError("the model fell over")

        with patch("rag.pipeline.ask_question_stream", fake_stream):
            r = client.post("/ask", json={"question": "q"})

        assert r.status_code == 200  # cannot be changed after the first byte
        name, payload = sse_events(r.text)[-1]
        assert name == "error"
        assert "the model fell over" in payload["error"]

    @pytest.mark.parametrize(
        "body",
        [{}, {"question": ""}, {"question": "q", "top_k": 0}, {"question": "q", "top_k": 500}],
        ids=["no-question", "empty-question", "top_k-too-low", "top_k-too-high"],
    )
    def test_bad_requests_are_rejected_before_any_work(self, client, body):
        assert client.post("/ask", json=body).status_code == 422


class TestChunks:
    def test_show_returns_404_for_an_unknown_id(self, client):
        with patch("rag.admin.manage.show_entries_by_id", return_value=[]):
            r = client.get("/chunks/nope")
        assert r.status_code == 404

    def test_delete_is_a_dry_run_unless_asked(self, client):
        """Deleting a corpus should not follow from a field left out."""
        with patch("rag.admin.manage.resolve_ids", return_value=["a", "b"]) as resolve:
            with patch("rag.admin.manage.delete_by_ids") as delete:
                r = client.request("DELETE", "/chunks", json={"course": "Maths"})

        assert resolve.called
        delete.assert_not_called()  # a default request must not delete anything
        body = r.json()
        assert body["dry_run"] is True
        assert body["matched"] == 2

    def test_delete_runs_when_dry_run_is_off(self, client):
        with patch("rag.admin.manage.resolve_ids", return_value=["a"]):
            with patch("rag.admin.manage.delete_by_ids", return_value=(1, 1)) as delete:
                r = client.request("DELETE", "/chunks", json={"course": "Maths", "dry_run": False})

        delete.assert_called_once_with(["a"])
        assert r.json() == {"dry_run": False, "matched": 1, "deleted_bm25": 1, "deleted_vector": 1}


class TestIngest:
    def test_a_file_without_an_extension_is_refused(self, client):
        """The loader is chosen from the suffix, so a bare name routes wrong."""
        r = client.post("/ingest", files={"file": ("notes", b"hello", "text/plain")})
        assert r.status_code == 422
        assert "extension" in r.json()["detail"]

    @staticmethod
    def _upload(client, tmp_root, filename=b"notes.md", body=b"# hi", query=""):
        """Upload a file with the project root redirected at a tmp dir."""
        seen = {}

        @dataclass
        class FakeIngestResult:
            path: str
            doc_type: str = ""
            total_chunks: int = 1

        def fake_ingest(*, path, doc_meta):
            seen["path"] = path
            seen["text"] = path.read_bytes()
            seen["meta"] = doc_meta
            return FakeIngestResult(path=str(path), doc_type=str(doc_meta.doc_type))

        with patch("rag.api.app.project_root", lambda: tmp_root):
            with patch("rag.pipeline.ingest_file", fake_ingest):
                name = filename.decode() if isinstance(filename, bytes) else filename
                r = client.post(f"/ingest{query}", files={"file": (name, body, "text/markdown")})
        return r, seen

    def test_the_upload_keeps_its_suffix_on_disk(self, client, tmp_path):
        """infer_doc_type_from_path switches on it; mkstemp names lose it."""
        r, seen = self._upload(client, tmp_path)
        assert r.status_code == 201
        assert seen["path"].suffix == ".md"
        assert seen["text"] == b"# hi"

    def test_the_upload_is_kept_where_source_path_can_find_it(self, client, tmp_path):
        """source_path lands in every chunk, and three commands read it back.

        Streaming the upload through a TemporaryDirectory left every chunk
        pointing at a path under /tmp that no longer existed, which makes
        `rag reingest`, `list --path` and `delete --path` quietly useless.
        """
        _r, seen = self._upload(client, tmp_path)
        stored = seen["path"]
        assert stored.is_file(), "the uploaded file must still exist after the request"
        assert stored.parent == tmp_path / "data" / "uploads"

    def test_doc_type_is_inferred_from_the_filename(self, client, tmp_path):
        """DocTypeEnum.other is truthy, so ingest_file never infers past it.

        Going through normalize_cli_metadata directly typed every single
        upload as "other" regardless of extension.
        """
        _r, seen = self._upload(client, tmp_path)
        assert str(seen["meta"].doc_type).endswith("md"), f"got {seen['meta'].doc_type!r}, expected md"

    def test_an_explicit_doc_type_still_wins(self, client, tmp_path):
        _r, seen = self._upload(client, tmp_path, query="?doc_type=txt")
        assert str(seen["meta"].doc_type).endswith("txt")

    def test_a_bad_tag_is_reported_and_the_file_is_not_left_behind(self, client, tmp_path):
        r, _seen = self._upload(client, tmp_path, query="?tags=not%20a%20valid%20tag!")
        assert r.status_code == 422
        leftovers = list((tmp_path / "data" / "uploads").glob("*")) if (tmp_path / "data" / "uploads").exists() else []
        assert leftovers == [], f"a rejected upload was left on disk: {leftovers}"

    def test_a_traversing_filename_cannot_escape_the_uploads_directory(self, client, tmp_path):
        r, seen = self._upload(client, tmp_path, filename="../../evil.md")
        assert r.status_code == 201
        assert seen["path"].parent == tmp_path / "data" / "uploads"
        assert seen["path"].name == "evil.md"


class TestJobs:
    def test_a_long_operation_returns_202_and_a_job_id(self, client):
        with patch("rag.admin.backup.rebuild_embeddings", return_value={"updated": 3}):
            r = client.post("/admin/rebuild", json={"model": "some/model"})
        assert r.status_code == 202
        assert r.json()["state"] in ("running", "succeeded")
        assert r.json()["operation"] == "rebuild"

    def test_a_finished_job_reports_its_result(self, client):
        with patch("rag.admin.backup.rebuild_embeddings", return_value={"updated": 3}):
            job_id = client.post("/admin/rebuild", json={"model": "m"}).json()["id"]
            for _ in range(200):
                body = client.get(f"/admin/jobs/{job_id}").json()
                if body["state"] != "running":
                    break
        assert body["state"] == "succeeded"
        assert body["result"] == {"updated": 3}

    def test_a_failed_job_records_why_instead_of_hanging_on_running(self, client):
        with patch("rag.admin.backup.rebuild_embeddings", side_effect=RuntimeError("no such model")):
            job_id = client.post("/admin/rebuild", json={"model": "m"}).json()["id"]
            for _ in range(200):
                body = client.get(f"/admin/jobs/{job_id}").json()
                if body["state"] != "running":
                    break
        assert body["state"] == "failed"
        assert "no such model" in body["error"]

    def test_unknown_job_is_404(self, client):
        assert client.get("/admin/jobs/nope").status_code == 404


class TestBindingIsLocalOnly:
    def test_serve_defaults_to_loopback(self):
        """There is no auth anywhere; the bind address is the security model."""
        from cli.main import build_parser

        args = build_parser().parse_args(["serve"])
        assert args.host == "127.0.0.1"
