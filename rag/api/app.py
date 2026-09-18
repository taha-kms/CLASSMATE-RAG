"""The local HTTP API.

One process serves the JSON endpoints and, when it has been built, the
frontend. There is no authentication and none is planned: every student runs
their own copy against their own corpus, on their own machine, and `rag serve`
binds to 127.0.0.1 for that reason. The delete and admin endpoints are
destructive and are reachable by anyone who can reach the port, so the bind
address is the whole security model. Do not move it to 0.0.0.0.

Handlers are deliberately `def`, not `async def`. StickyModelLoader acquires
the generation lock *before* it returns the token generator, and holds it until
that generator is exhausted. On the event loop that would block every other
request in the process, including the ones that never touch the model; in
FastAPI's threadpool it behaves the way the lock was designed for.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from rag.api.jobs import JobRegistry
from rag.api.models import (
    AskRequest,
    DeleteRequest,
    DumpRequest,
    Filters,
    PreviewRequest,
    RebuildRequest,
    RestoreRequest,
)
from rag.bootstrap import project_root

log = logging.getLogger(__name__)


def _filter_mapping(f: Filters) -> dict[str, object]:
    """Drop unset fields, so a missing filter means "do not filter".

    Passing None through would reinstate #40, where a None in the `where`
    mapping was compared for equality and matched nothing.
    """
    return {k: v for k, v in f.model_dump().items() if v is not None}


def _to_jsonable(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: _to_jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _sse(event: str, payload: Any) -> str:
    """One Server-Sent Event.

    The named event goes on its own line so the browser can use
    `addEventListener("token", ...)` rather than switching on a field inside
    the payload.
    """
    return f"event: {event}\ndata: {json.dumps(_to_jsonable(payload), ensure_ascii=False)}\n\n"


def create_app() -> FastAPI:
    app = FastAPI(
        title="CLASSMATE-RAG",
        description="Local API over your own course materials. No authentication: bind to localhost only.",
        version="1.0.0",
    )
    jobs = JobRegistry()

    # ---------------------------------------------------------------- ask

    @app.post("/ask")
    def ask(req: AskRequest) -> StreamingResponse:
        """Answer a question, streamed as it is produced."""
        from rag.generation.stream import FinalEvent, ReplaceEvent, StageEvent, TokenEvent
        from rag.metadata import normalize_cli_metadata
        from rag.pipeline import ask_question_stream

        filters = normalize_cli_metadata(**_filter_mapping(Filters(**req.model_dump(exclude_unset=False))))

        def events():
            # A "waiting" stage arrives here like any other. The check that
            # produces it lives in the pipeline, immediately before generation:
            # retrieval runs first and can take seconds, so checking the lock
            # at the top of this generator reported it free and said nothing.
            try:
                for event in ask_question_stream(
                    question=req.question,
                    filters=filters,
                    top_k=req.top_k,
                    hybrid=req.hybrid,
                    forced_subject=req.forced_subject,
                ):
                    if isinstance(event, StageEvent):
                        yield _sse("stage", {"stage": event.stage, "message": event.message})
                    elif isinstance(event, TokenEvent):
                        yield _sse("token", {"text": event.text})
                    elif isinstance(event, ReplaceEvent):
                        yield _sse("replace", {"reason": event.reason})
                    elif isinstance(event, FinalEvent):
                        yield _sse("final", event.result)
            except Exception as e:  # noqa: BLE001 - the response has already begun
                # Headers went out with the first event, so raising here gives
                # the client a truncated stream and no reason. An error event
                # is the only way left to say what happened.
                log.exception("ask stream failed")
                yield _sse("error", {"error": f"{type(e).__name__}: {e}"})

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                # nginx and friends buffer text/event-stream by default, which
                # defeats the point of streaming.
                "X-Accel-Buffering": "no",
            },
        )

    # ------------------------------------------------------------ preview

    @app.post("/preview")
    def preview(req: PreviewRequest) -> dict[str, Any]:
        """Retrieval only, with scores. No model is loaded."""
        from rag.admin import retrieve_preview

        results = retrieve_preview(
            question=req.question,
            filters=_filter_mapping(Filters(**req.model_dump(exclude_unset=False))),
            top_k=req.top_k,
            hybrid=req.hybrid,
        )
        return {"question": req.question, "results": _to_jsonable(results)}

    # ------------------------------------------------------------- ingest

    @app.post("/ingest", status_code=201)
    def ingest(
        file: Annotated[UploadFile, File()],
        course: str | None = None,
        unit: str | None = None,
        language: str | None = None,
        doc_type: str | None = None,
        author: str | None = None,
        semester: str | None = None,
        tags: str | None = None,
    ) -> dict[str, Any]:
        """Ingest an uploaded file. Fields mirror `rag add`."""
        from rag.loaders import infer_doc_type_from_path
        from rag.metadata import normalize_cli_metadata
        from rag.metadata.validation import validate_cli_metadata
        from rag.pipeline import ingest_file

        # .name alone defeats a traversal attempt: "../../etc/passwd" reduces
        # to "passwd", which then fails the suffix check below.
        name = Path(file.filename or "").name
        if not Path(name).suffix:
            raise HTTPException(
                status_code=422,
                detail="The upload needs a filename with an extension: the loader is chosen from it.",
            )

        # Uploads are kept, not streamed through a temp file. source_path goes
        # into every chunk's metadata, and `rag reingest`, `list --path` and
        # `delete --path` all read it back; a path under /tmp that no longer
        # exists makes those silently useless.
        uploads = project_root() / "data" / "uploads"
        uploads.mkdir(parents=True, exist_ok=True)
        target = uploads / name
        with target.open("wb") as out:
            while chunk := file.file.read(1024 * 1024):
                out.write(chunk)

        # doc_type has to be inferred from the name when the caller did not say,
        # exactly as `rag add` does. Going through normalize_cli_metadata instead
        # types every upload as "other": the enum defaults to DocTypeEnum.other,
        # which is truthy, so ingest_file never falls back to inferring it.
        try:
            clean = validate_cli_metadata(
                {
                    "course": course,
                    "unit": unit,
                    "language": language,
                    "doc_type": doc_type,
                    "author": author,
                    "semester": semester,
                    "tags": tags,
                },
                # fixup=False to match `rag add` without --fixup: a bad tag or
                # language is reported rather than quietly rewritten.
                fixup=False,
                inferred_doc_type=infer_doc_type_from_path(target),
                explicit_doc_type=doc_type is not None,
            )
        except ValueError as e:
            target.unlink(missing_ok=True)
            raise HTTPException(status_code=422, detail=str(e)) from e

        meta = normalize_cli_metadata(
            course=clean.get("course"),
            unit=clean.get("unit"),
            language=clean.get("language"),
            doc_type=clean.get("doc_type"),
            author=clean.get("author"),
            semester=clean.get("semester"),
            tags=clean.get("tags"),
        )

        result = ingest_file(path=target, doc_meta=meta)
        return _to_jsonable(result)

    # ------------------------------------------------------------- chunks

    @app.get("/chunks")
    def list_chunks(
        course: str | None = None,
        unit: str | None = None,
        language: str | None = None,
        doc_type: str | None = None,
        author: str | None = None,
        semester: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        from rag.admin.manage import list_entries

        where = _filter_mapping(
            Filters(
                course=course,
                unit=unit,
                language=language,  # type: ignore[arg-type]
                doc_type=doc_type,  # type: ignore[arg-type]
                author=author,
                semester=semester,
            )
        )
        entries = list_entries(where=where, limit=limit, offset=offset)
        return {"count": len(entries), "offset": offset, "chunks": _to_jsonable(entries)}

    @app.get("/chunks/{chunk_id}")
    def show_chunk(chunk_id: str) -> dict[str, Any]:
        from rag.admin.manage import show_entries_by_id

        found = show_entries_by_id([chunk_id])
        if not found:
            raise HTTPException(status_code=404, detail=f"No chunk with id {chunk_id!r}")
        return _to_jsonable(found[0])

    @app.delete("/chunks")
    def delete_chunks(req: DeleteRequest) -> dict[str, Any]:
        """Delete by ids, by source path, or by filter. Dry run by default."""
        from rag.admin.manage import delete_by_ids, resolve_ids

        ids = resolve_ids(
            ids=req.ids,
            where=_filter_mapping(Filters(**req.model_dump(exclude_unset=False))),
            path=req.path,
        )
        if req.dry_run:
            return {"dry_run": True, "matched": len(ids), "ids": ids}
        n_bm25, n_vec = delete_by_ids(ids)
        return {"dry_run": False, "matched": len(ids), "deleted_bm25": n_bm25, "deleted_vector": n_vec}

    # -------------------------------------------------------------- stats

    @app.get("/stats")
    def stats() -> dict[str, Any]:
        from rag.admin import index_stats

        return _to_jsonable(index_stats())

    @app.get("/reconcile")
    def reconcile(apply: bool = False) -> dict[str, Any]:
        from rag.admin.manage import reconcile_stores

        return _to_jsonable(reconcile_stores(dry_run=not apply))

    # -------------------------------------------------------------- admin

    @app.post("/admin/vacuum")
    def vacuum() -> dict[str, Any]:
        """Fast enough to answer inline, unlike its neighbours."""
        from rag.admin.backup import vacuum_indexes

        return _to_jsonable(vacuum_indexes())

    @app.post("/admin/dump", status_code=202)
    def dump(req: DumpRequest) -> dict[str, Any]:
        from rag.admin.backup import dump_index

        job = jobs.submit(
            "dump",
            lambda: {
                "written": dump_index(
                    req.path,
                    include_embedding_checksum=req.include_embedding_checksum,
                    batch_size=req.batch_size,
                ),
                "path": req.path,
            },
        )
        return job.as_dict()

    @app.post("/admin/restore", status_code=202)
    def restore(req: RestoreRequest) -> dict[str, Any]:
        from rag.admin.backup import restore_dump

        job = jobs.submit(
            "restore",
            lambda: {"restored": restore_dump(req.path, batch_size=req.batch_size), "path": req.path},
        )
        return job.as_dict()

    @app.post("/admin/rebuild", status_code=202)
    def rebuild(req: RebuildRequest) -> dict[str, Any]:
        from rag.admin.backup import rebuild_embeddings

        job = jobs.submit(
            "rebuild",
            lambda: _to_jsonable(rebuild_embeddings(req.model, batch_size=req.batch_size)),
        )
        return job.as_dict()

    @app.get("/admin/jobs")
    def list_jobs() -> dict[str, Any]:
        return {"jobs": [j.as_dict() for j in jobs.all()]}

    @app.get("/admin/jobs/{job_id}")
    def get_job(job_id: str) -> dict[str, Any]:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"No job with id {job_id!r}")
        return job.as_dict()

    # ----------------------------------------------------------- frontend

    _mount_frontend(app)
    return app


def _mount_frontend(app: FastAPI) -> None:
    """Serve the built frontend, if it has been built.

    Mounted last so it cannot shadow an API route, and only when the directory
    exists: the API has to start on a machine where `npm run build` has never
    been run, which is every machine before the first build.
    """
    dist = project_root() / "web" / "dist"
    if not (dist / "index.html").is_file():
        log.info("No built frontend at %s; serving the API only.", dist)
        return

    from fastapi.staticfiles import StaticFiles

    app.mount("/", StaticFiles(directory=str(dist), html=True), name="frontend")


app = create_app()
