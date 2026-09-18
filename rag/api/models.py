"""Request and response shapes for the local API.

These mirror the CLI's flags rather than the internal dataclasses, so that
`rag ask --course Maths` and `POST /ask {"course": "Maths"}` stay recognisably
the same thing.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Language = Literal["en", "it", "auto"]
DocType = Literal["pdf", "docx", "pptx", "md", "txt", "html", "csv", "epub", "other"]


class Filters(BaseModel):
    """The metadata filters every read endpoint accepts.

    All optional. A field left out means "do not filter on this", which is the
    distinction #40 got wrong in the BM25 store: a None there was read as
    "must equal None" and quietly excluded everything.
    """

    course: str | None = None
    unit: str | None = None
    language: Language | None = None
    doc_type: DocType | None = None
    author: str | None = None
    semester: str | None = None
    tags: list[str] | None = None


class AskRequest(Filters):
    question: str = Field(min_length=1)
    top_k: int = Field(default=8, ge=1, le=100)
    hybrid: bool = True
    forced_subject: str | None = None


class PreviewRequest(Filters):
    question: str = Field(min_length=1)
    top_k: int = Field(default=8, ge=1, le=100)
    hybrid: bool = True


class DeleteRequest(Filters):
    """Delete by explicit ids, by source path, or by filter -- as `rag delete`.

    dry_run defaults to True. Deleting a corpus is not something to do because
    a field was left out of a JSON body.
    """

    ids: list[str] | None = None
    path: str | None = None
    dry_run: bool = True


class RestoreRequest(BaseModel):
    path: str
    batch_size: int = Field(default=256, ge=1, le=4096)


class DumpRequest(BaseModel):
    path: str
    batch_size: int = Field(default=256, ge=1, le=4096)
    include_embedding_checksum: bool = True


class RebuildRequest(BaseModel):
    model: str = Field(min_length=1)
    batch_size: int = Field(default=256, ge=1, le=4096)


class JobResponse(BaseModel):
    id: str
    operation: str
    state: str
    started_at: str
    finished_at: str | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
