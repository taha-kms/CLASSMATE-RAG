"""
Metadata schema for CLASSMATE-RAG (CLI-only flow).

Defines:
- accepted metadata fields,
- enums for language and doc type,
- dataclasses for document- and chunk-level metadata,
- CLI normalization helpers.

Now includes support for PowerPoint (.pptx).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, List, Optional, Tuple


class LanguageEnum(str, Enum):
    en = "en"
    it = "it"
    auto = "auto"  # CLI accepts; replaced per-chunk when detection is enabled


class DocTypeEnum(str, Enum):
    pdf = "pdf"
    docx = "docx"
    pptx = "pptx"
    md = "md"
    txt = "txt"
    html = "html"
    csv = "csv"
    other = "other"


# Keep stable for downstream filters
METADATA_FIELDS: Tuple[str, ...] = (
    "course",
    "unit",
    "language",
    "doc_type",
    "author",
    "semester",
    "tags",
    "source_path",
    "page",
    "chunk_id",
    "created_at",
)


@dataclass(frozen=True)
class DocumentMetadata:
    course: Optional[str] = None
    unit: Optional[str] = None
    language: LanguageEnum = LanguageEnum.auto
    doc_type: DocTypeEnum = DocTypeEnum.other
    author: Optional[str] = None
    semester: Optional[str] = None
    tags: Optional[List[str]] = None
    source_path: Optional[str] = None
    created_at: Optional[str] = None

    def to_dict(self) -> Dict:
        d = asdict(self)
        if d.get("tags") is None:
            d.pop("tags", None)
        return d


@dataclass(frozen=True)
class ChunkMetadata:
    course: Optional[str] = None
    unit: Optional[str] = None
    language: LanguageEnum = LanguageEnum.auto
    doc_type: DocTypeEnum = DocTypeEnum.other
    author: Optional[str] = None
    semester: Optional[str] = None
    tags: Optional[List[str]] = None
    source_path: Optional[str] = None
    page: Optional[int] = None
    chunk_id: Optional[int] = None
    created_at: Optional[str] = None

    def to_dict(self) -> Dict:
        d = asdict(self)
        if d.get("tags") is None:
            d.pop("tags", None)
        return d


# --- Normalization helpers ---

def _clean_str(v: Optional[str]) -> Optional[str]:
    if v is None:
        return None
    v2 = v.strip()
    return v2 or None


def _parse_tags(v: Optional[str | List[str]]) -> Optional[List[str]]:
    if v is None:
        return None
    if isinstance(v, list):
        tags = [t.strip() for t in v if isinstance(t, str) and t.strip()]
        return tags or None
    parts = [p.strip() for p in str(v).split(",")]
    tags = [p for p in parts if p]
    return tags or None


def _normalize_language(v: Optional[str]) -> LanguageEnum:
    if not v:
        return LanguageEnum.auto
    v = v.strip().lower()
    if v in ("en", "eng", "english"):
        return LanguageEnum.en
    if v in ("it", "ita", "italian", "italiano"):
        return LanguageEnum.it
    if v in ("auto", "detect", "auto-detect"):
        return LanguageEnum.auto
    return LanguageEnum.auto


def _normalize_doc_type(v: Optional[str]) -> DocTypeEnum:
    if not v:
        return DocTypeEnum.other
    v = v.strip().lower()
    mapping = {
        "pdf": DocTypeEnum.pdf,
        "docx": DocTypeEnum.docx,
        "pptx": DocTypeEnum.pptx,
        "ppt": DocTypeEnum.pptx,  # allow legacy extension
        "md": DocTypeEnum.md,
        "markdown": DocTypeEnum.md,
        "txt": DocTypeEnum.txt,
        "text": DocTypeEnum.txt,
        "html": DocTypeEnum.html,
        "htm": DocTypeEnum.html,
        "csv": DocTypeEnum.csv,
    }
    return mapping.get(v, DocTypeEnum.other)


def normalize_cli_metadata(
    *,
    course: Optional[str] = None,
    unit: Optional[str] = None,
    language: Optional[str] = None,
    doc_type: Optional[str] = None,
    author: Optional[str] = None,
    semester: Optional[str] = None,
    tags: Optional[str | List[str]] = None,
) -> DocumentMetadata:
    """
    Normalize CLI-provided metadata for 'add' (ingest) or 'ask' (filters).
    - trims strings
    - normalizes enums
    - parses tag list
    """
    lang_enum = _normalize_language(language)
    dt_enum = _normalize_doc_type(doc_type)
    tag_list = _parse_tags(tags)
    if tag_list:
        seen = set()
        norm = []
        for t in tag_list:
            tl = t.lower()
            if tl not in seen:
                seen.add(tl)
                norm.append(tl)
        tag_list = norm

    meta = DocumentMetadata(
        course=_clean_str(course),
        unit=_clean_str(unit),
        language=lang_enum,
        doc_type=dt_enum,
        author=_clean_str(author),
        semester=_clean_str(semester),
        tags=tag_list or None,
        source_path=None,
        created_at=None,
    )
    return meta
