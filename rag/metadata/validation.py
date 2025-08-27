"""
Pydantic-based metadata validation for CLASSMATE-RAG (CLI boundary).

Goals
- Enforce allowed types (strings for scalar fields, list[str] for tags).
- Disallow empty strings (convert to None; optionally error if not fixup).
- Normalize language/doc_type values to supported enums (en/it/auto and known types).
- Default doc_type to inferred (never "other" by accident) when ingesting a file.
- Normalize and (optionally) slug tags.

Usage
- validate_cli_metadata(raw: dict, *, fixup: bool = False,
                        inferred_doc_type: str | None = None,
                        explicit_doc_type: bool = False) -> dict
  Returns a clean dict compatible with normalize_cli_metadata().

Notes
- This module performs *validation and normalization*. The final enum coercion is still
  handled by rag.metadata.normalize_cli_metadata().
"""

from __future__ import annotations

import re
from typing import List, Optional, Dict, Any

try:
    # Pydantic v1 (most common). If v2 is used, BaseModel import path is the same via shim.
    from pydantic import BaseModel, validator, root_validator
except Exception as e:  # pragma: no cover
    raise ImportError(
        "pydantic is required for metadata validation. Please add 'pydantic>=1.10,<3' to requirements.txt"
    ) from e


# ---- helpers ----

_DOC_TYPES = {"pdf", "docx", "pptx", "md", "txt", "html", "csv", "other"}
_LANGS = {"en", "it", "auto"}

_slug_re = re.compile(r"[^a-z0-9]+")

def _slug_tag(t: str) -> str:
    s = (t or "").lower().strip()
    s = _slug_re.sub("_", s)
    s = s.strip("_")
    return s

def _clean_str(v: Optional[str]) -> Optional[str]:
    if v is None:
        return None
    v2 = str(v).strip()
    return v2 or None

def _norm_lang(v: Optional[str]) -> Optional[str]:
    if v is None:
        return None
    v = v.strip().lower()
    if v in {"en", "eng", "english"}:
        return "en"
    if v in {"it", "ita", "italian", "italiano"}:
        return "it"
    if v in {"auto", "detect", "auto-detect"}:
        return "auto"
    return None  # unknown -> handled by validators

def _norm_doc_type(v: Optional[str]) -> Optional[str]:
    if v is None:
        return None
    v = v.strip().lower()
    if v in _DOC_TYPES:
        return v
    mapping = {
        "ppt": "pptx",
        "markdown": "md",
        "text": "txt",
        "htm": "html",
    }
    return mapping.get(v, None)


# ---- models ----

class _MetaInput(BaseModel):
    course: Optional[str] = None
    unit: Optional[str] = None
    language: Optional[str] = None
    doc_type: Optional[str] = None
    author: Optional[str] = None
    semester: Optional[str] = None
    tags: Optional[List[str]] = None  # already split; CLI may pass comma string -> we’ll split earlier

    class Config:
        anystr_strip_whitespace = True

    @validator("course", "unit", "author", "semester", pre=True, always=True)
    def _trim_or_none(cls, v):
        return _clean_str(v)

    @validator("language", pre=True, always=True)
    def _language_norm(cls, v):
        v2 = _norm_lang(v)
        if v is None:
            return None
        if v2 is None:
            # unknown language -> reject here; caller may choose fixup path
            raise ValueError(f"unsupported language '{v}' (allowed: en/it/auto)")
        return v2

    @validator("doc_type", pre=True, always=True)
    def _doc_type_norm(cls, v):
        if v is None:
            return None
        v2 = _norm_doc_type(v)
        if v2 is None:
            raise ValueError(f"unsupported doc_type '{v}' (allowed: {sorted(_DOC_TYPES)})")
        return v2

    @validator("tags", pre=True, always=True)
    def _ensure_tags_list(cls, v):
        if v is None:
            return None
        # Accept list or comma-separated string
        if isinstance(v, str):
            parts = [p.strip() for p in v.split(",")]
            arr = [p for p in parts if p]
        else:
            arr = [str(x).strip() for x in list(v)]
            arr = [x for x in arr if x]
        return arr or None

    @root_validator
    def _disallow_empty_strings(cls, values):
        # Any field that became "" should be None already; enforce no empty strings remain.
        for k, v in values.items():
            if isinstance(v, str) and not v.strip():
                values[k] = None
        return values


def validate_cli_metadata(
    raw: Dict[str, Any],
    *,
    fixup: bool = False,
    inferred_doc_type: Optional[str] = None,
    explicit_doc_type: bool = False,
) -> Dict[str, Any]:
    """
    Validate and normalize a metadata dict.

    Behavior:
    - If 'language' not provided -> None (later normalize_cli_metadata will map None->auto).
    - If invalid language/doc_type:
        * fixup=False -> raise ValueError
        * fixup=True  -> coerce to sensible defaults (language->auto, doc_type->inferred or None)
    - doc_type rules:
        * If explicit_doc_type=True and provided invalid -> error (even with fixup).
        * If doc_type is missing or 'other' but an inferred_doc_type is given -> use inferred.
        * Never auto-set to 'other' unless explicitly provided.
    - tags:
        * fixup=False -> must be alnum/underscore/hyphen only; else error.
        * fixup=True  -> slugify to lowercase snake_case.

    Returns a cleaned dict with keys: course, unit, language, doc_type, author, semester, tags
    """
    try:
        data = _MetaInput(**raw).dict()
    except Exception as e:
        if not fixup:
            # re-raise clearly
            raise
        # attempt a permissive path:
        data = {
            "course": _clean_str(raw.get("course")),
            "unit": _clean_str(raw.get("unit")),
            "language": _norm_lang(raw.get("language")) or "auto",
            "doc_type": _norm_doc_type(raw.get("doc_type")),
            "author": _clean_str(raw.get("author")),
            "semester": _clean_str(raw.get("semester")),
            "tags": None,
        }
        # tags permissive split
        tv = raw.get("tags")
        if tv is not None:
            if isinstance(tv, str):
                arr = [p.strip() for p in tv.split(",") if p.strip()]
            else:
                arr = [str(x).strip() for x in list(tv) if str(x).strip()]
            data["tags"] = arr or None

    # ---- fixup passes ----

    # language default on fixup
    if fixup and (data.get("language") is None):
        data["language"] = "auto"

    # doc_type inference & constraints
    dt = data.get("doc_type")
    if dt is None or dt == "other":
        if inferred_doc_type:
            data["doc_type"] = inferred_doc_type
        else:
            # if explicitly requested 'other', keep it; otherwise leave None and let normalize handle
            if explicit_doc_type and dt == "other":
                data["doc_type"] = "other"
            else:
                data["doc_type"] = None

    # tag normalization
    tags = data.get("tags")
    if tags:
        out: List[str] = []
        for t in tags:
            if fixup:
                s = _slug_tag(t)
                if s:
                    out.append(s)
            else:
                # enforce conservative charset without fixup
                if re.fullmatch(r"[A-Za-z0-9_\-]+", t):
                    out.append(t)
                else:
                    raise ValueError(f"invalid tag '{t}'; use letters, numbers, '_' or '-' (or pass --fixup)")
        # dedupe preserving order
        seen = set()
        clean_tags = []
        for t in out:
            if t not in seen:
                seen.add(t)
                clean_tags.append(t)
        data["tags"] = clean_tags or None

    return data
