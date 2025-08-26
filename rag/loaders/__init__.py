from .pptx_loader import load_pptx_slides
from .pdf_loader import load_pdf_pages
from .docx_loader import load_docx_blocks
from .text_loader import load_txt_text, load_md_text
from .html_loader import load_html_text
from .csv_loader import load_csv_text

from pathlib import Path


__all__ = [
    "load_pptx_slides",
    "load_pdf_pages",
    "load_docx_blocks",
    "load_txt_text",
    "load_md_text",
    "load_html_text",
    "load_csv_text",
    "infer_doc_type_from_path",
    "load_document_by_type",
]


def infer_doc_type_from_path(path: str | Path) -> str:
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        return "pdf"
    if ext == ".docx":
        return "docx"
    if ext in (".pptx", ".ppt"):
        return "pptx"
    if ext in (".md", ".markdown"):
        return "md"
    if ext == ".txt":
        return "txt"
    if ext in (".html", ".htm"):
        return "html"
    if ext == ".csv":
        return "csv"
    return "other"


def load_document_by_type(path: str | Path, doc_type: str, *, enable_ocr: bool = False):
    """
    Dispatch to the appropriate loader and return a list of (page_index, text) tuples.
    'page_index' is 1-based when applicable; for formats without pages we return a single (1, text).
    """
    dt = doc_type.lower()
    if dt == "pdf":
        return load_pdf_pages(path, enable_ocr=enable_ocr)
    if dt == "docx":
        return load_docx_blocks(path)
    if dt == "pptx":
        return load_pptx_slides(path)
    if dt == "md":
        return [(1, load_md_text(path))]
    if dt == "txt":
        return [(1, load_txt_text(path))]
    if dt == "html":
        return [(1, load_html_text(path))]
    if dt == "csv":
        return [(1, load_csv_text(path))]
    # Fallback: treat as plain text
    return [(1, load_txt_text(path))]
