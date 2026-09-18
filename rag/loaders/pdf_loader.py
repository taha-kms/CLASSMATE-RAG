"""
PDF loader.

- Primary path: extract per-page text via pypdf (fast, no external deps).
- Optional OCR path: for pages with no text (scanned PDFs), try pdf2image + pytesseract.
  Note: OCR requires system dependencies (poppler for pdf2image, Tesseract engine).
"""

from __future__ import annotations

import logging
from pathlib import Path

from pypdf import PdfReader

from rag.utils.text import normalize_text

log = logging.getLogger(__name__)


def _ocr_page_images_to_text(pdf_path: Path, page_index_zero: int, ocr_lang: str = "eng+ita") -> str:
    """
    Best-effort OCR for a single page.
    Requires pdf2image + poppler and pytesseract + tesseract installed.
    Returns empty string if OCR unavailable or fails.
    """
    try:
        import pytesseract
        from pdf2image import convert_from_path
    except ImportError:
        log.debug("OCR extras not installed; skipping OCR for this page")
        return ""

    try:
        # Convert the specific page to image(s)
        images = convert_from_path(str(pdf_path), first_page=page_index_zero + 1, last_page=page_index_zero + 1)
        texts = []
        for img in images:
            txt = pytesseract.image_to_string(img, lang=ocr_lang)
            texts.append(txt or "")
        return normalize_text("\n".join(texts))
    except Exception:  # noqa: BLE001 - poppler/tesseract are external binaries
        # OCR reaches outside Python entirely, so the failure modes are open
        # ended: a missing binary, an unsupported language pack, a page that
        # renders to nothing. OCR is already best-effort, so log and move on.
        log.warning("OCR failed for page %d of %s", page_index_zero + 1, pdf_path, exc_info=True)
        return ""


def load_pdf_pages(path: str | Path, enable_ocr: bool = False, ocr_lang: str = "eng+ita") -> list[tuple[int, str]]:
    p = Path(path)
    if not p.exists() or p.suffix.lower() != ".pdf":
        raise ValueError(f"Expected an existing .pdf file, got: {path}")

    reader = PdfReader(str(p))
    out: list[tuple[int, str]] = []

    for i, page in enumerate(reader.pages):
        try:
            txt = page.extract_text() or ""
        except Exception:  # noqa: BLE001 - corrupt PDFs raise almost anything
            # Narrowing this to PyPdfError would regress robustness: damaged
            # PDFs surface as KeyError, TypeError and friends from deep inside
            # the object graph. One unreadable page should not cost the file.
            log.warning("Could not extract text from page %d of %s", i + 1, p, exc_info=True)
            txt = ""
        txt = normalize_text(txt)
        if enable_ocr and not txt:
            # Try OCR if no text extracted
            txt = _ocr_page_images_to_text(p, i, ocr_lang=ocr_lang)
        out.append((i + 1, txt))

    return out
