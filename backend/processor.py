"""
Document processor.

Fast, document-agnostic extraction:
- Searchable PDFs: PyMuPDF per-page text extraction.
- Scanned or image-only PDFs: Docling fallback.
- DOCX and image files: Docling.
- Chunks retain page numbers and prefer natural text boundaries.
"""

import logging
import re
import uuid
from pathlib import Path

from backend.config import settings
from backend.storage import add_chunks, clear_document

logger = logging.getLogger(__name__)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def chunk_page_text(text: str) -> list[str]:
    """
    Create chunks that favor paragraph and sentence endings. This prevents
    avoidable mid-sentence splits while respecting configured size/overlap.
    """
    text = normalize_text(text)
    if not text:
        return []

    size = max(400, int(settings.chunk_size))
    overlap = max(0, min(int(settings.chunk_overlap), size // 2))
    chunks = []
    start = 0
    total = len(text)

    while start < total:
        end = min(total, start + size)

        if end < total:
            window = text[start:end]
            choices = [
                window.rfind(". "),
                window.rfind("? "),
                window.rfind("! "),
                window.rfind("; "),
                window.rfind(": "),
            ]
            boundary = max(choices)
            if boundary >= max(120, int(size * 0.45)):
                end = start + boundary + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= total:
            break

        next_start = max(start + 1, end - overlap)
        if next_start <= start:
            next_start = end
        start = next_start

    return chunks


def extract_pdf_pages_fast(path: Path) -> list[tuple[int, str]]:
    """Extract embedded PDF text per page without OCR."""
    try:
        import fitz

        document = fitz.open(str(path))
        try:
            pages = []
            for number, page in enumerate(document, start=1):
                text = page.get_text("text").strip()
                if text:
                    pages.append((number, text))
            return pages
        finally:
            document.close()
    except Exception as error:
        logger.warning("Fast PDF extraction failed for %s: %s", path.name, error)
        return []


def extract_with_docling(path: Path) -> tuple[str, str]:
    """Fallback for scanned PDFs and normal conversion for DOCX/images."""
    from docling.document_converter import DocumentConverter

    converter = DocumentConverter()
    result = converter.convert(str(path))
    document = result.document

    try:
        text = document.export_to_markdown()
    except Exception:
        text = document.export_to_text()

    return (text or "").strip(), "docling"


def extract_text(path: Path) -> tuple[str, str]:
    ext = path.suffix.lower()

    if ext in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="ignore"), "digital"

    if ext == ".pdf":
        pages = extract_pdf_pages_fast(path)
        text = "\n".join(page_text for _, page_text in pages).strip()

        if len(text) >= 100:
            return text, "pymupdf-fast"

        return extract_with_docling(path)

    if ext in {".docx", ".png", ".jpg", ".jpeg"}:
        return extract_with_docling(path)

    raise ValueError(f"Unsupported file type: {ext}")


def process_upload(path: Path, original_name: str) -> dict:
    ext = path.suffix.lower()
    data = []
    method = "unknown"

    if ext == ".pdf":
        pages = extract_pdf_pages_fast(path)
        page_text_total = sum(len(text.strip()) for _, text in pages)

        if page_text_total >= 100:
            method = "pymupdf-fast"

            for page_number, page_text in pages:
                for index, chunk in enumerate(chunk_page_text(page_text)):
                    data.append({
                        "id": str(uuid.uuid4()),
                        "filename": original_name,
                        "document_name": original_name,
                        "chunk_index": len(data),
                        "chunk_id": f"{page_number}-{index}",
                        "page_number": page_number,
                        "text": chunk,
                    })
        else:
            text, method = extract_with_docling(path)
            for index, chunk in enumerate(chunk_page_text(text)):
                data.append({
                    "id": str(uuid.uuid4()),
                    "filename": original_name,
                    "document_name": original_name,
                    "chunk_index": index,
                    "chunk_id": index,
                    "page_number": None,
                    "text": chunk,
                })
    else:
        text, method = extract_text(path)
        for index, chunk in enumerate(chunk_page_text(text)):
            data.append({
                "id": str(uuid.uuid4()),
                "filename": original_name,
                "document_name": original_name,
                "chunk_index": index,
                "chunk_id": index,
                "page_number": None,
                "text": chunk,
            })

    if not data:
        return {
            "status": "FAILED",
            "error": "No readable text found in this document.",
        }

    clear_document(original_name)
    add_chunks(data)

    logger.info(
        "Indexed %s chunks from %s using %s",
        len(data),
        original_name,
        method,
    )

    return {
        "status": "INDEXED",
        "extraction_method": method,
        "chunk_count": len(data),
    }
