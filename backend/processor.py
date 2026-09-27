"""
Large-document processor: page-aware chunks for PDFs, DOCX, TXT, MD, and images.
Designed for 500+ page PDFs. Stores page_number with every PDF chunk.
"""

import logging
import uuid
from pathlib import Path

from backend.config import settings
from backend.storage import add_chunks

logger = logging.getLogger(__name__)
MIN_CHARS = 25


def chunk_text(text: str) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []

    size = max(300, int(settings.chunk_size))
    overlap = max(0, min(int(settings.chunk_overlap), size // 2))
    step = max(1, size - overlap)

    return [
        text[start:start + size].strip()
        for start in range(0, len(text), step)
        if text[start:start + size].strip()
    ]


def extract_pdf_pages(path: Path) -> tuple[list[tuple[int, str]], str]:
    import fitz

    doc = fitz.open(str(path))
    pages = []
    digital_chars = 0

    for index, page in enumerate(doc):
        text = page.get_text("text", sort=True).strip()
        digital_chars += len(text)
        if text:
            pages.append((index + 1, text))

    if digital_chars >= MIN_CHARS:
        return pages, "digital"

    from backend.ocr import get_reader
    reader = get_reader()
    pages = []

    for index, page in enumerate(doc):
        logger.info("OCR page %s", index + 1)
        pix = page.get_pixmap(dpi=150)
        lines = reader.readtext(
            pix.tobytes("png"),
            detail=0,
            paragraph=True,
            batch_size=1
        )
        text = "\n".join(str(line).strip() for line in lines if str(line).strip())
        if text:
            pages.append((index + 1, text))

    return pages, "ocr"


def extract_text(path: Path) -> tuple[str, str]:
    ext = path.suffix.lower()

    if ext in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="ignore"), "digital"

    if ext == ".docx":
        from docx import Document
        doc = Document(str(path))
        return "\n".join(p.text.strip() for p in doc.paragraphs if p.text.strip()), "digital"

    if ext in {".png", ".jpg", ".jpeg"}:
        from backend.ocr import ocr_image
        return ocr_image(path), "ocr"

    raise ValueError(f"Unsupported file type: {ext}")


def process_upload(path: Path, original_name: str) -> dict:
    ext = path.suffix.lower()
    data = []
    method = "digital"

    if ext == ".pdf":
        pages, method = extract_pdf_pages(path)

        for page_number, page_text in pages:
            for chunk_position, text in enumerate(chunk_text(page_text)):
                data.append({
                    "id": str(uuid.uuid4()),
                    "filename": original_name,
                    "document_name": original_name,
                    "chunk_index": len(data),
                    "chunk_id": len(data),
                    "page_number": page_number,
                    "page_chunk_index": chunk_position,
                    "text": text
                })
    else:
        text, method = extract_text(path)

        for chunk_position, value in enumerate(chunk_text(text)):
            data.append({
                "id": str(uuid.uuid4()),
                "filename": original_name,
                "document_name": original_name,
                "chunk_index": chunk_position,
                "chunk_id": chunk_position,
                "page_number": None,
                "page_chunk_index": chunk_position,
                "text": value
            })

    if not data:
        return {
            "status": "FAILED",
            "error": "No readable text found in this document."
        }

    add_chunks(data)

    logger.info(
        "Indexed %s chunks from %s using %s extraction",
        len(data),
        original_name,
        method
    )

    return {
        "status": "INDEXED",
        "extraction_method": method,
        "chunk_count": len(data)
    }
