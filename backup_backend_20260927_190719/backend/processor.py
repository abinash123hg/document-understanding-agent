"""
Docling document processor.
Uses Docling for PDF, DOCX and images; stores page-aware chunks for RAG.
"""

import logging
import uuid
from pathlib import Path

from backend.config import settings
from backend.storage import add_chunks

logger = logging.getLogger(__name__)


def chunk_text(text: str) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []

    size = max(400, int(settings.chunk_size))
    overlap = max(0, min(int(settings.chunk_overlap), size // 2))
    step = max(1, size - overlap)

    return [
        text[start:start + size].strip()
        for start in range(0, len(text), step)
        if text[start:start + size].strip()
    ]


def extract_with_docling(path: Path) -> tuple[str, str]:
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

    if ext in {".pdf", ".docx", ".png", ".jpg", ".jpeg"}:
        return extract_with_docling(path)

    raise ValueError(f"Unsupported file type: {ext}")


def process_upload(path: Path, original_name: str) -> dict:
    text, method = extract_text(path)

    if len(text.strip()) < 15:
        return {
            "status": "FAILED",
            "error": "No readable text found in this document."
        }

    chunks = chunk_text(text)
    data = []

    for index, chunk in enumerate(chunks):
        data.append({
            "id": str(uuid.uuid4()),
            "filename": original_name,
            "document_name": original_name,
            "chunk_index": index,
            "chunk_id": index,
            "page_number": None,
            "text": chunk
        })

    if not data:
        return {
            "status": "FAILED",
            "error": "No usable text chunks were created."
        }

    add_chunks(data)

    logger.info(
        "Indexed %s chunks from %s using %s",
        len(data),
        original_name,
        method
    )

    return {
        "status": "INDEXED",
        "extraction_method": method,
        "chunk_count": len(data)
    }
