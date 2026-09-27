"""
Extract text from documents → clean → chunk → store
Supports PDF, DOCX, TXT, MD and images (OCR)
"""

import logging
import uuid
from pathlib import Path

from backend.config import settings
from backend.storage import add_chunks

logger = logging.getLogger(__name__)
MIN_CHARS = 25   # if digital text is shorter than this → treat as scanned


def extract_text(path: Path) -> tuple[str, str]:
    """
    Extract text from file.
    Returns: (text, method) where method is 'digital' or 'ocr'
    """
    ext = path.suffix.lower()

    # Plain text
    if ext in {".txt", ".md"}:
        text = path.read_text(encoding="utf-8", errors="ignore")
        return text, "digital"

    # Word document
    if ext == ".docx":
        from docx import Document
        doc = Document(str(path))
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        return "\n".join(paragraphs), "digital"

    # Images → OCR
    if ext in {".png", ".jpg", ".jpeg"}:
        from backend.ocr import ocr_image
        return ocr_image(path), "ocr"

    # PDF
    if ext == ".pdf":
        import fitz  # PyMuPDF

        doc = fitz.open(str(path))
        digital_text = "\n".join(page.get_text("text").strip() for page in doc)

        if len(digital_text.strip()) >= MIN_CHARS:
            return digital_text, "digital"

        # Faster OCR settings for scanned PDFs
        from backend.ocr import get_reader
        reader = get_reader()
        pages = []

        for i, page in enumerate(doc):
            logger.info(f"OCR page {i + 1}")
            # Lower DPI = much faster (150 is good balance)
            pix = page.get_pixmap(dpi=150)
            img_bytes = pix.tobytes("png")

            results = reader.readtext(
                img_bytes,
                detail=0,
                paragraph=True,
                batch_size=1
            )
            text = "\n".join([t.strip() for t in results if t.strip()])
            if text:
                pages.append(text)

        return "\n\n".join(pages), "ocr"

    raise ValueError(f"Unsupported file type: {ext}")


def chunk_text(text: str) -> list[str]:
    """Split text into overlapping chunks"""
    chunks = []
    step = settings.chunk_size - settings.chunk_overlap

    for i in range(0, len(text), step):
        piece = text[i : i + settings.chunk_size].strip()
        if piece:
            chunks.append(piece)

    return chunks


def process_upload(path: Path, original_name: str) -> dict:
    """
    Full pipeline:
    extract text → chunk → store
    """
    text, method = extract_text(path)

    if len(text.strip()) < 15:
        return {
            "status": "FAILED",
            "error": "No readable text found in this document."
        }

    chunks = chunk_text(text)

    data = []
    for i, t in enumerate(chunks):
        data.append({
            "id": str(uuid.uuid4()),
            "filename": original_name,
            "document_name": original_name,
            "chunk_index": i,
            "chunk_id": i,
            "text": t
        })

    add_chunks(data)

    logger.info(
        f"Indexed {len(chunks)} chunks from {original_name} ({method})"
    )

    return {
        "status": "INDEXED",
        "extraction_method": method,
        "chunk_count": len(chunks)
    }