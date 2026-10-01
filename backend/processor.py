"""
Document processor: extraction, normalization and chunking.

Routing rules
- .txt / .md      -> read directly                       (digital_text)
- .docx           -> python-docx                         (digital_text)
- .pdf            -> embedded text per page when present (digital_text);
                     only pages with no usable text are rasterized and
                     recognised, which keeps normal PDFs near-instant
- .png/.jpg/.jpeg -> preprocessing + TrOCR               (handwritten_ocr)

Every chunk keeps its document name, page number, chunk id, extraction method
and content type, so citations can point at a real page.
"""

import logging
import re
import uuid
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from backend.config import settings
from backend.storage import add_chunks, clear_document

logger = logging.getLogger(__name__)

# A page needs at least this many embedded characters to be trusted as digital
# text. Below it the page is treated as scanned and sent to recognition.
EMBEDDED_TEXT_MIN_CHARS = 20
URL_ONLY_TEXT = re.compile(r"^(?:https?://|www\.)\S+$", re.IGNORECASE)

DIGITAL = "digital_text"
HANDWRITTEN = "handwritten_ocr"


def normalize_text(text: str) -> str:
    """
    Tidy whitespace without touching content.

    Line breaks are preserved on purpose: OCR output is line-structured, and
    collapsing it into one run hides line boundaries from the chunker. Numbers,
    formulas, names and units are never rewritten.
    """
    text = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_page_text(text: str) -> list[str]:
    """
    Split text into chunks that prefer paragraph and sentence boundaries.

    Loop safety is explicit: every iteration either consumes characters or
    forces the start forward, so overlapping windows can never spin.
    """
    text = normalize_text(text)
    if not text:
        return []

    size = max(400, int(settings.chunk_size))
    overlap = max(0, min(int(settings.chunk_overlap), size // 2))
    chunks: list[str] = []
    start = 0
    total = len(text)

    while start < total:
        end = min(total, start + size)

        if end < total:
            window = text[start:end]
            boundary = max(
                window.rfind("\n\n"),
                window.rfind(". "),
                window.rfind("? "),
                window.rfind("! "),
                window.rfind("; "),
                window.rfind(": "),
                window.rfind("\n"),
            )
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


def _pixmap_to_gray(pixmap) -> np.ndarray:
    samples = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
        pixmap.height, pixmap.width, pixmap.n
    )
    if pixmap.n == 1:
        return samples[:, :, 0].copy()
    if pixmap.n == 3:
        return cv2.cvtColor(samples, cv2.COLOR_RGB2GRAY)
    return cv2.cvtColor(samples, cv2.COLOR_RGBA2GRAY)


def _downscale(gray: np.ndarray) -> np.ndarray:
    limit = int(settings.pdf_max_dimension)
    height, width = gray.shape[:2]
    longest = max(height, width)

    if longest <= limit:
        return gray

    scale = limit / longest
    return cv2.resize(
        gray,
        (max(1, int(width * scale)), max(1, int(height * scale))),
        interpolation=cv2.INTER_AREA,
    )


def _open_pdf(path: Path):
    """Turn a broken file into a plain rejection instead of a stack trace."""
    import pymupdf

    try:
        return pymupdf.open(str(path))
    except Exception as error:
        raise ValueError(f"Could not open this PDF: {error}") from error


def extract_pdf_pages(path: Path) -> list[dict]:
    """
    Per-page extraction with real page numbers.

    Embedded text is read first because it is effectively free. A page only
    pays the rasterize-and-recognise cost when it has no usable text layer,
    which is what keeps ordinary PDF uploads fast.
    """
    import pymupdf

    from backend import handwriting

    pages: list[dict] = []
    ocr_pages = 0

    document = _open_pdf(path)
    try:
        raw_texts = [page.get_text("text").strip() for page in document]
        line_frequency = Counter()
        for raw_text in raw_texts:
            line_frequency.update({
                line.strip() for line in raw_text.splitlines() if line.strip()
            })
        boilerplate = {
            line for line, count in line_frequency.items()
            if count >= 2 or URL_ONLY_TEXT.fullmatch(line.rstrip(".,;:)]}"))
        }

        for number, (page, text) in enumerate(zip(document, raw_texts), start=1):
            usable_text = "\n".join(
                line for line in text.splitlines()
                if line.strip() and line.strip() not in boilerplate
            )

            if len(usable_text.strip()) >= EMBEDDED_TEXT_MIN_CHARS:
                pages.append({
                    "page_number": number,
                    "text": text,
                    "content_type": DIGITAL,
                    "method": "pymupdf",
                })
                continue

            if ocr_pages >= settings.pdf_max_ocr_pages:
                raise ValueError(
                    f"This PDF has more than {settings.pdf_max_ocr_pages} "
                    f"scanned pages. Split it into smaller files, or raise "
                    f"PDF_MAX_OCR_PAGES in your .env."
                )

            ocr_pages += 1
            pixmap = page.get_pixmap(dpi=settings.pdf_dpi, alpha=False)
            ocr_text, confidence = handwriting.recognize_array(
                _downscale(_pixmap_to_gray(pixmap))
            )
            logger.info(
                "Recognised page %d of %s (confidence %.3f)",
                number, path.name, confidence,
            )

            pages.append({
                "page_number": number,
                "text": ocr_text,
                "content_type": HANDWRITTEN,
                "method": "trocr",
                "ocr_confidence": round(confidence, 4),
            })
    finally:
        document.close()

    return pages


def extract_docx(path: Path) -> str:
    import docx

    document = docx.Document(str(path))
    parts = [paragraph.text for paragraph in document.paragraphs]

    for table in document.tables:
        for row in table.rows:
            parts.append(" ".join(cell.text for cell in row.cells))

    return "\n".join(part for part in parts if part.strip())


def _chunk_records(
    text: str,
    document_name: str,
    page_number,
    content_type: str,
    method: str,
    start_index: int,
    ocr_confidence=None,
) -> list[dict]:
    records = []

    for index, chunk in enumerate(chunk_page_text(text)):
        position = start_index + index
        record = {
            "id": str(uuid.uuid4()),
            "filename": document_name,
            "document_name": document_name,
            "chunk_index": position,
            "chunk_id": f"{page_number or 0}-{position}",
            "page_number": page_number,
            "content_type": content_type,
            "extraction_method": method,
            "text": chunk,
        }
        if ocr_confidence is not None:
            record["ocr_confidence"] = ocr_confidence
        records.append(record)

    return records


def _summarize_content_type(content_types: set) -> str:
    if content_types == {HANDWRITTEN}:
        return HANDWRITTEN
    if content_types == {DIGITAL}:
        return DIGITAL
    return "mixed"


def process_upload(path: Path, original_name: str) -> dict:
    ext = path.suffix.lower()
    records: list[dict] = []
    methods: set[str] = set()
    content_types: set[str] = set()
    page_count = 0
    ocr_pages = 0

    if ext == ".pdf":
        pages = extract_pdf_pages(path)
        page_count = len(pages)

        for page in pages:
            methods.add(page["method"])
            content_types.add(page["content_type"])
            if page["content_type"] == HANDWRITTEN:
                ocr_pages += 1

            records.extend(_chunk_records(
                page["text"], original_name, page["page_number"],
                page["content_type"], page["method"], len(records),
                page.get("ocr_confidence"),
            ))

    elif ext == ".docx":
        methods.add("python-docx")
        content_types.add(DIGITAL)
        records.extend(_chunk_records(
            extract_docx(path), original_name, None, DIGITAL, "python-docx", 0
        ))

    elif ext in {".txt", ".md"}:
        methods.add("plaintext")
        content_types.add(DIGITAL)
        records.extend(_chunk_records(
            path.read_text(encoding="utf-8", errors="ignore"),
            original_name, None, DIGITAL, "plaintext", 0,
        ))

    elif ext in {".png", ".jpg", ".jpeg"}:
        from backend import handwriting

        text, confidence = handwriting.recognize_image(path)
        methods.add("trocr")
        content_types.add(HANDWRITTEN)
        page_count = 1
        ocr_pages = 1
        records.extend(_chunk_records(
            text, original_name, 1, HANDWRITTEN, "trocr", 0, round(confidence, 4)
        ))

    else:
        raise ValueError(f"Unsupported file type: {ext}")

    if not records:
        return {
            "status": "FAILED",
            "error": "No readable text found in this document.",
        }

    # Drop the previous version everywhere, including the vector index, so a
    # re-upload can never leave stale chunks retrievable under the same name.
    from backend import retriever

    clear_document(original_name)
    retriever.purge_document(original_name)
    add_chunks(records)
    retriever.index_document(original_name)

    logger.info(
        "Indexed %d chunks from %s (%s, %d page(s), %d OCR'd)",
        len(records), original_name, "+".join(sorted(methods)), page_count, ocr_pages,
    )

    return {
        "status": "INDEXED",
        "extraction_method": "+".join(sorted(methods)),
        "content_type": _summarize_content_type(content_types),
        "chunk_count": len(records),
        "page_count": page_count,
        "ocr_page_count": ocr_pages,
    }
