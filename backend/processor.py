"""
Document processor: extraction, normalization and chunking.

Routing rules
- .txt / .md      -> read directly                       (digital_text)
- .docx           -> python-docx                         (digital_text)
- .pdf            -> embedded text per page when present (digital_text);
                     a page with no usable text layer is rasterized and read by
                     Tesseract first                       (digital_text),
                     and only a page Tesseract cannot read goes to TrOCR
                     (handwritten_ocr). This keeps printed scans on the fast path
- .png/.jpg/.jpeg -> preprocessing + TrOCR               (handwritten_ocr)

Every chunk keeps its document name, page number, chunk id, extraction method
and content type, so citations can point at a real page.
"""

import logging
import re
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
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


def chunk_lines(lines: list[tuple[str, float]]) -> list[tuple[str, float]]:
    """
    Group recognised lines into chunks, each carrying its worst line.

    A page-level mean hides the fact that one line of a page was unreadable, and
    a chunk averages over only its own lines still lets three clean lines carry a
    garbled fourth. Taking the minimum confidence of the lines that formed a
    chunk means the number attached to a chunk is the number for the weakest
    evidence inside it, which is what the answer stage should be refusing on.
    """
    kept = [(line.strip(), float(confidence))
            for line, confidence in lines if str(line or "").strip()]
    if not kept:
        return []

    size = max(400, int(settings.chunk_size))
    overlap = max(0, min(int(settings.chunk_overlap), size // 2))

    chunks: list[tuple[str, float]] = []
    current: list[tuple[str, float]] = []
    length = 0
    index = 0

    while index < len(kept):
        line, confidence = kept[index]

        if current and length + len(line) + 1 > size:
            chunks.append(_join_chunk(current))

            # Carry the tail of the finished chunk into the next one so a
            # sentence split across the boundary stays readable. If even the
            # carried lines leave no room for this line, the chunk simply starts
            # fresh; carrying further would spin.
            carry = _carry_lines(current, overlap)
            carry_length = sum(len(item[0]) + 1 for item in carry)
            if carry and carry_length + len(line) + 1 <= size:
                current, length = carry, carry_length
            else:
                current, length = [], 0
            continue

        current.append((line, confidence))
        length += len(line) + 1
        index += 1

    if current:
        chunks.append(_join_chunk(current))

    return chunks


def _join_chunk(items: list[tuple[str, float]]) -> tuple[str, float]:
    return "\n".join(line for line, _ in items), min(
        confidence for _, confidence in items
    )


def _carry_lines(items: list[tuple[str, float]], overlap: int) -> list[tuple[str, float]]:
    """Trailing lines of a finished chunk that fit inside the overlap budget."""
    if overlap <= 0:
        return []

    carry: list[tuple[str, float]] = []
    length = 0
    for item in reversed(items):
        extra = len(item[0]) + 1
        if carry and length + extra > overlap:
            break
        carry.insert(0, item)
        length += extra
    return carry


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

    # MuPDF prints non-fatal complaints ("No default Layer config") straight to
    # stderr even when the page reads fine. Failures that matter still raise,
    # and the rejection below turns them into a 422.
    pymupdf.TOOLS.mupdf_display_errors(False)

    try:
        return pymupdf.open(str(path))
    except Exception as error:
        raise ValueError(f"Could not open this PDF: {error}") from error


def _boilerplate_lines(raw_texts: list[str]) -> set[str]:
    """Lines that repeat across pages, plus URL-only lines: headers, footers,
    watermarks. They are noise for retrieval and they crowd out real content."""
    frequency: Counter = Counter()
    for raw_text in raw_texts:
        frequency.update({
            line.strip() for line in raw_text.splitlines() if line.strip()
        })

    return {
        line for line, count in frequency.items()
        if count >= 2 or URL_ONLY_TEXT.fullmatch(line.rstrip(".,;:)]}"))
    }


def _usable_page_text(raw_text: str, boilerplate: set[str]) -> str:
    """
    The page with its boilerplate removed.

    This filtered text is what gets indexed. Keeping the raw page instead put
    the repeated header in every chunk of every page, where it competed with the
    sentence that actually answers a question.
    """
    return "\n".join(
        line for line in raw_text.splitlines()
        if line.strip() and line.strip() not in boilerplate
    ).strip()


def _printed_text_is_usable(text: str) -> bool:
    """
    Whether a printed-page OCR pass produced something worth keeping.

    An empty result or a string that is mostly non-letters means the fast path
    failed and the page should go to TrOCR rather than be indexed as garbage.
    """
    text = normalize_text(text)
    if len(text) < EMBEDDED_TEXT_MIN_CHARS:
        return False

    content = [char for char in text if not char.isspace()]
    letters = sum(1 for char in content if char.isalpha())
    return letters >= 0.5 * len(content)


def _tesseract_confidence(image) -> float | None:
    """
    Tesseract's own mean word confidence, scaled to 0..1 so it can share the
    handwriting gate. None means the engine could not say.
    """
    try:
        import pytesseract
        from pytesseract import Output

        data = pytesseract.image_to_data(image, output_type=Output.DICT)
        scores = [
            float(value) for value, word in zip(data.get("conf", []), data.get("text", []))
            if str(word).strip() and float(value) >= 0
        ]
    except Exception as error:
        logger.warning("Tesseract could not report a confidence for this page: %s", error)
        return None

    return round(sum(scores) / len(scores) / 100.0, 4) if scores else None


def _printed_page_text(gray: np.ndarray) -> tuple[str, float | None] | None:
    """
    Read a rasterized page with Tesseract. None means "not usable, use TrOCR".

    Printed scans are the case TrOCR is worst at and slowest for, so they go
    through the local binary first. When that binary is not installed the page
    is still recognised; an upload never dies over a missing optional tool.
    """
    if not settings.printed_scan_ocr:
        return None

    try:
        import pytesseract
        from PIL import Image

        image = Image.fromarray(gray)
        text = pytesseract.image_to_string(image)
    except Exception as error:
        logger.warning(
            "Tesseract could not read this page (%s); falling back to TrOCR. "
            "Install the tesseract-ocr binary to keep printed scans fast.",
            error,
        )
        return None

    stripped = text.strip()
    if not _printed_text_is_usable(stripped):
        return None

    confidence = _tesseract_confidence(image)
    if confidence is None:
        # An OCR reading nobody can vouch for is not evidence, and handing it on
        # without a score would let it past the confidence gate as though it were
        # an embedded text layer. TrOCR reads the page instead.
        return None

    return stripped, confidence


def _recognize_page(gray: np.ndarray) -> dict:
    """One rasterized page: printed reader first, TrOCR only when it fails."""
    printed = _printed_page_text(gray)
    if printed is not None:
        printed_text, confidence = printed
        # Text that came out of an image is OCR output whatever engine made it, so
        # it carries the confidence the same way a TrOCR page does and faces the
        # same gate.
        return {
            "text": printed_text,
            "content_type": DIGITAL,
            "method": "pytesseract",
            "ocr_confidence": confidence,
        }

    from backend import handwriting

    lines = handwriting.recognize_page_lines(gray)
    return {
        "lines": lines,
        "text": "\n".join(line for line, _ in lines if line.strip()),
        "content_type": HANDWRITTEN,
        "method": "trocr",
        "ocr_confidence": round(handwriting.mean_confidence(lines), 4),
    }


def _recognize_scanned_pages(items: list[tuple[int, np.ndarray]]) -> list[dict]:
    """
    Recognize rasterized pages a few at a time, returning them in page order.

    Rasterizing stays on the calling thread because a pymupdf document is not
    safe to share; only recognition is handed to the pool, capped so two pages
    never become forty running at once on a laptop CPU.
    """
    if not items:
        return []

    total = len(items)
    results: list[dict] = []

    with ThreadPoolExecutor(
        max_workers=max(1, int(settings.ocr_pages_at_a_time))
    ) as pool:
        futures = [
            (number, pool.submit(_recognize_page, gray))
            for number, gray in items
        ]

        for finished, (number, future) in enumerate(futures, start=1):
            page = future.result()
            logger.info(
                "Recognised scanned page %d of %d (PDF page %d)",
                finished, total, number,
            )
            page["page_number"] = number
            results.append(page)

    results.sort(key=lambda page: page["page_number"])
    return results


def extract_pdf_pages(path: Path) -> list[dict]:
    """
    Per-page extraction with real page numbers.

    Embedded text is read first because it is effectively free, and the filtered
    version of it is what gets indexed. A page only pays the rasterize-and-
    recognise cost when it has no usable text layer, which is what keeps normal
    PDF uploads near-instant.
    """
    document = _open_pdf(path)
    try:
        raw_texts = [page.get_text("text").strip() for page in document]
        boilerplate = _boilerplate_lines(raw_texts)

        pages: list[dict] = []
        needs_ocr: list[int] = []

        for number, raw_text in enumerate(raw_texts, start=1):
            usable_text = _usable_page_text(raw_text, boilerplate)
            if len(usable_text) >= EMBEDDED_TEXT_MIN_CHARS:
                pages.append({
                    "page_number": number,
                    "text": usable_text,
                    "content_type": DIGITAL,
                    "method": "pymupdf",
                })
                continue

            if len(raw_text) >= EMBEDDED_TEXT_MIN_CHARS and not document[
                number - 1
            ].get_images():
                # The page is pure text and every line of it was boilerplate, so
                # there is nothing left to index. OCR-ing it would pay for a
                # rasterised page and hand the same header back unfiltered.
                # A page that also carries an image is a different matter: that
                # is a scan with a typed header, and the image needs reading.
                continue

            needs_ocr.append(number)

        # Decided before a single page is rasterized: an oversized scan is
        # rejected immediately instead of after minutes of work that is thrown
        # away anyway.
        if len(needs_ocr) > settings.pdf_max_ocr_pages:
            raise ValueError(
                f"This PDF has {len(needs_ocr)} scanned pages, but at most "
                f"{settings.pdf_max_ocr_pages} are recognised per upload. Split "
                f"it into files of {settings.pdf_max_ocr_pages} scanned pages or "
                f"fewer and upload each one, or raise PDF_MAX_OCR_PAGES in .env."
            )

        items = [
            (number, _downscale(_pixmap_to_gray(
                document[number - 1].get_pixmap(dpi=settings.pdf_dpi, alpha=False)
            )))
            for number in needs_ocr
        ]
    finally:
        document.close()

    pages.extend(_recognize_scanned_pages(items))
    pages.sort(key=lambda page: page["page_number"])
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
    chunks: list,
    document_name: str,
    page_number,
    content_type: str,
    method: str,
    start_index: int,
    page_confidence=None,
) -> list[dict]:
    """
    Turn chunked text into stored records.

    A chunk arrives either as plain text or as (text, confidence) from the
    handwritten path, where the confidence is the minimum of the lines inside it.
    page_confidence covers a chunk that came out of an OCR engine without per-line
    scores, so it still faces the confidence gate instead of reading as an
    embedded text layer.
    """
    records = []

    for index, item in enumerate(chunks):
        if isinstance(item, tuple):
            chunk, confidence = item
        else:
            chunk, confidence = item, page_confidence

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
        if confidence is not None:
            record["ocr_confidence"] = round(float(confidence), 4)
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
                # Recognised lines keep their own confidence, so the chunk they
                # form is scored by its weakest line.
                chunks = chunk_lines(page["lines"])
            else:
                chunks = chunk_page_text(page["text"])

            # A page that was rasterized and read by an engine, whichever one it
            # was. Counting only TrOCR pages reported "0 OCR pages" for a scan
            # that had just spent seconds under Tesseract.
            if page["method"] in {"trocr", "pytesseract"}:
                ocr_pages += 1

            records.extend(_chunk_records(
                chunks, original_name, page["page_number"],
                page["content_type"], page["method"], len(records),
                page.get("ocr_confidence"),
            ))

    elif ext == ".docx":
        methods.add("python-docx")
        content_types.add(DIGITAL)
        records.extend(_chunk_records(
            chunk_page_text(extract_docx(path)),
            original_name, None, DIGITAL, "python-docx", 0
        ))

    elif ext in {".txt", ".md"}:
        methods.add("plaintext")
        content_types.add(DIGITAL)
        records.extend(_chunk_records(
            chunk_page_text(path.read_text(encoding="utf-8", errors="ignore")),
            original_name, None, DIGITAL, "plaintext", 0,
        ))

    elif ext in {".png", ".jpg", ".jpeg"}:
        from backend import handwriting, imaging

        lines = handwriting.recognize_page_lines(imaging.load_image(path))
        methods.add("trocr")
        content_types.add(HANDWRITTEN)
        page_count = 1
        ocr_pages = 1
        records.extend(_chunk_records(
            chunk_lines(lines), original_name, 1, HANDWRITTEN, "trocr", 0
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
