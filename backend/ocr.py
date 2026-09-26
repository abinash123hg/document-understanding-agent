"""
Fast OCR using EasyOCR - optimized for speed on CPU
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)
_reader = None


def get_reader():
    """Load EasyOCR only once (lazy loading)"""
    global _reader
    if _reader is None:
        import easyocr
        logger.info("Loading EasyOCR model (first time only)...")
        # gpu=False keeps it under your hardware limit
        _reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _reader


def ocr_image(path: Path) -> str:
    """Extract text from image - optimized for speed"""
    reader = get_reader()
    # Lower detail = much faster
    results = reader.readtext(
        str(path),
        detail=0,          # only text, no boxes
        paragraph=True,    # faster grouping
        batch_size=1
    )
    return "\n".join([t.strip() for t in results if t.strip()])