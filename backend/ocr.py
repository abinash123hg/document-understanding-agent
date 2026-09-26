"""OCR using EasyOCR - fully local, handles handwritten text."""
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
_reader = None


def get_reader():
    """Load EasyOCR once (lazy). English, CPU mode."""
    global _reader
    if _reader is None:
        import easyocr
        logger.info("Loading EasyOCR model (first time only)...")
        _reader = easyocr.Reader(["en"], gpu=False)
    return _reader


def ocr_image(path: Path) -> str:
    """Extract text from an image (handwritten or printed)."""
    results = get_reader().readtext(str(path))
    lines = [t.strip() for _, t, conf in results if t.strip() and conf >= 0.3]
    return "\n".join(lines)
