"""
Stage 4-5 tests: normalization, chunking and extraction metadata.

These cover the two rules the pipeline depends on: cleaning must never change
content, and chunking must always terminate.
"""

import pytest

from backend import processor
from backend.config import settings


@pytest.fixture(autouse=True)
def no_vector_index(monkeypatch):
    """Keep these tests about text, not embeddings."""
    from backend import retriever

    monkeypatch.setattr(retriever, "index_document", lambda name: 0)
    monkeypatch.setattr(retriever, "purge_document", lambda name: None)


def test_normalize_keeps_numbers_units_names_and_formulas():
    raw = "Section 3.1\nDr. Rao paid  1,480.50   rupees for 200 litres (pH=7.2)   "
    clean = processor.normalize_text(raw)

    assert "1,480.50" in clean
    assert "200 litres" in clean
    assert "pH=7.2" in clean
    assert "Dr. Rao" in clean
    assert "   " not in clean


def test_normalize_preserves_line_breaks_and_caps_blank_runs():
    assert processor.normalize_text("alpha\n\n\n\nbeta\ngamma") == "alpha\n\nbeta\ngamma"


def test_chunker_terminates_without_any_boundary():
    text = "x" * 5000
    chunks = processor.chunk_page_text(text)

    assert chunks
    assert all(len(chunk) <= max(400, settings.chunk_size) for chunk in chunks)


def test_chunker_terminates_on_tiny_overlap_window():
    text = "a. " * 400
    chunks = processor.chunk_page_text(text)
    normalized = processor.normalize_text(text)

    assert chunks
    assert all(chunk in normalized for chunk in chunks)
    assert normalized.endswith(chunks[-1])


def test_chunker_prefers_sentence_boundaries():
    text = "First sentence here. Second sentence follows. Third one ends it."
    chunks = processor.chunk_page_text(text)

    assert chunks[0].endswith(".")


def test_oversized_overlap_cannot_spin_the_loop(monkeypatch):
    monkeypatch.setattr(settings, "chunk_size", 400)
    monkeypatch.setattr(settings, "chunk_overlap", 400)

    chunks = processor.chunk_page_text("word " * 300)

    assert len(chunks) > 1


def test_empty_text_yields_no_chunks():
    assert processor.chunk_page_text("   \n\t  ") == []


def test_pdf_pages_keep_real_page_numbers(tmp_path):
    import pymupdf

    path = tmp_path / "two_pages.pdf"
    document = pymupdf.open()
    for number in (1, 2):
        page = document.new_page(width=595, height=842)
        page.insert_text((72, 90), f"Page {number} body text with enough characters.", fontsize=12)
        page.insert_text((72, 110), "Extra sentence so the page is clearly digital.", fontsize=12)
    document.save(str(path))
    document.close()

    pages = processor.extract_pdf_pages(path)

    assert [page["page_number"] for page in pages] == [1, 2]
    assert all(page["content_type"] == "digital_text" for page in pages)
    assert all(page["method"] == "pymupdf" for page in pages)
    assert "Page 2 body text" in pages[1]["text"]


def test_opening_a_pdf_silences_mupdf_stderr_without_hiding_failures(tmp_path):
    """MuPDF prints complaints like 'No default Layer config' to stderr even for
    pages that read fine. Turning the echo off must not turn rejection off."""
    import pymupdf

    shown = pymupdf.TOOLS.mupdf_display_errors()
    try:
        path = tmp_path / "one_page.pdf"
        document = pymupdf.open()
        page = document.new_page(width=595, height=842)
        page.insert_text((72, 90), "Body text long enough to be read as digital.", fontsize=12)
        document.save(str(path))
        document.close()

        assert pymupdf.TOOLS.mupdf_display_errors(True) is True
        processor.extract_pdf_pages(path)
        assert pymupdf.TOOLS.mupdf_display_errors() is False

        broken = tmp_path / "broken.pdf"
        broken.write_bytes(b"%PDF-1.4\nthis is not a real document body\n")
        with pytest.raises(ValueError, match="Could not open this PDF"):
            processor.extract_pdf_pages(broken)
    finally:
        pymupdf.TOOLS.mupdf_display_errors(shown)


def test_image_only_pdf_page_is_routed_to_recognition(tmp_path, monkeypatch):
    import pymupdf
    from PIL import Image

    image_path = tmp_path / "page.png"
    Image.new("L", (600, 800), 255).save(image_path)

    path = tmp_path / "scanned.pdf"
    document = pymupdf.open()
    page = document.new_page(width=595, height=842)
    page.insert_image(page.rect, filename=str(image_path))
    document.save(str(path))
    document.close()

    seen = {}

    def fake_recognize(gray):
        seen["shape"] = gray.shape
        return "recognised line one\nrecognised line two", 0.77

    from backend import handwriting

    monkeypatch.setattr(handwriting, "recognize_array", fake_recognize)

    pages = processor.extract_pdf_pages(path)

    assert pages[0]["content_type"] == "handwritten_ocr"
    assert pages[0]["method"] == "trocr"
    assert pages[0]["ocr_confidence"] == 0.77
    assert seen["shape"][0] > 0, "the page should have been rasterized"


def test_repeated_plain_text_footer_routes_scanned_pages_to_recognition(tmp_path, monkeypatch):
    import pymupdf
    from PIL import Image

    image_path = tmp_path / "page.png"
    Image.new("L", (600, 800), 255).save(image_path)

    path = tmp_path / "repeated_footer_scan.pdf"
    document = pymupdf.open()
    for _ in range(2):
        page = document.new_page(width=595, height=842)
        page.insert_image(page.rect, filename=str(image_path))
        page.insert_text((72, 90), "Sample University 2024", fontsize=12)
    document.save(str(path))
    document.close()

    from backend import handwriting

    seen = []
    monkeypatch.setattr(
        handwriting,
        "recognize_array",
        lambda gray: (seen.append(gray.shape) or "recognized content", 0.8),
    )

    pages = processor.extract_pdf_pages(path)

    assert len(seen) == 2
    assert all(page["content_type"] == "handwritten_ocr" for page in pages)
    assert all(page["method"] == "trocr" for page in pages)


def test_distinct_prose_text_layers_stay_digital(tmp_path, monkeypatch):
    import pymupdf

    path = tmp_path / "digital_prose.pdf"
    document = pymupdf.open()
    for number in range(1, 3):
        page = document.new_page(width=595, height=842)
        page.insert_text(
            (72, 90),
            f"Page {number} explains a distinct method using measured observations.",
            fontsize=12,
        )
    document.save(str(path))
    document.close()

    from backend import handwriting

    recognition_calls = []
    monkeypatch.setattr(
        handwriting,
        "recognize_array",
        lambda gray: recognition_calls.append(gray.shape),
    )

    pages = processor.extract_pdf_pages(path)

    assert [page["page_number"] for page in pages] == [1, 2]
    assert all(page["content_type"] == "digital_text" for page in pages)
    assert all(page["method"] == "pymupdf" for page in pages)
    assert "Page 1 explains" in pages[0]["text"]
    assert "Page 2 explains" in pages[1]["text"]
    assert recognition_calls == []


def test_single_bare_url_text_layer_routes_to_recognition(tmp_path, monkeypatch):
    import pymupdf

    path = tmp_path / "url_only.pdf"
    document = pymupdf.open()
    page = document.new_page(width=595, height=842)
    page.insert_text((72, 90), "https://example.com", fontsize=12)
    document.save(str(path))
    document.close()

    from backend import handwriting

    monkeypatch.setattr(handwriting, "recognize_array", lambda gray: ("recognized content", 0.8))

    pages = processor.extract_pdf_pages(path)

    assert pages[0]["content_type"] == "handwritten_ocr"
    assert pages[0]["method"] == "trocr"
    assert pages[0]["text"] == "recognized content"


def test_ocr_page_budget_raises_instead_of_silently_truncating(tmp_path, monkeypatch):
    import pymupdf

    path = tmp_path / "many_scans.pdf"
    document = pymupdf.open()
    for _ in range(3):
        document.new_page(width=200, height=200)
    document.save(str(path))
    document.close()

    monkeypatch.setattr(settings, "pdf_max_ocr_pages", 1)
    monkeypatch.setattr(processor, "EMBEDDED_TEXT_MIN_CHARS", 5000)

    from backend import handwriting

    monkeypatch.setattr(handwriting, "recognize_array", lambda gray: ("text", 0.9))

    with pytest.raises(ValueError, match="more than 1"):
        processor.extract_pdf_pages(path)


def test_process_upload_stores_full_metadata(tmp_path):
    from backend import storage

    path = tmp_path / "notes.txt"
    path.write_text("Clause 4.2 limits liability to 5,000 rupees per year.", encoding="utf-8")

    result = processor.process_upload(path, "notes.txt")

    assert result["status"] == "INDEXED"
    assert result["chunk_count"] >= 1

    chunks = storage.get_chunks(document_name="notes.txt")
    assert chunks[0]["content_type"] == "digital_text"
    assert chunks[0]["extraction_method"] == "plaintext"
    assert chunks[0]["chunk_id"]
    assert chunks[0]["page_number"] is None
    assert "5,000" in chunks[0]["text"]


def test_reupload_leaves_no_previous_version(tmp_path):
    from backend import storage

    path = tmp_path / "policy.txt"
    path.write_text("The committee meets every Monday.", encoding="utf-8")
    processor.process_upload(path, "policy.txt")

    path.write_text("The board meets every Thursday instead.", encoding="utf-8")
    processor.process_upload(path, "policy.txt")

    chunks = storage.get_chunks(document_name="policy.txt")

    assert len(chunks) == 1
    assert "Thursday" in chunks[0]["text"]
    assert all("Monday" not in chunk["text"] for chunk in chunks)


def test_unsupported_extension_is_rejected(tmp_path):
    path = tmp_path / "sheet.xlsx"
    path.write_text("not a spreadsheet", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported file type"):
        processor.process_upload(path, "sheet.xlsx")


def test_document_with_no_text_fails_loudly(tmp_path):
    path = tmp_path / "blank.txt"
    path.write_text("   ", encoding="utf-8")

    result = processor.process_upload(path, "blank.txt")

    assert result["status"] == "FAILED"
    assert "No readable text" in result["error"]
