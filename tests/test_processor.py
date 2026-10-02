"""
Stage 4-5 tests: normalization, chunking, extraction routing and metadata.

These cover the rules the pipeline depends on: cleaning must never change
content, chunking must always terminate, a page with a text layer must never pay
for recognition, and a chunk's confidence must describe the worst line inside it.
"""

import pytest

from backend import handwriting, processor, storage
from backend.config import Settings, settings


@pytest.fixture(autouse=True)
def no_vector_index(monkeypatch):
    """Keep these tests about text, not embeddings."""
    from backend import retriever

    monkeypatch.setattr(retriever, "index_document", lambda name: 0)
    monkeypatch.setattr(retriever, "purge_document", lambda name: None)


@pytest.fixture
def trocr_only(monkeypatch):
    """
    Force every scanned page onto the TrOCR path.

    The printed-scan fast path is a routing decision of its own and is tested
    separately; without this fixture these tests would depend on whether a
    Tesseract binary happens to be installed.
    """
    monkeypatch.setattr(settings, "printed_scan_ocr", False)


@pytest.fixture
def no_tesseract(monkeypatch):
    """
    Pretend the optional binary is absent, exactly as the fallback sees it.

    Only the wrapper call is replaced, so the page still travels the real
    printed-scan branch and the TrOCR fallback is exercised end to end.
    """
    def refuse(*args, **kwargs):
        raise OSError("tesseract is not installed or it's not in your PATH")

    monkeypatch.setattr("pytesseract.image_to_string", refuse, raising=True)


def make_scanned_pdf(path, pages=1, image_path=None):
    import pymupdf

    document = pymupdf.open()
    for _ in range(pages):
        page = document.new_page(width=595, height=842)
        if image_path is not None:
            page.insert_image(page.rect, filename=str(image_path))
    document.save(str(path))
    document.close()


def blank_page_image(tmp_path):
    from PIL import Image

    image_path = tmp_path / "page.png"
    Image.new("L", (600, 800), 255).save(image_path)
    return image_path


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


def test_indexed_pdf_text_has_the_repeated_header_removed(tmp_path):
    """
    The filter is not only a gate any more.

    A page that repeats "Sample University 2024" on every page used to be indexed
    with that header inside it, so the boilerplate was retrieved, reranked and
    quoted back as though it were content.
    """
    import pymupdf

    path = tmp_path / "headed.pdf"
    document = pymupdf.open()
    for number in (1, 2):
        page = document.new_page(width=595, height=842)
        page.insert_text((72, 90), "Sample University 2024", fontsize=12)
        page.insert_text(
            (72, 130),
            f"Page {number} reports a distinct measurement of {number * 12} litres.",
            fontsize=12,
        )
    document.save(str(path))
    document.close()

    pages = processor.extract_pdf_pages(path)

    assert len(pages) == 2
    assert all(page["content_type"] == "digital_text" for page in pages)
    assert all("Sample University 2024" not in page["text"] for page in pages)
    assert "distinct measurement of 24 litres" in pages[1]["text"]


def test_a_page_of_only_boilerplate_is_dropped_not_recognised(tmp_path, monkeypatch):
    """
    A page whose entire text layer is the running header used to count as a scan.

    It then paid for a rasterised page and had the same header indexed back at it
    unfiltered by the OCR path, which undoes the filter one page at a time.
    """
    import pymupdf

    calls = []

    def spy(gray):
        calls.append(gray)
        return {"page_number": 0, "text": "", "lines": [], "content_type": "handwritten_ocr",
                "method": "trocr", "ocr_confidence": 0.0}

    monkeypatch.setattr(processor, "_recognize_page", spy)

    path = tmp_path / "header_only.pdf"
    document = pymupdf.open()
    for number in (1, 2, 3):
        page = document.new_page(width=595, height=842)
        page.insert_text((72, 90), "Sample University 2024", fontsize=12)
        if number < 3:
            page.insert_text(
                (72, 130),
                f"Page {number} reports {number * 12} litres of storage capacity.",
                fontsize=12,
            )
    document.save(str(path))
    document.close()

    pages = processor.extract_pdf_pages(path)

    assert [page["page_number"] for page in pages] == [1, 2]
    assert not calls, "a page with a text layer must never reach the recogniser"


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


def test_image_only_pdf_page_is_routed_to_recognition(tmp_path, monkeypatch, trocr_only):
    path = tmp_path / "scanned.pdf"
    make_scanned_pdf(path, 1, blank_page_image(tmp_path))

    seen = {}

    def fake_recognize_lines(gray):
        seen["shape"] = gray.shape
        return [("recognised line one", 0.9), ("recognised line two", 0.64)]

    monkeypatch.setattr(handwriting, "recognize_page_lines", fake_recognize_lines)

    pages = processor.extract_pdf_pages(path)

    assert pages[0]["content_type"] == "handwritten_ocr"
    assert pages[0]["method"] == "trocr"
    # Mean is what the page reports; each chunk gets the minimum of its own lines.
    assert pages[0]["ocr_confidence"] == pytest.approx(0.77)
    assert pages[0]["lines"] == [("recognised line one", 0.9), ("recognised line two", 0.64)]
    assert seen["shape"][0] > 0, "the page should have been rasterized"


def test_repeated_plain_text_footer_routes_scanned_pages_to_recognition(
    tmp_path, monkeypatch, trocr_only
):
    path = tmp_path / "repeated_footer_scan.pdf"
    image_path = blank_page_image(tmp_path)

    import pymupdf

    document = pymupdf.open()
    for _ in range(2):
        page = document.new_page(width=595, height=842)
        page.insert_image(page.rect, filename=str(image_path))
        page.insert_text((72, 90), "Sample University 2024", fontsize=12)
    document.save(str(path))
    document.close()

    seen = []
    monkeypatch.setattr(
        handwriting,
        "recognize_page_lines",
        lambda gray: seen.append(gray.shape) or [("recognized content", 0.8)],
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

    recognition_calls = []
    monkeypatch.setattr(
        processor,
        "_recognize_page",
        lambda gray: recognition_calls.append(gray.shape),
    )

    pages = processor.extract_pdf_pages(path)

    assert [page["page_number"] for page in pages] == [1, 2]
    assert all(page["content_type"] == "digital_text" for page in pages)
    assert all(page["method"] == "pymupdf" for page in pages)
    assert "Page 1 explains" in pages[0]["text"]
    assert "Page 2 explains" in pages[1]["text"]
    assert recognition_calls == [], "a page with a text layer must never be recognized"


def test_single_bare_url_text_layer_routes_to_recognition(tmp_path, monkeypatch, trocr_only):
    import pymupdf

    path = tmp_path / "url_only.pdf"
    document = pymupdf.open()
    page = document.new_page(width=595, height=842)
    page.insert_text((72, 90), "https://example.com", fontsize=12)
    document.save(str(path))
    document.close()

    monkeypatch.setattr(
        handwriting, "recognize_page_lines", lambda gray: [("recognized content", 0.8)]
    )

    pages = processor.extract_pdf_pages(path)

    assert pages[0]["content_type"] == "handwritten_ocr"
    assert pages[0]["method"] == "trocr"
    assert pages[0]["text"] == "recognized content"


def test_scanned_page_is_read_by_tesseract_before_trocr(tmp_path, monkeypatch):
    """Printed scans are the case TrOCR is slowest and worst at, so the local
    Tesseract binary reads them and TrOCR is never called."""
    path = tmp_path / "printed_scan.pdf"
    make_scanned_pdf(path, 1, blank_page_image(tmp_path))

    printed = ("This printed paragraph is long enough to be trusted by the "
               "fast path and carries real words.")
    monkeypatch.setattr(processor, "_printed_page_text", lambda gray: (printed, 0.93))

    trocr_calls = []
    monkeypatch.setattr(
        handwriting,
        "recognize_page_lines",
        lambda gray: trocr_calls.append(gray) or [("should not run", 0.5)],
    )

    pages = processor.extract_pdf_pages(path)

    assert pages[0]["method"] == "pytesseract"
    assert pages[0]["content_type"] == "digital_text"
    assert pages[0]["text"] == printed
    assert pages[0]["ocr_confidence"] == 0.93
    assert trocr_calls == []


def test_a_printed_page_that_reports_no_confidence_goes_to_trocr(tmp_path, monkeypatch):
    """
    Text with no engine score behind it is not evidence.

    Accepting it would index an OCR reading as though it were an embedded text
    layer, which never faces the confidence gate.
    """
    path = tmp_path / "blind_scan.pdf"
    make_scanned_pdf(path, 1, blank_page_image(tmp_path))

    monkeypatch.setattr("pytesseract.image_to_string",
                        lambda image: "A printed paragraph long enough to look real.",
                        raising=True)

    def no_score(image, **kwargs):
        raise RuntimeError("image_to_data is unavailable")

    monkeypatch.setattr("pytesseract.image_to_data", no_score, raising=True)
    monkeypatch.setattr(
        handwriting, "recognize_page_lines", lambda gray: [("read by trocr instead", 0.7)]
    )

    pages = processor.extract_pdf_pages(path)

    assert pages[0]["method"] == "trocr"
    assert pages[0]["content_type"] == "handwritten_ocr"


def test_mostly_non_letter_tesseract_output_falls_through_to_trocr(tmp_path, monkeypatch):
    """A garbled printed read must not be indexed as content."""
    path = tmp_path / "junk_scan.pdf"
    make_scanned_pdf(path, 1, blank_page_image(tmp_path))

    monkeypatch.setattr(processor, "_printed_page_text", lambda gray: None)
    monkeypatch.setattr(
        handwriting, "recognize_page_lines", lambda gray: [("actual handwriting", 0.6)]
    )

    pages = processor.extract_pdf_pages(path)

    assert pages[0]["method"] == "trocr"
    assert pages[0]["content_type"] == "handwritten_ocr"


def test_printed_text_usability_check_rejects_short_and_symbol_runs():
    assert processor._printed_text_is_usable(
        "A sentence long enough to count as a real printed paragraph."
    )
    assert not processor._printed_text_is_usable("ok")
    assert not processor._printed_text_is_usable("|||| |||| ~~~~ ---- #### (((( ~~~~ "
                                                 "---- |||| ~~~~")
    assert not processor._printed_text_is_usable("")


def test_missing_tesseract_binary_falls_back_to_trocr_without_crashing(
    tmp_path, monkeypatch, no_tesseract
):
    """An optional tool must never take the upload down with it."""
    path = tmp_path / "scan_no_binary.pdf"
    make_scanned_pdf(path, 1, blank_page_image(tmp_path))

    monkeypatch.setattr(
        handwriting, "recognize_page_lines", lambda gray: [("handwriting recovered", 0.55)]
    )

    pages = processor.extract_pdf_pages(path)

    assert pages[0]["method"] == "trocr"
    assert "handwriting recovered" in pages[0]["text"]


def test_real_tesseract_path_does_not_raise_when_binary_is_absent(monkeypatch):
    """The wrapper raises when the binary is missing; the page must come back as
    unusable so the caller falls through to TrOCR."""
    def refuse(image):
        raise OSError("tesseract is not installed")

    import numpy

    monkeypatch.setattr("pytesseract.image_to_string", refuse, raising=True)

    assert processor._printed_page_text(numpy.zeros((40, 40), dtype=numpy.uint8)) is None


def test_ocr_page_budget_raises_instead_of_silently_truncating(tmp_path, monkeypatch, trocr_only):
    path = tmp_path / "many_scans.pdf"
    make_scanned_pdf(path, 3)

    monkeypatch.setattr(settings, "pdf_max_ocr_pages", 1)
    monkeypatch.setattr(processor, "EMBEDDED_TEXT_MIN_CHARS", 5000)

    recognized = []
    monkeypatch.setattr(
        handwriting,
        "recognize_page_lines",
        lambda gray: recognized.append(gray.shape) or [("text", 0.9)],
    )

    with pytest.raises(ValueError, match="at most 1 are recognised per upload"):
        processor.extract_pdf_pages(path)

    assert recognized == [], "the budget is decided before any page is recognized"


def test_ocr_page_budget_default_is_forty():
    """The limit is a laptop guarantee, not a suggestion: it is what stops a
    300-page scan hanging an upload."""
    assert Settings.model_fields["pdf_max_ocr_pages"].default == 40


def test_scanned_pages_are_recognised_two_at_a_time(tmp_path, monkeypatch, trocr_only):
    workers = {}
    real_executor = processor.ThreadPoolExecutor

    def spy_executor(max_workers=None):
        workers["max_workers"] = max_workers
        return real_executor(max_workers=max_workers)

    monkeypatch.setattr(processor, "ThreadPoolExecutor", spy_executor)
    monkeypatch.setattr(
        handwriting, "recognize_page_lines", lambda gray: [("line", 0.7)]
    )

    path = tmp_path / "three_scans.pdf"
    make_scanned_pdf(path, 3)

    pages = processor.extract_pdf_pages(path)

    assert workers["max_workers"] == settings.ocr_pages_at_a_time == 2
    assert len(pages) == 3
    assert [page["page_number"] for page in pages] == [1, 2, 3], "page order survives"


def test_chunk_confidence_is_the_worst_line_inside_the_chunk():
    """
    A few clean lines must not vouch for a garbled one.

    The chunk that holds the bad line is the one that gets refused, and the
    chunks that do not contain it keep their own high confidence.
    """
    def line(text):
        # Long enough that two lines fill a chunk and a third has to start the
        # next one, so the four lines really do land in more than one chunk.
        return f"{text} " + "observed in the field " * 9

    lines = [
        (line("The survey covered twelve villages in June."), 0.93),
        (line("Each household received a storage tank."), 0.88),
        (line("zzzz qqqw rxn pwv"), 0.05),
        (line("Water quality was tested at 46 points."), 0.91),
    ]

    chunks = processor.chunk_lines(lines)

    assert len(chunks) > 1, "the lines must spread over more than one chunk"
    garbled = [confidence for text, confidence in chunks if "zzzz" in text]
    clean = [confidence for text, confidence in chunks if "zzzz" not in text]

    assert garbled and max(garbled) == 0.05, "the bad line sets its chunk's score"
    assert clean and min(clean) >= 0.88, "one bad line must not poison the other chunks"
    assert all(0.0 <= confidence <= 1.0 for _, confidence in chunks)


def test_chunk_lines_drops_blank_lines_and_terminates_on_one_huge_line():
    lines = [("   ", 0.9), ("x" * 1200, 0.42), ("Short line.", 0.8)]

    chunks = processor.chunk_lines(lines)

    assert chunks
    assert all(text.strip() for text, _ in chunks)
    assert ("x" * 1200, 0.42) in chunks, "an oversized line stands as its own chunk"


def test_handwritten_chunks_carry_line_minimum_confidence(tmp_path, monkeypatch, trocr_only):
    """End to end: the stored record's ocr_confidence is the minimum of the lines
    that formed that chunk, so the answer stage refuses on the weakest evidence
    it actually contains."""
    path = tmp_path / "notes_scan.pdf"
    make_scanned_pdf(path, 1, blank_page_image(tmp_path))

    monkeypatch.setattr(
        handwriting,
        "recognize_page_lines",
        lambda gray: [
            ("The survey covered twelve villages in June.", 0.93),
            ("zzzz qqqw rxn pwv unq", 0.07),
        ],
    )

    result = processor.process_upload(path, "notes_scan.pdf")

    assert result["status"] == "INDEXED"
    chunks = storage.get_chunks(document_name="notes_scan.pdf")
    assert chunks
    assert all(chunk["ocr_confidence"] == 0.07 for chunk in chunks), \
        "both lines land in one chunk, so the chunk takes the worse score"
    assert all(chunk["content_type"] == "handwritten_ocr" for chunk in chunks)


def test_low_confidence_chunk_is_not_citable_by_the_answer_stage(
    tmp_path, monkeypatch, trocr_only
):
    """The minimum-confidence chunk score is what the excerpt gate reads, so the
    garbled chunk is dropped before the model ever sees it."""
    from backend import llm

    path = tmp_path / "mixed_scan.pdf"
    make_scanned_pdf(path, 1, blank_page_image(tmp_path))

    monkeypatch.setattr(
        handwriting,
        "recognize_page_lines",
        lambda gray: [("zzzz qqqw rxn pwv unq", 0.07)],
    )

    processor.process_upload(path, "mixed_scan.pdf")
    chunk = storage.get_chunks(document_name="mixed_scan.pdf")[0]

    assert chunk["ocr_confidence"] < settings.htr_min_confidence
    assert not llm._is_trusted_excerpt(chunk)


def test_process_upload_stores_full_metadata(tmp_path):
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


def test_a_tesseract_read_page_is_counted_as_an_ocr_page(tmp_path, monkeypatch):
    """
    A page read by Tesseract was rasterized and OCR'd, and the upload report must
    say so: counting only TrOCR pages reported "0 OCR pages" for a scan that had
    just spent seconds under the other engine.
    """
    path = tmp_path / "printed_upload.pdf"
    make_scanned_pdf(path, 1, blank_page_image(tmp_path))

    monkeypatch.setattr(
        processor, "_printed_page_text",
        lambda gray: ("The printed report lists 200 litres for each tank.", 0.91),
    )

    result = processor.process_upload(path, "printed_upload.pdf")

    assert result["page_count"] == 1
    assert result["ocr_page_count"] == 1

    chunk = storage.get_chunks(document_name="printed_upload.pdf")[0]
    assert chunk["extraction_method"] == "pytesseract"
    assert chunk["ocr_confidence"] == 0.91


def test_digital_chunks_carry_no_ocr_confidence(tmp_path):
    path = tmp_path / "clean.txt"
    path.write_text("A typed sentence with the figure 200 litres in it.", encoding="utf-8")

    processor.process_upload(path, "clean.txt")

    chunk = storage.get_chunks(document_name="clean.txt")[0]

    assert "ocr_confidence" not in chunk
    assert chunk["content_type"] == "digital_text"


def test_reupload_leaves_no_previous_version(tmp_path):
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


def test_defaults_fit_the_laptop_budget():
    """
    Pinned so a future default cannot quietly blow past the machine's limits.

    1.5b drops citations and 7b is 4.7 GB, which with TrOCR breaks the 5-6 GB
    ceiling; the recognition model stays at trocr-base; DPI is 150 because 200
    buys nothing on these pages for the extra seconds.
    """
    defaults = Settings.model_fields

    assert defaults["llm_model"].default == "qwen2.5:3b"
    assert defaults["htr_model"].default == "microsoft/trocr-base-handwritten"
    assert defaults["pdf_dpi"].default == 150
    assert defaults["max_upload_size_mb"].default == 100
    assert defaults["denoise_images"].default is False
    assert defaults["trocr_line_batch"].default == 4
    assert defaults["htr_min_confidence"].default == 0.30


def test_no_cuda_anywhere_in_the_backend():
    from pathlib import Path

    backend_dir = Path(__file__).resolve().parents[1] / "backend"
    sources = sorted(backend_dir.glob("*.py"))

    assert sources, "the backend package should be inspectable"
    assert not [
        path.name for path in sources if b"cuda" in path.read_bytes().lower()
    ], "this project is CPU-only"
