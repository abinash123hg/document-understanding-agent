"""
API tests: the contract the frontend depends on, and the two guarantees that
live at the boundary - a size limit and refusal without a selected document.
"""

import pytest

from backend import llm, retriever, storage
from backend.config import settings
from backend.main import ALLOWED_EXTENSIONS

DOC = "api_test_notes.txt"

BODY = (
    "The Kolar water survey covered twelve villages. "
    "Each household received a 200 litre storage tank in June 2024. "
    "Water quality was tested at 46 sample points."
)


@pytest.fixture
def uploaded(client, fresh_document):
    fresh_document(DOC)
    response = client.post(
        "/upload", files={"file": (DOC, BODY.encode("utf-8"), "text/plain")}
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_root_reports_both_models(client):
    body = client.get("/").json()

    assert body["status"] == "running"
    assert body["llm_model"] == settings.llm_model
    assert body["handwriting_model"] == settings.htr_model
    assert body["mode"] == "document-only"


def test_health_reports_the_real_model_state(client):
    body = client.get("/health").json()

    assert body["backend"] == "ok"
    assert set(body) >= {"ollama", "llm_model_pulled", "handwriting_loaded",
                         "document_count", "chunk_count"}
    assert isinstance(body["handwriting_loaded"], bool)


def test_upload_returns_extraction_metadata(uploaded):
    assert uploaded["status"] == "INDEXED"
    assert uploaded["document_name"] == DOC
    assert uploaded["chunk_count"] >= 1
    assert uploaded["content_type"] == "digital_text"
    assert uploaded["extraction_method"] == "plaintext"


def test_documents_endpoint_lists_uploaded_files(client, uploaded):
    names = [item["document_name"] for item in client.get("/documents").json()["documents"]]

    assert DOC in names


def test_documents_upload_alias_indexes_the_same_way(client, fresh_document):
    fresh_document("alias.txt")

    response = client.post(
        "/documents/upload",
        files={"file": ("alias.txt", "The alias route stores this text.".encode(), "text/plain")},
    )

    assert response.status_code == 200
    assert storage.get_chunks(document_name="alias.txt")


def test_extension_outside_the_allowed_list_is_rejected(client):
    response = client.post("/upload", files={"file": ("book.epub", b"data", "application/epub")})

    assert response.status_code == 400
    assert ".pdf" in response.json()["detail"]


def test_every_allowed_extension_is_one_the_processor_handles():
    from backend import processor

    assert ALLOWED_EXTENSIONS == {".pdf", ".txt", ".md", ".docx", ".png", ".jpg", ".jpeg"}
    assert processor.HANDWRITTEN == "handwritten_ocr"


def test_oversized_upload_is_refused(client, monkeypatch, fresh_document):
    fresh_document("big.txt")
    monkeypatch.setattr(settings, "max_upload_size_mb", 0)

    response = client.post("/upload", files={"file": ("big.txt", b"x" * 64, "text/plain")})

    assert response.status_code == 413
    assert storage.get_chunks(document_name="big.txt") == []


def test_chat_without_a_selected_document_refuses(client):
    body = client.post("/chat", json={"question": "Anything at all?"}).json()

    assert body["answer"] == llm.NOT_FOUND
    assert body["sources"] == []
    assert body["refusal_reason"] == "no_evidence"


def test_chat_for_an_unuploaded_document_refuses(client):
    body = client.post(
        "/chat",
        json={"question": "How many villages?", "document_name": "not-uploaded.txt"},
    ).json()

    assert body["answer"] == llm.NOT_FOUND
    assert body["sources"] == []


def test_chat_returns_citable_sources(client, uploaded, monkeypatch):
    trusted = [
        {
            "filename": DOC, "document_name": DOC, "chunk_index": 0,
            "chunk_id": "0-0", "page_number": 1, "content_type": "digital_text",
            "extraction_method": "plaintext", "score": 4.2, "text": BODY,
        }
    ]
    monkeypatch.setattr(
        llm, "generate_answer_result",
        lambda question, sources: ("Twelve villages were surveyed. [S1]", None, trusted),
    )

    body = client.post(
        "/chat", json={"question": "How many villages?", "document_name": DOC}
    ).json()

    assert body["answer"] == "Twelve villages were surveyed. [S1]"
    assert body["refusal_reason"] is None
    assert body["document_name"] == DOC
    source = body["sources"][0]
    assert set(source) >= {"filename", "chunk_index", "chunk_id", "page_number",
                           "content_type", "extraction_method", "score", "text"}
    assert source["content_type"] == "digital_text"


def retrieved_row(**overrides):
    row = {
        "id": "r",
        "filename": DOC,
        "document_name": DOC,
        "chunk_index": 0,
        "chunk_id": "1-0",
        "page_number": 1,
        "content_type": "digital_text",
        "extraction_method": "pymupdf",
        "score": 1.0,
        "text": "A legible sentence.",
    }
    row.update(overrides)
    return row


def test_sources_returned_match_the_labels_the_model_used(client, monkeypatch):
    """
    The raw retrieval list must never be handed to the reader.

    It also contains excerpts the answer stage filtered out, so its index 1 is
    not the excerpt a printed [S2] points at, and following that citation lands
    on a passage the answer did not use.
    """
    unreadable = retrieved_row(
        chunk_index=0, chunk_id="1-0", content_type="handwritten_ocr",
        extraction_method="trocr", ocr_confidence=0.02,
        text="Unreadable scratchings.",
    )
    digital = retrieved_row(
        chunk_index=5, chunk_id="1-5", score=2.5,
        text="Twelve villages were surveyed.",
    )
    handwritten = retrieved_row(
        chunk_index=9, chunk_id="1-9", content_type="handwritten_ocr",
        extraction_method="trocr", ocr_confidence=0.90, score=4.0,
        text="Each household received a 200 litre tank.",
    )

    monkeypatch.setattr(retriever, "retrieve", lambda **kwargs: [unreadable, digital, handwritten])

    def fake_chat(messages, num_predict, timeout=180):
        system = messages[0]["content"]
        if system.startswith("Return only PASS"):
            return "PASS"
        return "Each household received a 200 litre tank. [S2]"

    monkeypatch.setattr(llm, "ollama_chat", fake_chat)

    body = client.post(
        "/chat", json={"question": "What did each household get?", "document_name": DOC}
    ).json()

    assert [row["chunk_id"] for row in body["sources"]] == ["1-5", "1-9"]
    assert "Unreadable" not in str(body["sources"])
    assert body["sources"][1]["text"].startswith("Each household")

    # Every label in the answer resolves to the source at that index.
    for label in llm.cited_labels(body["answer"]):
        assert body["sources"][label - 1]["text"] in body["answer"]


def test_the_retrieved_sources_are_scoped_to_one_document(client, uploaded, monkeypatch):
    captured = {}

    def spy(question, sources):
        captured["documents"] = {row["document_name"] for row in sources}
        return "Twelve villages. [S1]", None, list(sources)

    monkeypatch.setattr(llm, "generate_answer_result", spy)

    client.post("/chat", json={"question": "How many villages?", "document_name": DOC})

    assert captured["documents"] == {DOC}


def test_llm_failure_becomes_a_service_error(client, uploaded, monkeypatch):
    def boom(question, sources):
        raise RuntimeError("Ollama is not reachable")

    monkeypatch.setattr(llm, "generate_answer_result", boom)

    response = client.post("/chat", json={"question": "Villages?", "document_name": DOC})

    assert response.status_code == 503
    assert "unavailable" in response.json()["detail"]


@pytest.mark.parametrize(
    "reason",
    ["no_evidence", "low_handwriting_confidence", "untraceable_citation",
     "unsupported_claim", "not_answerable"],
)
def test_chat_returns_exact_refusal_and_reason(client, uploaded, monkeypatch, reason):
    monkeypatch.setattr(
        llm,
        "generate_answer_result",
        lambda question, sources: (llm.NOT_FOUND, reason, []),
    )

    body = client.post(
        "/chat", json={"question": "Anything?", "document_name": DOC}
    ).json()

    assert body["answer"] == (
        "I could not find enough information in the uploaded documents "
        "to answer this confidently."
    )
    assert body["refusal_reason"] == reason
    # A refusal carries no evidence list: citation chips beside a refusal would
    # point at excerpts that produced no answer.
    assert body["sources"] == []


def test_refused_chat_truncation_cannot_leak_a_citation(client, uploaded, monkeypatch):
    """The API keeps the refusal reason and the source list in agreement."""
    monkeypatch.setattr(
        llm, "generate_answer_result",
        lambda question, sources: (llm.NOT_FOUND, "untraceable_citation", []),
    )

    body = client.post(
        "/chat", json={"question": "Anything?", "document_name": DOC}
    ).json()

    assert body["refusal_reason"] == "untraceable_citation"
    assert "[S" not in body["answer"]


def test_source_text_is_capped_for_the_response(client, uploaded, monkeypatch):
    long_excerpt = retrieved_row(text="W" * 900)

    monkeypatch.setattr(retriever, "retrieve", lambda **kwargs: [long_excerpt])

    def fake_chat(messages, num_predict, timeout=180):
        system = messages[0]["content"]
        if system.startswith("Return only PASS"):
            return "PASS"
        return "The document describes this. [S1]"

    monkeypatch.setattr(llm, "ollama_chat", fake_chat)

    body = client.post(
        "/chat", json={"question": "Anything?", "document_name": DOC}
    ).json()

    assert len(body["sources"][0]["text"]) == 400


def test_a_whole_document_request_is_served_coverage_not_ranking(client, uploaded, monkeypatch):
    """The endpoint is where the two evidence windows split, so the choice itself
    is the contract: a summary takes coverage, a fact question keeps relevance, and
    the returned sources are whichever window the model was allowed to label."""
    used = []
    real_overview = retriever.document_overview
    real_retrieve = retriever.retrieve

    monkeypatch.setattr(retriever, "document_overview",
                        lambda **kwargs: used.append("coverage") or real_overview(**kwargs))
    monkeypatch.setattr(retriever, "retrieve",
                        lambda **kwargs: used.append("ranking") or real_retrieve(**kwargs))
    monkeypatch.setattr(llm, "generate_answer_result",
                        lambda question, sources: ("Noted. [S1]", None, list(sources)))

    summary = client.post(
        "/chat", json={"question": "summarize this document", "document_name": DOC}
    ).json()
    fact = client.post(
        "/chat", json={"question": "How many villages were surveyed?", "document_name": DOC}
    ).json()

    assert used == ["coverage", "ranking"]
    assert summary["answer"] == "Noted. [S1]"
    assert all(row["filename"] == DOC for row in summary["sources"])
    assert fact["sources"], "the fact path still returns its own evidence"


def test_cors_defaults_cover_every_origin_the_page_can_come_from():
    """The frontend is served from 5500, and index.html can also be opened off
    the backend's own port; a fourth origin from Live Server must not fail the
    browser check. A wildcard with credentials is not an option."""
    origins = {o.strip() for o in settings.cors_origins.split(",") if o.strip()}

    assert origins >= {
        "http://localhost:5500", "http://127.0.0.1:5500",
        "http://localhost:8000", "http://127.0.0.1:8000",
    }
    assert "*" not in origins


def test_upload_limit_default_is_a_hundred_megabytes():
    assert settings.max_upload_size_mb == 100
    assert settings.max_upload_bytes == 100 * 1024 * 1024


def test_empty_question_is_rejected(client, uploaded):
    assert client.post("/chat", json={"question": "   ", "document_name": DOC}).status_code == 400


def test_delete_clears_both_stores(client, uploaded):
    assert retriever.retrieve("villages", document_name=DOC)

    response = client.delete(f"/documents/{DOC}")

    assert response.status_code == 200
    assert storage.get_chunks(document_name=DOC) == []
    assert retriever.retrieve("villages", document_name=DOC) == []


def test_reupload_replaces_content_rather_than_adding_to_it(client, uploaded):
    client.post("/upload", files={"file": (DOC, "Only the revised figure matters here.", "text/plain")})

    chunks = storage.get_chunks(document_name=DOC)

    assert len(chunks) == 1
    assert "twelve villages" not in chunks[0]["text"]


def test_corrupt_pdf_is_rejected_with_a_readable_message(client):
    response = client.post(
        "/upload", files={"file": ("blank.pdf", b"%PDF-1.4 not really", "application/pdf")}
    )

    assert response.status_code == 422
    assert "Could not open this PDF" in response.json()["detail"]
