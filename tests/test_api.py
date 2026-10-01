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


def test_chat_for_an_unuploaded_document_refuses(client):
    body = client.post(
        "/chat",
        json={"question": "How many villages?", "document_name": "not-uploaded.txt"},
    ).json()

    assert body["answer"] == llm.NOT_FOUND
    assert body["sources"] == []


def test_chat_returns_citable_sources(client, uploaded, monkeypatch):
    monkeypatch.setattr(
        llm, "generate_answer",
        lambda question, sources: "Twelve villages were surveyed. [S1]",
    )

    body = client.post(
        "/chat", json={"question": "How many villages?", "document_name": DOC}
    ).json()

    assert body["answer"] == "Twelve villages were surveyed. [S1]"
    assert body["document_name"] == DOC
    source = body["sources"][0]
    assert set(source) >= {"filename", "chunk_index", "chunk_id", "page_number",
                           "content_type", "extraction_method", "score", "text"}
    assert source["content_type"] == "digital_text"


def test_the_retrieved_sources_are_scoped_to_one_document(client, uploaded, monkeypatch):
    captured = {}

    def spy(question, sources):
        captured["documents"] = {row["document_name"] for row in sources}
        return "Twelve villages. [S1]"

    monkeypatch.setattr(llm, "generate_answer", spy)

    client.post("/chat", json={"question": "How many villages?", "document_name": DOC})

    assert captured["documents"] == {DOC}


def test_llm_failure_becomes_a_service_error(client, uploaded, monkeypatch):
    def boom(question, sources):
        raise RuntimeError("Ollama is not reachable")

    monkeypatch.setattr(llm, "generate_answer", boom)

    response = client.post("/chat", json={"question": "Villages?", "document_name": DOC})

    assert response.status_code == 503
    assert "unavailable" in response.json()["detail"]


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
