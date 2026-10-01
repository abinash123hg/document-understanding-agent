"""
Stage 8 tests: answer generation guards.

The three behaviours that stop hallucination are checked here without a
network call: the refusal wording, the citation requirement, and the rule that
a missing Ollama model is never fetched on demand.
"""

import pytest
import requests

from backend import llm
from backend.config import settings


def digital_chunk(text="The tank holds 200 litres.", **overrides):
    chunk = {
        "id": "c1",
        "text": text,
        "filename": "notes.txt",
        "document_name": "notes.txt",
        "chunk_index": 0,
        "chunk_id": "0-0",
        "page_number": 3,
        "content_type": "digital_text",
        "extraction_method": "plaintext",
        "score": 4.2,
    }
    chunk.update(overrides)
    return chunk


def test_refusal_message_is_exact():
    assert llm.NOT_FOUND == (
        "I could not find enough information in the uploaded documents "
        "to answer this confidently."
    )


def test_generate_answer_refuses_without_sources():
    assert llm.generate_answer("How much water?", []) == llm.NOT_FOUND


def test_empty_sources_report_no_evidence_reason():
    assert llm.generate_answer_result("How much water?", []) == (
        llm.NOT_FOUND, "no_evidence"
    )


def test_build_context_labels_every_excerpt_with_page_and_type():
    context = llm.build_context([digital_chunk(), digital_chunk(page_number=7)])

    assert "[S1 | notes.txt, page 3 | digital_text]" in context
    assert "[S2 | notes.txt, page 7 | digital_text]" in context


def test_build_context_skips_blank_sources():
    assert llm.build_context([digital_chunk("   ")]) == ""


def test_citation_check_accepts_labels_only():
    assert llm._is_cited("The tank holds 200 litres. [S1]")
    assert not llm._is_cited("The tank holds about 100 litres, probably.")


def test_labels_must_point_at_an_excerpt_that_was_supplied():
    assert llm.citations_are_real("The tank holds 200 litres. [S1]", 2)
    assert llm.citations_are_real("[S1] says 200 litres and [S2] says June", 2)
    assert not llm.citations_are_real("The tank holds 200 litres. [S4]", 2)
    assert not llm.citations_are_real("The tank holds 200 litres.", 2)


def test_fabricated_label_is_refused_without_reaching_the_verifier(monkeypatch):
    calls = []

    def fake_chat(messages, num_predict, timeout=180):
        calls.append(messages[0]["content"])
        return "The tank holds 200 litres. [S7]"

    monkeypatch.setattr(llm, "ollama_chat", fake_chat)

    # one excerpt was supplied, so [S7] resolves to nothing
    assert llm.generate_answer("Capacity?", [digital_chunk()]) == llm.NOT_FOUND
    assert len(calls) == 2, "the label gets one correction attempt"
    assert not any(system.startswith("Return only PASS") for system in calls), \
        "an invented citation must never be sent for verification"


def test_low_confidence_handwriting_is_not_answerable():
    sources = [digital_chunk(content_type="handwritten_ocr", ocr_confidence=0.11)]

    assert not llm._evidence_is_legible(sources)
    assert llm.generate_answer("What is written?", sources) == llm.NOT_FOUND
    assert llm.generate_answer_result("What is written?", sources) == (
        llm.NOT_FOUND, "low_handwriting_confidence"
    )


def test_digital_evidence_needs_no_ocr_confidence():
    assert llm._evidence_is_legible([digital_chunk()])
    assert llm._evidence_is_legible(
        [digital_chunk(content_type="handwritten_ocr", ocr_confidence=0.95)]
    )


def test_uncited_draft_is_replaced_by_the_refusal(monkeypatch):
    monkeypatch.setattr(
        llm, "ollama_chat", lambda messages, num_predict, timeout=180: "A plain guess."
    )

    assert llm.generate_answer("Capacity?", [digital_chunk()]) == llm.NOT_FOUND


def test_uncited_draft_gets_one_retry_then_still_needs_a_citation(monkeypatch):
    calls = []

    def fake_chat(messages, num_predict, timeout=180):
        calls.append(messages[1]["content"])
        if messages[0]["content"].startswith("Return only PASS"):
            return "PASS"
        # First draft omits the label, the retry supplies it.
        return "The tank holds 200 litres." if len(calls) == 1 else "The tank holds 200 litres. [S1]"

    monkeypatch.setattr(llm, "ollama_chat", fake_chat)

    assert llm.generate_answer("Capacity?", [digital_chunk()]) == "The tank holds 200 litres. [S1]"
    assert any("THIS ANSWER CANNOT BE USED AS-IS" in message for message in calls), \
        "the retry must ask for the citation rather than relax the requirement"


def test_two_uncited_drafts_are_refused_without_calling_the_verifier(monkeypatch):
    calls = []

    def fake_chat(messages, num_predict, timeout=180):
        calls.append(messages[0]["content"])
        return "Still no label."

    monkeypatch.setattr(llm, "ollama_chat", fake_chat)

    assert llm.generate_answer("Capacity?", [digital_chunk()]) == llm.NOT_FOUND
    assert len(calls) == 2, "one retry only - a refusal must not cost a verification"


def test_draft_is_rejected_when_the_verifier_fails(monkeypatch):
    calls = []

    def fake_chat(messages, num_predict, timeout=180):
        calls.append(messages[0]["content"])
        return "FAIL" if len(calls) > 1 else "The tank holds 500 litres. [S1]"

    monkeypatch.setattr(llm, "ollama_chat", fake_chat)

    assert llm.generate_answer("Capacity?", [digital_chunk()]) == llm.NOT_FOUND
    assert len(calls) == 2, "the draft must be verified before it is returned"


def test_verifier_refusal_reports_unsupported_claim(monkeypatch):
    monkeypatch.setattr(
        llm, "ollama_chat",
        lambda messages, num_predict, timeout=180: (
            "FAIL" if messages[0]["content"].startswith("Return only PASS")
            else "The tank holds 500 litres. [S1]"
        ),
    )

    assert llm.generate_answer_result("Capacity?", [digital_chunk()]) == (
        llm.NOT_FOUND, "unsupported_claim"
    )


def test_verified_draft_is_returned(monkeypatch):
    def fake_chat(messages, num_predict, timeout=180):
        return "PASS" if messages[0]["content"].startswith("Return only PASS") else "The tank holds 200 litres. [S1]"

    monkeypatch.setattr(llm, "ollama_chat", fake_chat)

    assert llm.generate_answer("Capacity?", [digital_chunk()]) == "The tank holds 200 litres. [S1]"
    assert llm.generate_answer_result("Capacity?", [digital_chunk()]) == (
        "The tank holds 200 litres. [S1]", None
    )


def test_exact_refusal_draft_short_circuits(monkeypatch):
    calls = []

    def fake_chat(messages, num_predict, timeout=180):
        calls.append(messages)
        return llm.NOT_FOUND

    monkeypatch.setattr(llm, "ollama_chat", fake_chat)

    assert llm.generate_answer("Anything?", [digital_chunk()]) == llm.NOT_FOUND
    assert len(calls) == 1, "a refusal must not be sent through the verifier"


def test_verifier_failure_is_fail_closed(monkeypatch):
    def boom(messages, num_predict, timeout=180):
        raise requests.exceptions.ConnectionError("ollama down")

    monkeypatch.setattr(llm, "ollama_chat", boom)

    assert llm.verify_answer("q", "context", "draft") is False


class FakeResponse:
    def __init__(self, payload, status=200):
        self.payload = payload
        self.status = status

    def raise_for_status(self):
        if self.status != 200:
            raise requests.exceptions.HTTPError(f"HTTP {self.status}")

    def json(self):
        return self.payload

    @property
    def status_code(self):
        return self.status


def test_missing_model_is_never_pulled_automatically(monkeypatch):
    """Ollama downloads any model it is asked for, so the guard has to stop the
    request before /api/chat is reached."""
    called = []

    def fake_get(url, timeout=None):
        return FakeResponse({"models": [{"name": "llama3:latest"}]})

    def fake_post(url, json=None, timeout=None):
        called.append(url)
        raise AssertionError("/api/chat must not be called for a missing model")

    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr(requests, "post", fake_post)

    with pytest.raises(RuntimeError, match="is not pulled in Ollama"):
        llm.ollama_chat([{"role": "user", "content": "hi"}], num_predict=8)

    assert called == []


def test_configured_model_passes_the_guard_and_reaches_chat(monkeypatch):
    sent = {}

    def fake_get(url, timeout=None):
        return FakeResponse({"models": [{"name": settings.llm_model}]})

    def fake_post(url, json=None, timeout=None):
        sent["url"] = url
        sent["payload"] = json
        return FakeResponse({"message": {"content": "OK"}})

    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr(requests, "post", fake_post)

    assert llm.ollama_chat([{"role": "user", "content": "hi"}], num_predict=8) == "OK"
    assert sent["payload"]["model"] == settings.llm_model
    assert sent["payload"]["options"]["temperature"] == 0.0


def test_unreachable_ollama_reports_the_fix(monkeypatch):
    def refuse(url, timeout=None):
        raise requests.exceptions.ConnectionError("connection refused")

    monkeypatch.setattr(requests, "get", refuse)

    with pytest.raises(RuntimeError, match="ollama serve"):
        llm._require_model()

    assert llm.is_ollama_reachable() is False
    assert llm.is_ollama_ready() is False


def test_generation_errors_surface_as_runtime_error(monkeypatch):
    monkeypatch.setattr(
        llm, "ollama_chat",
        lambda messages, num_predict, timeout=180: (_ for _ in ()).throw(
            requests.exceptions.ReadTimeout("slow")
        ),
    )

    with pytest.raises(RuntimeError):
        llm.generate_answer("Capacity?", [digital_chunk()])
