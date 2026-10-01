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


def test_generate_answer_result_refuses_without_sources():
    assert llm.generate_answer_result("How much water?", []) == (
        llm.NOT_FOUND, "no_evidence"
    )


def test_empty_sources_report_no_evidence_reason():
    assert llm.generate_answer_result("How much water?", []) == (
        llm.NOT_FOUND, "no_evidence"
    )


def test_blank_excerpt_reports_no_evidence_before_confidence_gate():
    blank_handwriting = digital_chunk(
        "  ", content_type="handwritten_ocr", ocr_confidence=0.01
    )

    assert llm.generate_answer_result("How much water?", [blank_handwriting]) == (
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
    assert llm.generate_answer_result("Capacity?", [digital_chunk()]) == (
        llm.NOT_FOUND, "untraceable_citation"
    )
    assert len(calls) == 2, "the label gets one correction attempt"
    assert not any(system.startswith("Return only PASS") for system in calls), \
        "an invented citation must never be sent for verification"


def test_low_confidence_handwriting_is_not_answerable():
    sources = [digital_chunk(content_type="handwritten_ocr", ocr_confidence=0.11)]

    assert not llm._is_trusted_excerpt(sources[0])
    assert llm.generate_answer_result("What is written?", sources) == (
        llm.NOT_FOUND, "low_handwriting_confidence"
    )


def test_all_garbled_handwriting_refuses_with_exact_reason():
    sources = [
        digital_chunk(content_type="handwritten_ocr", ocr_confidence=0.05),
        digital_chunk("Unreadable second crop", content_type="handwritten_ocr", ocr_confidence=0.11),
    ]

    answer, reason = llm.generate_answer_result("What is written?", sources)

    assert answer == llm.NOT_FOUND
    assert answer == (
        "I could not find enough information in the uploaded documents "
        "to answer this confidently."
    )
    assert reason == "low_handwriting_confidence"


def test_model_decline_with_real_excerpt_is_not_answerable(monkeypatch):
    monkeypatch.setattr(
        llm, "ollama_chat", lambda messages, num_predict, timeout=180: llm.NOT_FOUND
    )

    answer, reason = llm.generate_answer_result(
        "What is the answer?", [digital_chunk()]
    )

    assert answer == llm.NOT_FOUND
    assert reason == "not_answerable"
    assert reason != "no_evidence"


def test_digital_evidence_needs_no_ocr_confidence():
    assert llm._is_trusted_excerpt(digital_chunk())
    assert llm._is_trusted_excerpt(
        digital_chunk(content_type="handwritten_ocr", ocr_confidence=0.95)
    )


def test_confident_handwriting_is_still_shown_alongside_digital_text():
    sources = [
        digital_chunk("The tank holds 200 litres."),
        digital_chunk(
            "Six households joined in June.",
            content_type="handwritten_ocr", ocr_confidence=0.62,
        ),
    ]

    assert [llm._is_trusted_excerpt(source) for source in sources] == [True, True]


def test_uncited_draft_over_two_excerpts_is_replaced_by_the_refusal(monkeypatch):
    monkeypatch.setattr(
        llm, "ollama_chat", lambda messages, num_predict, timeout=180: "A plain guess."
    )

    sources = [digital_chunk(), digital_chunk("Six households joined in June.")]

    assert llm.generate_answer_result("Capacity?", sources) == (
        llm.NOT_FOUND, "untraceable_citation"
    )


def test_single_unlabeled_answer_takes_its_label_from_the_only_excerpt(monkeypatch):
    """Which excerpt the answer came from is already settled when exactly one
    was supplied, so a missing label is a formatting failure, not a grounding
    one. The verifier still has to approve every claim."""
    calls = []

    def fake_chat(messages, num_predict, timeout=180):
        calls.append(messages)
        if messages[0]["content"].startswith("Return only PASS"):
            return "PASS"
        return "The tank holds 200 litres."

    monkeypatch.setattr(llm, "ollama_chat", fake_chat)

    answer, reason = llm.generate_answer_result("Capacity?", [digital_chunk()])

    assert answer == "The tank holds 200 litres. [S1]"
    assert reason is None
    assert len(calls) == 2, "one draft plus one verification, no correction retry"
    assert not any("THIS ANSWER CANNOT BE USED AS-IS" in str(message)
                   for message in calls)


def test_uncited_draft_gets_one_retry_then_still_needs_a_citation(monkeypatch):
    calls = []
    sources = [digital_chunk(), digital_chunk("Six households joined in June.")]

    def fake_chat(messages, num_predict, timeout=180):
        calls.append(messages[1]["content"])
        if messages[0]["content"].startswith("Return only PASS"):
            return "PASS"
        # First draft omits the label, the retry supplies it.
        return "The tank holds 200 litres." if len(calls) == 1 else "The tank holds 200 litres. [S1]"

    monkeypatch.setattr(llm, "ollama_chat", fake_chat)

    answer, reason = llm.generate_answer_result("Capacity?", sources)

    assert answer == "The tank holds 200 litres. [S1]"
    assert reason is None
    assert any("THIS ANSWER CANNOT BE USED AS-IS" in message for message in calls), \
        "the retry must ask for the citation rather than relax the requirement"
    assert llm.citations_are_real(answer, 2)


def test_two_uncited_drafts_are_refused_without_calling_the_verifier(monkeypatch):
    calls = []
    sources = [digital_chunk(), digital_chunk("Six households joined in June.")]

    def fake_chat(messages, num_predict, timeout=180):
        calls.append(messages[0]["content"])
        return "Still no label."

    monkeypatch.setattr(llm, "ollama_chat", fake_chat)

    assert llm.generate_answer_result("Capacity?", sources) == (
        llm.NOT_FOUND, "untraceable_citation"
    )
    assert len(calls) == 2, "one retry only - a refusal must not cost a verification"


def test_unreadable_handwriting_never_reaches_the_model(monkeypatch):
    """A garbled OCR line stays in the candidate list, but a model that is shown
    it will cite it, and the number it carries is not evidence."""
    calls = []
    sources = [
        digital_chunk("The tank holds 200 litres."),
        digital_chunk(
            "The tank holds 5000 litres.",
            content_type="handwritten_ocr", ocr_confidence=0.05,
        ),
    ]

    def fake_chat(messages, num_predict, timeout=180):
        calls.append(messages)
        if messages[0]["content"].startswith("Return only PASS"):
            return "PASS"
        return "The tank holds 200 litres. [S1]"

    monkeypatch.setattr(llm, "ollama_chat", fake_chat)

    answer, reason = llm.generate_answer_result("Capacity?", sources)

    assert answer == "The tank holds 200 litres. [S1]"
    assert reason is None
    assert all("5000" not in str(message) for message in calls), \
        "an excerpt below the confidence floor must not be citable text"


def test_draft_is_rejected_when_the_verifier_fails(monkeypatch):
    calls = []

    def fake_chat(messages, num_predict, timeout=180):
        calls.append(messages[0]["content"])
        return "FAIL" if len(calls) > 1 else "The tank holds 500 litres. [S1]"

    monkeypatch.setattr(llm, "ollama_chat", fake_chat)

    assert llm.generate_answer_result("Capacity?", [digital_chunk()]) == (
        llm.NOT_FOUND, "unsupported_claim"
    )
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


def test_mixed_evidence_answer_still_requires_real_citation_and_verification(monkeypatch):
    calls = []
    sources = [
        digital_chunk("The tank holds 200 litres."),
        digital_chunk(
            "Six households joined in June.",
            content_type="handwritten_ocr", ocr_confidence=0.71,
        ),
    ]

    def fake_chat(messages, num_predict, timeout=180):
        calls.append(messages)
        if messages[0]["content"].startswith("Return only PASS"):
            return "PASS"
        return "The tank holds 200 litres. [S1]"

    monkeypatch.setattr(llm, "ollama_chat", fake_chat)

    answer, reason = llm.generate_answer_result("Capacity?", sources)

    assert answer == "The tank holds 200 litres. [S1]"
    assert reason is None
    assert llm.citations_are_real(answer, 2)
    assert len(calls) == 2
    assert calls[1][0]["content"].startswith("Return only PASS")


def test_verified_draft_is_returned(monkeypatch):
    def fake_chat(messages, num_predict, timeout=180):
        return "PASS" if messages[0]["content"].startswith("Return only PASS") else "The tank holds 200 litres. [S1]"

    monkeypatch.setattr(llm, "ollama_chat", fake_chat)

    assert llm.generate_answer_result("Capacity?", [digital_chunk()]) == (
        "The tank holds 200 litres. [S1]", None
    )


def test_exact_refusal_draft_short_circuits(monkeypatch):
    calls = []

    def fake_chat(messages, num_predict, timeout=180):
        calls.append(messages)
        return llm.NOT_FOUND

    monkeypatch.setattr(llm, "ollama_chat", fake_chat)

    assert llm.generate_answer_result("Anything?", [digital_chunk()]) == (
        llm.NOT_FOUND, "not_answerable"
    )
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
        llm.generate_answer_result("Capacity?", [digital_chunk()])
