"""
Stage 8 tests: answer generation guards.

The behaviours that stop hallucination are checked here without a network call:
the refusal wording, the citation requirement, the rule that a missing Ollama
model is never fetched on demand, and the contract that the excerpts returned to
the caller are exactly the ones the model was allowed to label.
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


def approve(monkeypatch, draft="The tank holds 200 litres. [S1]"):
    """Patch ollama_chat so the draft is accepted by the verifier."""
    calls = []

    def fake_chat(messages, num_predict, timeout=180):
        calls.append(messages)
        if messages[0]["content"].startswith("Return only PASS"):
            return "PASS"
        return draft

    monkeypatch.setattr(llm, "ollama_chat", fake_chat)
    return calls


def test_refusal_message_is_exact():
    assert llm.NOT_FOUND == (
        "I could not find enough information in the uploaded documents "
        "to answer this confidently."
    )


def test_generate_answer_result_refuses_without_sources():
    assert llm.generate_answer_result("How much water?", []) == (
        llm.NOT_FOUND, "no_evidence", []
    )


def test_empty_sources_report_no_evidence_reason():
    assert llm.generate_answer_result("How much water?", []) == (
        llm.NOT_FOUND, "no_evidence", []
    )


def test_blank_excerpt_reports_no_evidence_before_confidence_gate():
    blank_handwriting = digital_chunk(
        "  ", content_type="handwritten_ocr", ocr_confidence=0.01
    )

    assert llm.generate_answer_result("How much water?", [blank_handwriting]) == (
        llm.NOT_FOUND, "no_evidence", []
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

    # two excerpts were supplied, so [S7] names evidence that was never shown
    two = [digital_chunk(), digital_chunk("Six households joined in June.", chunk_index=1)]

    assert llm.generate_answer_result("Capacity?", two) == (
        llm.NOT_FOUND, "untraceable_citation", []
    )
    assert len(calls) == 2, "the label gets one correction attempt"
    assert not any(system.startswith("Return only PASS") for system in calls), \
        "an invented citation must never be sent for verification"


def test_low_confidence_handwriting_is_not_answerable():
    sources = [digital_chunk(content_type="handwritten_ocr", ocr_confidence=0.11)]

    assert not llm._is_trusted_excerpt(sources[0])
    assert llm.generate_answer_result("What is written?", sources) == (
        llm.NOT_FOUND, "low_handwriting_confidence", []
    )


def test_all_garbled_handwriting_refuses_with_exact_reason():
    sources = [
        digital_chunk(content_type="handwritten_ocr", ocr_confidence=0.05),
        digital_chunk("Unreadable second crop", content_type="handwritten_ocr", ocr_confidence=0.11),
    ]

    answer, reason, trusted = llm.generate_answer_result("What is written?", sources)

    assert answer == llm.NOT_FOUND
    assert answer == (
        "I could not find enough information in the uploaded documents "
        "to answer this confidently."
    )
    assert reason == "low_handwriting_confidence"
    assert trusted == []


def test_model_decline_with_real_excerpt_is_not_answerable(monkeypatch):
    monkeypatch.setattr(
        llm, "ollama_chat", lambda messages, num_predict, timeout=180: llm.NOT_FOUND
    )

    answer, reason, trusted = llm.generate_answer_result(
        "What is the answer?", [digital_chunk()]
    )

    assert answer == llm.NOT_FOUND
    assert reason == "not_answerable"
    assert reason != "no_evidence"
    assert trusted == []


def test_digital_evidence_needs_no_ocr_confidence():
    assert llm._is_trusted_excerpt(digital_chunk())
    assert llm._is_trusted_excerpt(
        digital_chunk(content_type="handwritten_ocr", ocr_confidence=0.95)
    )


def test_an_ocr_score_on_a_digital_excerpt_still_gates_it():
    """
    A Tesseract reading of a page image arrives tagged digital_text, because the
    page really is printed - but it is still OCR output, and an unreliable one
    must not become citable evidence because of the label it happens to carry.
    """
    assert not llm._is_trusted_excerpt(digital_chunk(ocr_confidence=0.22))
    assert llm._is_trusted_excerpt(digital_chunk(ocr_confidence=0.88))


def test_a_misnumbered_label_on_one_excerpt_is_repaired_not_refused(monkeypatch):
    """
    A real 3b run stated the right fact and signed it [S2] against a single
    excerpt, which the guard read as a fabricated citation. Refusing a correct
    answer over a label typo is the false negative users see as "it never
    answers", and with one excerpt there is no other evidence to misattribute to.
    """
    approve(monkeypatch, draft="The tank holds 200 litres. [S2]")

    answer, reason, trusted = llm.generate_answer_result(
        "How much water?", [digital_chunk()]
    )

    assert reason is None
    assert answer == "The tank holds 200 litres. [S1]"
    assert trusted == [digital_chunk()]


def test_a_doubled_label_on_one_excerpt_becomes_one_chip(monkeypatch):
    """
    A real 3b run closed its sentence with '. [S1] [S2]'. Renumbering alone would
    leave two identical chips pointing at the same excerpt, so a run of labels
    collapses into the single label that excerpt has.
    """
    approve(monkeypatch, draft="The tank holds 200 litres. [S1] [S2]")

    answer, reason, trusted = llm.generate_answer_result(
        "How much water?", [digital_chunk()]
    )

    assert reason is None
    assert answer == "The tank holds 200 litres. [S1]"
    assert trusted == [digital_chunk()]


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
        llm.NOT_FOUND, "untraceable_citation", []
    )


def test_single_unlabeled_answer_takes_its_label_from_the_only_excerpt(monkeypatch):
    """Which excerpt the answer came from is already settled when exactly one
    was supplied, so a missing label is a formatting failure, not a grounding
    one. The verifier still has to approve every claim."""
    calls = approve(monkeypatch, draft="The tank holds 200 litres.")
    excerpt = digital_chunk()

    answer, reason, trusted = llm.generate_answer_result("Capacity?", [excerpt])

    assert answer == "The tank holds 200 litres. [S1]"
    assert reason is None
    assert trusted == [excerpt], "the one excerpt is the one the label points at"
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

    answer, reason, _ = llm.generate_answer_result("Capacity?", sources)

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
        llm.NOT_FOUND, "untraceable_citation", []
    )
    assert len(calls) == 2, "one retry only - a refusal must not cost a verification"


def test_unreadable_handwriting_never_reaches_the_model(monkeypatch):
    """A garbled OCR line stays in the candidate list, but a model that is shown
    it will cite it, and the number it carries is not evidence."""
    calls = approve(monkeypatch)
    legible = digital_chunk("The tank holds 200 litres.")
    garbled = digital_chunk(
        "The tank holds 5000 litres.",
        content_type="handwritten_ocr", ocr_confidence=0.05,
    )

    answer, reason, trusted = llm.generate_answer_result("Capacity?", [legible, garbled])

    assert answer == "The tank holds 200 litres. [S1]"
    assert reason is None
    assert trusted == [legible], "the filtered excerpt must not be returned as a source"
    assert all("5000" not in str(message) for message in calls), \
        "an excerpt below the confidence floor must not be citable text"


def test_returned_sources_are_the_numbered_excerpts_in_order(monkeypatch):
    """[S2] must resolve to sources[1] for the reader, so the list handed back
    is the list that was labelled, in the same order, with nothing filtered in
    front of it shifting the numbering."""
    approve(monkeypatch, draft="Six households joined in June. [S2]")

    low_page = digital_chunk(
        "Unreadable line", content_type="handwritten_ocr", ocr_confidence=0.02
    )
    digital = digital_chunk("The tank holds 200 litres.")
    handwritten = digital_chunk(
        "Six households joined in June.",
        content_type="handwritten_ocr", ocr_confidence=0.80, chunk_index=4,
    )

    answer, reason, trusted = llm.generate_answer_result(
        "Who joined?", [low_page, digital, handwritten]
    )

    assert reason is None
    assert trusted == [digital, handwritten]
    assert low_page not in trusted
    label = int(answer.split("[S")[1].split("]")[0])
    assert trusted[label - 1] is handwritten, "the cited label points at the right excerpt"
    assert f"[S{label} | {handwritten['filename']}, page " in llm.build_context(trusted)


def test_refusal_after_verification_returns_no_sources(monkeypatch):
    """Evidence that produced no accepted answer is not evidence, and listing it
    beside a refusal would give the reader citation chips for nothing."""
    monkeypatch.setattr(
        llm, "ollama_chat",
        lambda messages, num_predict, timeout=180: (
            "FAIL" if messages[0]["content"].startswith("Return only PASS")
            else "The tank holds 500 litres. [S1]"
        ),
    )

    answer, reason, trusted = llm.generate_answer_result("Capacity?", [digital_chunk()])

    assert answer == llm.NOT_FOUND
    assert reason == "unsupported_claim"
    assert trusted == []


def test_draft_is_rejected_when_the_verifier_fails(monkeypatch):
    calls = []

    def fake_chat(messages, num_predict, timeout=180):
        calls.append(messages[0]["content"])
        return "FAIL" if len(calls) > 1 else "The tank holds 500 litres. [S1]"

    monkeypatch.setattr(llm, "ollama_chat", fake_chat)

    assert llm.generate_answer_result("Capacity?", [digital_chunk()]) == (
        llm.NOT_FOUND, "unsupported_claim", []
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
        llm.NOT_FOUND, "unsupported_claim", []
    )


def test_mixed_evidence_answer_still_requires_real_citation_and_verification(monkeypatch):
    calls = approve(monkeypatch)
    sources = [
        digital_chunk("The tank holds 200 litres."),
        digital_chunk(
            "Six households joined in June.",
            content_type="handwritten_ocr", ocr_confidence=0.71,
        ),
    ]

    answer, reason, trusted = llm.generate_answer_result("Capacity?", sources)

    assert answer == "The tank holds 200 litres. [S1]"
    assert reason is None
    assert trusted == sources
    assert llm.citations_are_real(answer, 2)
    assert len(calls) == 2
    assert calls[1][0]["content"].startswith("Return only PASS")


def test_verified_draft_is_returned(monkeypatch):
    excerpt = digital_chunk()

    approve(monkeypatch)

    assert llm.generate_answer_result("Capacity?", [excerpt]) == (
        "The tank holds 200 litres. [S1]", None, [excerpt]
    )


def test_exact_refusal_draft_short_circuits(monkeypatch):
    calls = []

    def fake_chat(messages, num_predict, timeout=180):
        calls.append(messages)
        return llm.NOT_FOUND

    monkeypatch.setattr(llm, "ollama_chat", fake_chat)

    assert llm.generate_answer_result("Anything?", [digital_chunk()]) == (
        llm.NOT_FOUND, "not_answerable", []
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


def test_a_different_model_is_not_accepted_as_a_substitute(monkeypatch):
    """The guard compares against the configured name, so the presence of any
    other model must not let a request through under a different model."""
    monkeypatch.setattr(
        requests, "get",
        lambda url, timeout=None: FakeResponse({"models": [{"name": "qwen2.5:7b"}]}),
    )

    assert llm.is_ollama_ready() is False
    with pytest.raises(RuntimeError, match="is not pulled in Ollama"):
        llm._require_model()


def test_latest_tag_counts_as_the_configured_model(monkeypatch):
    """Ollama reports 'qwen2.5:3b:latest' for a model pulled without a tag. It
    is the same weights as 'qwen2.5:3b', so refusing it would be a false alarm."""
    monkeypatch.setattr(settings, "llm_model", "qwen2.5:3b")
    monkeypatch.setattr(
        requests, "get",
        lambda url, timeout=None: FakeResponse(
            {"models": [{"name": "qwen2.5:3b:latest"}]}
        ),
    )

    assert llm.is_ollama_ready() is True
    llm._require_model()


def test_bare_name_counts_when_the_list_shares_a_prefix(monkeypatch):
    """'qwen2.5:3b' pulled while 'qwen2.5:7b' is also present is still ready,
    and 'qwen2.5:1.5b' beside it must not satisfy the check."""
    monkeypatch.setattr(settings, "llm_model", "qwen2.5:3b")

    assert llm.model_is_pulled({"qwen2.5:3b", "qwen2.5:1.5b"})
    assert not llm.model_is_pulled({"qwen2.5:1.5b", "qwen2.5:7b"})


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
