"""
Stage 6-7 tests: embedding, hybrid retrieval and document isolation.

Isolation is the property these tests exist to protect: a question asked about
one document must never be answered from another.
"""

import pytest

from backend import retriever, storage
from backend.config import settings

DOC_A = "isolation_a.txt"
DOC_B = "isolation_b.txt"

TEXT_A = (
    "The Kolar water survey covered twelve villages. "
    "Each household received a 200 litre storage tank in June 2024."
)
TEXT_B = (
    "The Chennai office lease expires on 31 December 2026. "
    "Rent is revised every three years by the property committee."
)


@pytest.fixture(scope="module")
def indexed_pair():
    for name, text in ((DOC_A, TEXT_A), (DOC_B, TEXT_B)):
        storage.clear_document(name)
        retriever.purge_document(name)
        records = [{
            "id": f"{name}-0",
            "filename": name,
            "document_name": name,
            "chunk_index": 0,
            "chunk_id": "1-0",
            "page_number": 1,
            "content_type": "digital_text",
            "extraction_method": "plaintext",
            "text": text,
        }]
        storage.add_chunks(records)
        retriever.index_document(name)

    yield

    for name in (DOC_A, DOC_B):
        storage.clear_document(name)
        retriever.purge_document(name)


def test_retrieval_requires_a_selected_document(indexed_pair):
    assert retriever.retrieve("How many villages?", document_name=None) == []
    assert retriever.retrieve("How many villages?", document_name="") == []


def test_blank_query_returns_nothing(indexed_pair):
    assert retriever.retrieve("   ", document_name=DOC_A) == []


def test_unknown_document_returns_nothing(indexed_pair):
    assert retriever.retrieve("Any question at all?", document_name="never-uploaded.txt") == []


def test_answers_cannot_leak_across_documents(indexed_pair):
    results = retriever.retrieve("How many villages were surveyed?", document_name=DOC_A)

    assert results, "a relevant question must retrieve evidence"
    assert {row["document_name"] for row in results} == {DOC_A}
    assert all("Chennai" not in row["text"] for row in results)


def test_the_other_documents_text_is_unreachable_even_for_its_own_topic(indexed_pair):
    """The lease question is the strongest match for DOC_B. Asked against
    DOC_A it must return nothing rather than the right answer from the wrong
    file."""
    results = retriever.retrieve(
        "When does the Chennai office lease expire?",
        document_name=DOC_A,
    )

    assert all("lease" not in row["text"].lower() for row in results)


def test_retrieved_rows_carry_full_citation_metadata(indexed_pair):
    row = retriever.retrieve(
        "What size tank did each household get?", document_name=DOC_A
    )[0]

    assert row["page_number"] == 1
    assert row["chunk_id"] == "1-0"
    assert row["content_type"] == "digital_text"
    assert row["extraction_method"] == "plaintext"
    assert row["filename"] == DOC_A
    assert abs(row["score"] - row["rerank_score"]) < 1e-4


def test_top_k_is_respected(indexed_pair):
    results = retriever.retrieve(
        "tank villages survey lease rent committee", top_k=1, document_name=DOC_A
    )

    assert len(results) <= 1


def test_the_candidate_pool_grows_with_the_document():
    """A fixed candidate pool is a coverage trap that scales backwards: 12
    candidates cover 8% of a 150-chunk report but only 0.8% of a 1,500-chunk
    book, so on a long document the answer-bearing chunk stops reaching the
    reranker at all - and a chunk the reranker never sees cannot be recovered by
    reranking. The pool must therefore grow with the document and stay bounded."""
    assert retriever.candidate_pool(30) == settings.rerank_candidates
    assert retriever.candidate_pool(150) == settings.rerank_candidates
    assert retriever.candidate_pool(600) > settings.rerank_candidates
    assert retriever.candidate_pool(1433) == settings.rerank_candidates_max
    assert retriever.candidate_pool(120_000) == settings.rerank_candidates_max


def test_rerank_margin_is_measured_from_the_leader_not_an_absolute_score(monkeypatch):
    """Absolute cross-encoder logits track how well-formed a passage is, so a
    fixed cutoff silently throws away every candidate from a chunked document
    while still admitting junk from a tidy one. The leader must always survive
    for the answer stage to judge."""
    name = "margin_probe.txt"
    storage.clear_document(name)
    retriever.purge_document(name)
    records = [
        {
            "id": f"{name}-0", "filename": name, "document_name": name,
            "chunk_index": 0, "chunk_id": "1-0", "page_number": 1,
            "content_type": "digital_text", "extraction_method": "plaintext",
            "text": TEXT_A,
        },
        {
            "id": f"{name}-1", "filename": name, "document_name": name,
            "chunk_index": 1, "chunk_id": "1-1", "page_number": 1,
            "content_type": "digital_text", "extraction_method": "plaintext",
            "text": "Appendix tables of rainfall figures for the same district.",
        },
    ]
    storage.add_chunks(records)
    retriever.index_document(name)

    class FixedReranker:
        def predict(self, pairs):
            return [-7.0, -9.0][:len(pairs)]

    monkeypatch.setattr(retriever, "reranker", lambda: FixedReranker())

    try:
        monkeypatch.setattr(settings, "rerank_margin", 1.5)
        results = retriever.retrieve("villages survey tanks", document_name=name)
        assert [row["rerank_score"] for row in results] == [-7.0]

        monkeypatch.setattr(settings, "rerank_margin", 3.0)
        results = retriever.retrieve("villages survey tanks", document_name=name)
        assert sorted(row["rerank_score"] for row in results) == [-9.0, -7.0]
    finally:
        storage.clear_document(name)
        retriever.purge_document(name)


def test_purge_removes_the_vectors_a_reupload_would_otherwise_reuse(indexed_pair):
    """Deleting only the JSON chunks would leave the old embeddings searchable,
    and stale text looks perfectly grounded because it really is in the file."""
    with_vectors = retriever.collection().count()

    retriever.purge_document(DOC_B)

    assert retriever.collection().count() < with_vectors

    storage.clear_document(DOC_B)
    assert retriever.retrieve("twelve villages storage tank lease", document_name=DOC_B) == []

    storage.add_chunks([{
        "id": f"{DOC_B}-0", "filename": DOC_B, "document_name": DOC_B,
        "chunk_index": 0, "chunk_id": "1-0", "page_number": 1,
        "content_type": "digital_text", "extraction_method": "plaintext",
        "text": TEXT_B,
    }])
    retriever.index_document(DOC_B)


def test_indexing_twice_does_not_duplicate_chunks(indexed_pair):
    before = retriever.collection().count()

    assert retriever.index_document(DOC_A) == 1
    assert retriever.collection().count() == before


def test_rrf_rewards_chunks_both_stages_found():
    fused = dict(retriever.reciprocal_rank_fusion([["a", "b"], ["a", "c"]]))

    # 'a' is credited by both rankings; 'b' and 'c' each appear once at the
    # same depth, so they must tie rather than silently inherit an ordering.
    assert fused["a"] > fused["b"] == fused["c"]
    assert fused["a"] == pytest.approx(2 / (retriever.RRF_K + 1))


def test_off_topic_question_ranks_below_an_on_topic_one(indexed_pair):
    on_topic = retriever.retrieve("How many villages were surveyed?", document_name=DOC_A)
    off_topic = retriever.retrieve("What is the boiling point of mercury?", document_name=DOC_A)

    assert on_topic
    # An off-topic query still hands its best candidate over, because nothing
    # shorter than the leader can be compared against a fixed score; the
    # separation has to show up in the ranking the answer stage receives.
    assert off_topic, "the leader is always available for the model to judge"
    best_off = max(row["rerank_score"] for row in off_topic)
    assert best_off < on_topic[0]["rerank_score"]


def test_settings_drive_the_default_context_size(indexed_pair):
    results = retriever.retrieve("storage tank villages", document_name=DOC_A)

    assert len(results) <= settings.top_k


def _ten_chunk_document(fresh_document, name="overview_deck.pdf"):
    document = fresh_document(name)
    storage.add_chunks([
        {
            "id": f"ov-{i}",
            "filename": document,
            "document_name": document,
            "chunk_index": i,
            "chunk_id": f"{i // 2}-{i}",
            "page_number": (i // 2) + 1,
            "content_type": "digital_text",
            "extraction_method": "pytesseract",
            "text": f"Section {i} explains topic {i}.",
        }
        for i in range(10)
    ])
    return document


def test_a_whole_document_request_takes_coverage_not_relevance(fresh_document):
    """Ranking a deck against "summarize this" has no right answer to find, so the
    window is one chunk from each equal slice of the document, in reading order."""
    document = _ten_chunk_document(fresh_document)

    rows = retriever.document_overview(document, limit=5)

    assert [row["chunk_index"] for row in rows] == [0, 2, 4, 6, 8]
    assert [row["page_number"] for row in rows] == [1, 2, 3, 4, 5]
    assert all(row["document_name"] == document for row in rows)


def test_coverage_uses_its_own_window_rather_than_top_k(fresh_document, monkeypatch):
    """Six excerpts measured as too little for a summary: on a 13-page scanned
    deck the model answered "summarize this document" with the refusal string at
    6 chunks and with the deck's real topics at 10. So the overview window is its
    own setting, wider than the top_k a fact question needs, and a caller that
    passes no limit still gets it."""
    document = _ten_chunk_document(fresh_document)
    # Deliberately below top_k, so a window of this width can only have come from
    # the overview setting rather than the fact-answering one.
    monkeypatch.setattr(settings, "overview_chunks", 4)

    rows = retriever.document_overview(document)

    assert len(rows) == 4
    assert [row["chunk_index"] for row in rows] == [0, 2, 5, 7]


def test_coverage_stops_at_the_document_it_was_asked_about(fresh_document):
    """The same isolation rule as ranking: no selected document means no evidence,
    and an uploaded document cannot stand in for one that was never added."""
    document = _ten_chunk_document(fresh_document)

    assert retriever.document_overview("", limit=4) == []
    assert retriever.document_overview(None, limit=4) == []
    assert retriever.document_overview("not_uploaded.pdf", limit=4) == []
    assert all(row["document_name"] == document
               for row in retriever.document_overview(document, limit=3))


def test_a_short_document_is_not_padded_with_repeats(fresh_document):
    document = _ten_chunk_document(fresh_document, "overview_short.pdf")
    storage.clear_document(document)
    storage.add_chunks([
        {
            "id": "short-0", "filename": document, "document_name": document,
            "chunk_index": 0, "chunk_id": "1-0", "page_number": 1,
            "content_type": "digital_text", "extraction_method": "plaintext",
            "text": "One page of notes.",
        },
        {
            "id": "short-1", "filename": document, "document_name": document,
            "chunk_index": 1, "chunk_id": "1-1", "page_number": 1,
            "content_type": "digital_text", "extraction_method": "plaintext",
            "text": "A second page of notes.",
        },
    ])

    rows = retriever.document_overview(document, limit=6)

    assert len(rows) == 2, "six labels cannot come from two excerpts"


def test_blank_chunks_are_not_offered_as_evidence(fresh_document):
    document = fresh_document("overview_blanks.pdf")
    storage.add_chunks([
        {
            "id": "blank-0", "filename": document, "document_name": document,
            "chunk_index": 0, "chunk_id": "1-0", "page_number": 1,
            "content_type": "digital_text", "extraction_method": "plaintext",
            "text": "   ",
        },
        {
            "id": "blank-1", "filename": document, "document_name": document,
            "chunk_index": 1, "chunk_id": "1-1", "page_number": 1,
            "content_type": "digital_text", "extraction_method": "plaintext",
            "text": "The only readable page.",
        },
    ])

    rows = retriever.document_overview(document, limit=4)

    assert [row["text"] for row in rows] == ["The only readable page."]
