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
    best_off = max((row["rerank_score"] for row in off_topic), default=-99.0)
    assert best_off < on_topic[0]["rerank_score"]


def test_settings_drive_the_default_context_size(indexed_pair):
    results = retriever.retrieve("storage tank villages", document_name=DOC_A)

    assert len(results) <= settings.top_k
