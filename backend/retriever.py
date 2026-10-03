"""
Hybrid grounded retriever: Chroma dense search + BM25, fused by reciprocal
rank fusion, then reordered by a MiniLM cross-encoder.

Retrieval is always scoped to one document. With no document selected the
retriever returns nothing rather than searching everything, which is what makes
cross-document leakage structurally impossible.
"""

import logging
import re
from functools import lru_cache
from pathlib import Path

from backend.config import settings
from backend import storage

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

logger = logging.getLogger(__name__)

EMBED_MODEL = "all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
COLLECTION_NAME = "document_chunks"
RRF_K = 60

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "could",
    "do", "does", "for", "from", "how", "i", "in", "is", "it", "me",
    "my", "of", "on", "or", "please", "tell", "that", "the", "this",
    "to", "was", "what", "when", "where", "which", "who", "why", "will",
    "with", "would", "you", "your",
}


def tokens(text: str) -> list[str]:
    return [
        word for word in re.findall(r"[a-z0-9]{2,}", (text or "").lower())
        if word not in STOP_WORDS
    ]


@lru_cache(maxsize=1)
def embedder():
    return SentenceTransformer(EMBED_MODEL)


def candidate_pool(chunk_total: int) -> int:
    """
    How many fused chunks reach the reranker for one document.

    A fixed pool is a coverage trap that scales backwards: 12 chunks are 8% of a
    150-chunk report but 0.8% of a 1,500-chunk book, so past a certain size the
    answer-bearing chunk stops entering the pool at all - and a chunk the
    reranker never sees cannot be recovered by reranking.
    """
    return min(
        max(settings.rerank_candidates, chunk_total // 20),
        settings.rerank_candidates_max,
    )


@lru_cache(maxsize=1)
def reranker():
    return CrossEncoder(RERANK_MODEL)


@lru_cache(maxsize=1)
def collection():
    db_path = Path(settings.data_dir) / "chroma_db"
    db_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(db_path))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def index_document(document_name: str) -> int:
    """
    Embed one document's chunks into Chroma.

    Called once at upload time, not per query. Returns the number of chunks
    written.
    """
    ids, texts, metas = [], [], []

    for chunk in storage.get_chunks(document_name=document_name):
        text = str(chunk.get("text", "")).strip()
        chunk_id = str(chunk.get("id", "")).strip()
        if not text or not chunk_id:
            continue

        ids.append(chunk_id)
        texts.append(text)
        metas.append({
            "document_name": document_name,
            "filename": str(chunk.get("filename", document_name)),
            "content_type": str(chunk.get("content_type", "digital_text")),
        })

    if ids:
        vectors = embedder().encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        ).tolist()
        collection().upsert(
            ids=ids, documents=texts, metadatas=metas, embeddings=vectors
        )

    return len(ids)


def purge_document(document_name: str) -> None:
    """
    Remove a document from the vector index.

    Without this, deleting or re-uploading a file leaves its old vectors
    searchable under the same document name, and the stale text looks perfectly
    grounded to the verifier because it really is in the evidence.
    """
    if not document_name:
        return

    try:
        collection().delete(where={"document_name": document_name})
    except Exception as error:
        logger.warning("Could not purge %s from the vector index: %s", document_name, error)


def _dense_ranking(query: str, document_name: str, limit: int) -> list[str]:
    """Chunk ids ordered by cosine similarity, best first."""
    if limit <= 0:
        return []

    vector = embedder().encode(
        [query], normalize_embeddings=True, show_progress_bar=False
    ).tolist()

    result = collection().query(
        query_embeddings=vector,
        n_results=limit,
        where={"document_name": document_name},
        include=["metadatas", "distances"],
    )

    ids = (result.get("ids") or [[]])[0]
    return [str(chunk_id) for chunk_id in ids]


def _bm25_ranking(query: str, chunks: list[dict], limit: int) -> list[str]:
    """Chunk ids ordered by lexical score, dropping zero-scoring chunks."""
    query_tokens = tokens(query)
    if not query_tokens:
        return []

    corpus = [tokens(str(chunk.get("text", ""))) for chunk in chunks]
    scores = BM25Okapi(corpus).get_scores(query_tokens)

    order = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
    return [
        str(chunks[i].get("id", ""))
        for i in order[:limit]
        if scores[i] > 0 and str(chunks[i].get("id", "")).strip()
    ]


def reciprocal_rank_fusion(rankings: list[list[str]]) -> list[tuple[str, float]]:
    """
    Merge ranked id lists into one ordering.

    RRF scores a chunk by the sum of 1/(k + rank) across the lists it appears
    in, so agreement between the dense and lexical stages is rewarded without
    either stage's raw score scale dominating the other.
    """
    fused: dict[str, float] = {}

    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking):
            if chunk_id:
                fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank + 1)

    return sorted(fused.items(), key=lambda item: item[1], reverse=True)


def _row(chunk: dict, document_name: str, rrf_score: float) -> dict:
    page = chunk.get("page_number")

    return {
        "id": str(chunk.get("id", "")),
        "text": str(chunk.get("text", "")).strip(),
        "filename": str(chunk.get("filename", document_name)),
        "document_name": document_name,
        "chunk_index": int(chunk.get("chunk_index", 0)),
        "chunk_id": str(chunk.get("chunk_id", "")),
        "page_number": int(page) if page is not None else None,
        "content_type": str(chunk.get("content_type", "digital_text")),
        "extraction_method": str(chunk.get("extraction_method", "unknown")),
        "ocr_confidence": chunk.get("ocr_confidence"),
        "rrf_score": round(rrf_score, 6),
    }


def retrieve(query: str, *, top_k: int = None, document_name: str = None) -> list[dict]:
    """
    Return the best evidence chunks for one document.

    Nothing is indexed here; indexing happens once at upload. If the vector
    index has no entry for the document, retrieval fails closed and returns an
    empty list rather than falling back to a global search.
    """
    if not document_name or not str(query or "").strip():
        return []

    if top_k is None:
        top_k = settings.top_k

    chunks = storage.get_chunks(document_name=document_name)
    if not chunks:
        return []

    by_id = {str(chunk.get("id", "")): chunk for chunk in chunks}

    candidates = candidate_pool(len(chunks))
    fused = reciprocal_rank_fusion([
        _dense_ranking(query, document_name, min(candidates, len(chunks))),
        _bm25_ranking(query, chunks, candidates),
    ])

    rows = []
    for chunk_id, rrf_score in fused[:candidates]:
        chunk = by_id.get(chunk_id)
        if chunk and str(chunk.get("text", "")).strip():
            rows.append(_row(chunk, document_name, rrf_score))

    if not rows:
        return []

    rerank_scores = reranker().predict([(query, row["text"]) for row in rows])

    # The cutoff is the leader minus a margin, not a fixed score. Absolute
    # cross-encoder logits depend on how well-formed the passage is, so a fixed
    # floor silently throws away every candidate from a chunked document while
    # still letting through junk on a tidy one.
    cutoff = max(rerank_scores) - settings.rerank_margin

    ranked = []
    dropped_scores = []
    for row, rerank_score in zip(rows, rerank_scores):
        row["rerank_score"] = float(rerank_score)
        if row["rerank_score"] >= cutoff:
            row["score"] = round(row["rerank_score"], 4)
            ranked.append(row)
        else:
            dropped_scores.append(row["rerank_score"])

    if dropped_scores:
        logger.debug(
            "Rerank cutoff %.2f dropped %d/%d candidates; dropped scores=%s",
            cutoff,
            len(dropped_scores),
            len(rows),
            dropped_scores,
        )

    ranked.sort(key=lambda row: (row["rerank_score"], row["rrf_score"]), reverse=True)
    return ranked[:top_k]


def document_overview(document_name: str, limit: int = None) -> list[dict]:
    """
    Evidence for a request about the document as a whole, in reading order.

    Relevance ranking cannot serve this question: no chunk *is* the summary, so
    every candidate scores near -10 against "summarize this document" and the
    window fills with whichever slide happened to share a function word with the
    request. Measured on a 13-page slide deck: the six chunks that came back for
    that request were K-means centroids and a Naive Bayes table, and the model
    was right to refuse them. What a document-level request needs is coverage, so
    one chunk is taken from each equal slice of the document instead of ranking
    the whole document against the request.
    """
    if not document_name:
        return []

    if limit is None:
        limit = settings.overview_chunks

    chunks = storage.get_chunks(document_name=document_name)
    readable = [chunk for chunk in chunks if str(chunk.get("text", "")).strip()]
    if not readable:
        return []

    readable.sort(key=lambda chunk: (int(chunk.get("page_number") or 0),
                                     int(chunk.get("chunk_index") or 0)))

    if limit >= len(readable):
        picked = readable
    else:
        step = len(readable) / limit
        picked = [readable[int(index * step)] for index in range(limit)]

    return [_row(chunk, document_name, 0.0) for chunk in picked]
