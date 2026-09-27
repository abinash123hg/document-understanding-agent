"""
Hybrid grounded retriever: Chroma dense search + BM25 + MiniLM reranker.
Selected-document-only, with evidence thresholds and chunk metadata.
"""

from pathlib import Path
from functools import lru_cache
import re

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

from backend import storage
from backend.config import settings

EMBED_MODEL = "all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
COLLECTION_NAME = "document_chunks"
CANDIDATES = 12
MIN_RERANK_SCORE = -6.0

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "could",
    "do", "does", "for", "from", "how", "i", "in", "is", "it", "me",
    "my", "of", "on", "or", "please", "tell", "that", "the", "this",
    "to", "was", "what", "when", "where", "which", "who", "why", "will",
    "with", "would", "you", "your"
}


def tokens(text: str) -> list[str]:
    return [
        word for word in re.findall(r"[a-z0-9]{2,}", (text or "").lower())
        if word not in STOP_WORDS
    ]


@lru_cache(maxsize=1)
def embedder():
    return SentenceTransformer(EMBED_MODEL)


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
        metadata={"hnsw:space": "cosine"}
    )


def index_document(document_name: str) -> list[dict]:
    chunks = storage.get_chunks(document_name=document_name)
    valid = []
    ids, texts, metas = [], [], []

    for chunk in chunks:
        text = str(chunk.get("text", "")).strip()
        chunk_id = str(chunk.get("id", "")).strip()
        if not text or not chunk_id:
            continue

        valid.append(chunk)
        ids.append(chunk_id)
        texts.append(text)
        metas.append({
            "document_name": document_name,
            "filename": str(chunk.get("filename", document_name)),
            "chunk_index": int(chunk.get("chunk_index", 0)),
            "page_number": int(chunk.get("page_number") or 0)
        })

    if ids:
        vectors = embedder().encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        ).tolist()
        collection().upsert(
            ids=ids, documents=texts, metadatas=metas, embeddings=vectors
        )

    return valid


def retrieve(query: str, top_k: int = None, document_name: str = None) -> list[dict]:
    if not document_name:
        return []

    if top_k is None:
        top_k = settings.top_k

    chunks = index_document(document_name)
    if not chunks:
        return []

    query_vector = embedder().encode(
        [query], normalize_embeddings=True, show_progress_bar=False
    ).tolist()

    dense = collection().query(
        query_embeddings=query_vector,
        n_results=min(CANDIDATES, len(chunks)),
        where={"document_name": document_name},
        include=["documents", "metadatas", "distances"]
    )

    dense_docs = dense.get("documents", [[]])[0]
    dense_meta = dense.get("metadatas", [[]])[0]
    dense_distances = dense.get("distances", [[]])[0]

    candidates = {}
    for text, meta, distance in zip(dense_docs, dense_meta, dense_distances):
        key = (int(meta.get("chunk_index", 0)), text)
        candidates[key] = {
            "text": text,
            "filename": meta.get("filename", document_name),
            "document_name": document_name,
            "chunk_index": int(meta.get("chunk_index", 0)),
            "page_number": int(meta.get("page_number", 0)) or None,
            "dense_score": max(0.0, 1.0 - float(distance))
        }

    corpus_tokens = [tokens(str(chunk.get("text", ""))) for chunk in chunks]
    bm25 = BM25Okapi(corpus_tokens)
    bm25_scores = bm25.get_scores(tokens(query))

    for index in sorted(range(len(chunks)), key=lambda i: bm25_scores[i], reverse=True)[:CANDIDATES]:
        chunk = chunks[index]
        text = str(chunk.get("text", "")).strip()
        key = (int(chunk.get("chunk_index", 0)), text)
        if key not in candidates:
            candidates[key] = {
                "text": text,
                "filename": str(chunk.get("filename", document_name)),
                "document_name": document_name,
                "chunk_index": int(chunk.get("chunk_index", 0)),
                "page_number": int(chunk.get("page_number") or 0) or None,
                "dense_score": 0.0
            }
        candidates[key]["bm25_score"] = float(bm25_scores[index])

    rows = list(candidates.values())
    if not rows:
        return []

    pairs = [(query, row["text"]) for row in rows]
    rerank_scores = reranker().predict(pairs)

    ranked = []
    for row, rerank_score in zip(rows, rerank_scores):
        row["rerank_score"] = float(rerank_score)
        if row["rerank_score"] >= MIN_RERANK_SCORE:
            row["score"] = round(row["rerank_score"], 4)
            ranked.append(row)

    ranked.sort(key=lambda row: row["rerank_score"], reverse=True)
    return ranked[:top_k]


def search(query: str, top_k: int = None, document_name: str = None) -> list[dict]:
    return retrieve(query, top_k, document_name)
