"""
Hybrid semantic RAG retriever.
Uses embeddings + TF-IDF, selected-document-only retrieval, and evidence thresholds.
"""

import re
from functools import lru_cache

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from backend import storage
from backend.config import settings

MODEL_NAME = "all-MiniLM-L6-v2"
MIN_HYBRID_SCORE = 0.32
TOP_CANDIDATES = 8

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "could",
    "do", "does", "for", "from", "how", "i", "in", "is", "it", "me",
    "my", "of", "on", "or", "please", "tell", "that", "the", "this",
    "to", "was", "what", "when", "where", "which", "who", "why", "will",
    "with", "would", "you", "your"
}


@lru_cache(maxsize=1)
def get_embedder():
    return SentenceTransformer(MODEL_NAME)


def terms(text: str) -> set[str]:
    return {
        item for item in re.findall(r"[a-z0-9]{3,}", text.lower())
        if item not in STOP_WORDS
    }


def retrieve(query: str, top_k: int = None, document_name: str = None) -> list[dict]:
    if not document_name:
        return []

    if top_k is None:
        top_k = settings.top_k

    chunks = storage.get_chunks(document_name=document_name)
    if not chunks:
        return []

    valid_chunks = []
    texts = []

    for chunk in chunks:
        text = str(chunk.get("text", "")).strip()
        if text:
            valid_chunks.append(chunk)
            texts.append(text)

    if not texts:
        return []

    try:
        embedder = get_embedder()
        document_vectors = embedder.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        query_vector = embedder.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False
        )
        semantic_scores = (document_vectors @ query_vector[0]).tolist()

        vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            sublinear_tf=True,
            stop_words="english"
        )
        matrix = vectorizer.fit_transform(texts)
        lexical_scores = cosine_similarity(
            vectorizer.transform([query]), matrix
        ).flatten().tolist()
    except Exception:
        return []

    question_terms = terms(query)
    ranked = []

    for index, chunk in enumerate(valid_chunks):
        semantic = max(0.0, float(semantic_scores[index]))
        lexical = max(0.0, float(lexical_scores[index]))
        overlap = len(question_terms.intersection(terms(texts[index])))

        hybrid = (0.72 * semantic) + (0.28 * lexical)
        if overlap > 0:
            hybrid += 0.03

        ranked.append((hybrid, semantic, lexical, overlap, index, chunk))

    ranked.sort(reverse=True, key=lambda row: row[0])

    results = []
    for hybrid, semantic, lexical, overlap, index, chunk in ranked[:TOP_CANDIDATES]:
        if hybrid < MIN_HYBRID_SCORE:
            continue

        item = dict(chunk)
        item["score"] = round(hybrid, 4)
        item["semantic_score"] = round(semantic, 4)
        item["lexical_score"] = round(lexical, 4)
        item["term_overlap"] = overlap
        results.append(item)

        if len(results) >= top_k:
            break

    return results


def search(query: str, top_k: int = None, document_name: str = None) -> list[dict]:
    return retrieve(query, top_k, document_name)
