"""
TF-IDF Retriever – finds the most relevant chunks
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from backend import storage
from backend.config import settings


def retrieve(query: str, top_k: int = None, document_name: str = None) -> list[dict]:
    if top_k is None:
        top_k = settings.top_k

    chunks = storage.get_chunks(document_name=document_name)

    if not chunks:
        # Fallback: try without document filter
        chunks = storage.get_chunks()

    if not chunks:
        return []

    texts = [str(c.get("text", "")).strip() for c in chunks]
    valid = [(i, t) for i, t in enumerate(texts) if t]

    if not valid:
        return []

    indices, documents = zip(*valid)

    try:
        vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            sublinear_tf=True,
            max_features=5000
        )
        matrix = vectorizer.fit_transform(documents)
        query_vector = vectorizer.transform([query])
        scores = cosine_similarity(query_vector, matrix).flatten()
    except Exception:
        return []

    order = scores.argsort()[::-1]

    results = []
    for position in order:
        score = float(scores[position])
        if score < 0.05:          # very low threshold so it almost always finds something
            continue
        item = dict(chunks[indices[position]])
        item["score"] = score
        results.append(item)
        if len(results) >= top_k:
            break

    return results


def search(query: str, top_k: int = None, document_name: str = None) -> list[dict]:
    return retrieve(query, top_k, document_name)