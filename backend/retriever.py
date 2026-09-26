"""
TF-IDF Retriever – finds the most relevant chunks for a question
Works well with small models under 2GB
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from backend import storage
from backend.config import settings


def retrieve(query: str, top_k: int = None, document_name: str = None) -> list[dict]:
    """
    Return the most relevant chunks for the question.
    Only returns chunks that pass the minimum similarity threshold.
    """
    if top_k is None:
        top_k = settings.top_k

    # Get chunks (optionally filtered by document)
    chunks = storage.get_chunks(document_name=document_name)

    if not chunks:
        return []

    texts = [str(c.get("text", "")).strip() for c in chunks]
    valid = [(i, t) for i, t in enumerate(texts) if t]

    if not valid:
        return []

    indices, documents = zip(*valid)

    # Build TF-IDF matrix
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        sublinear_tf=True,
        max_features=8000
    )

    try:
        matrix = vectorizer.fit_transform(documents)
        query_vector = vectorizer.transform([query])
        scores = cosine_similarity(query_vector, matrix).flatten()
    except Exception:
        return []

    # Sort by score (highest first)
    order = scores.argsort()[::-1]

    results = []
    for position in order:
        score = float(scores[position])

        # Only keep chunks that are similar enough
        if score < settings.min_similarity:
            continue

        item = dict(chunks[indices[position]])
        item["score"] = score
        results.append(item)

        if len(results) >= top_k:
            break

    return results


def search(query: str, top_k: int = None, document_name: str = None) -> list[dict]:
    """Alias for retrieve – kept for compatibility"""
    return retrieve(query, top_k, document_name)