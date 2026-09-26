"""TF-IDF retriever scoped strictly to the selected document."""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from backend import storage
from backend.config import settings

def retrieve(query: str, top_k: int = None, document_name: str = None) -> list[dict]:
    if top_k is None:
        top_k = settings.top_k
    if not document_name:
        return []
    chunks = storage.get_chunks(document_name=document_name)
    if not chunks:
        return []
    valid = [(i, str(c.get("text", "")).strip()) for i, c in enumerate(chunks) if str(c.get("text", "")).strip()]
    if not valid:
        return []
    indices, documents = zip(*valid)
    try:
        vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), sublinear_tf=True, max_features=5000)
        matrix = vectorizer.fit_transform(documents)
        query_vector = vectorizer.transform([query])
        scores = cosine_similarity(query_vector, matrix).flatten()
    except ValueError:
        return []
    results = []
    for position in scores.argsort()[::-1]:
        score = float(scores[position])
        if score < settings.min_similarity:
            continue
        item = dict(chunks[indices[position]])
        item["score"] = score
        results.append(item)
        if len(results) >= max(1, top_k):
            break
    return results

def search(query: str, top_k: int = None, document_name: str = None) -> list[dict]:
    return retrieve(query, top_k, document_name)
