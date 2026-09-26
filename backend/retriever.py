from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from backend import storage

def _data():
    return storage.list_chunks()

def clear():
    return None

def add_chunks(chunks):
    return None

def retrieve(query, top_k=5):
    chunks = _data()
    if not chunks:
        return []

    texts = [str(c.get("text", "")) for c in chunks]
    valid = [(i, t) for i, t in enumerate(texts) if t.strip()]
    if not valid:
        return []

    indices, documents = zip(*valid)
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        sublinear_tf=True
    )
    matrix = vectorizer.fit_transform(documents)
    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, matrix).flatten()
    order = scores.argsort()[::-1][:min(top_k, len(documents))]

    results = []
    for position in order:
        item = dict(chunks[indices[position]])
        item["score"] = float(scores[position])
        results.append(item)
    return results

def search(query, top_k=5):
    return retrieve(query, top_k)
