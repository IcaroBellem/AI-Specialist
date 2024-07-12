import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def similarity_search(query, documents, vectorizer, embeddings, k=5):
    query_embedding = vectorizer.transform([query]).toarray()
    similarities = cosine_similarity(query_embedding, embeddings).flatten()
    indices = similarities.argsort()[-k:][::-1]
    return [documents[idx] for idx in indices]
