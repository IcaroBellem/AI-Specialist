import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def similarity_search(question_embedding, document_embeddings, k=5):
    similarities = cosine_similarity(question_embedding.reshape(1, -1), document_embeddings).flatten()
    indices = similarities.argsort()[-k:][::-1]
    return indices
