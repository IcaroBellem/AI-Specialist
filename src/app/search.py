import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def similarity_search(question_embedding, document_embeddings, k=21):
    similarities = cosine_similarity(question_embedding.reshape(1, -1), document_embeddings).flatten()
    top_k_indices = np.argpartition(similarities, -k)[-k:]
    top_k_indices = top_k_indices[np.argsort(similarities[top_k_indices])[::-1]]
    
    return top_k_indices
