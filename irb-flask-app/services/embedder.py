import numpy as np
from typing import List
from openai import OpenAI
from config import Config

client = OpenAI(api_key=Config.OPENAI_API_KEY)


def embed_texts(texts: List[str], model: str = None) -> np.ndarray:
    """
    Generate embeddings for a list of texts

    Args:
        texts: List of text strings to embed
        model: Embedding model to use (default from config)

    Returns:
        numpy array of shape (len(texts), embedding_dim)
    """
    if not model:
        model = Config.EMBEDDING_MODEL

    if not texts:
        return np.array([])

    embeddings = []
    batch_size = 64

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]

        response = client.embeddings.create(
            model=model,
            input=batch
        )

        batch_embeddings = [item.embedding for item in response.data]
        embeddings.extend(batch_embeddings)

    # Convert to numpy array and normalize
    arr = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0

    return arr / norms


def embed_single(text: str, model: str = None) -> np.ndarray:
    """Embed a single text string"""
    return embed_texts([text], model=model)[0]


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors"""
    return float(np.dot(vec1, vec2))


def find_similar_chunks(query_embedding: np.ndarray,
                        chunk_embeddings: List[np.ndarray],
                        top_k: int = 6) -> List[tuple]:
    """
    Find most similar chunks to query

    Args:
        query_embedding: Query vector
        chunk_embeddings: List of chunk embedding vectors
        top_k: Number of results to return

    Returns:
        List of (index, similarity_score) tuples
    """
    if not chunk_embeddings:
        return []

    # Stack embeddings
    embedding_matrix = np.vstack(chunk_embeddings)

    # Compute similarities
    similarities = embedding_matrix @ query_embedding

    # Get top k indices
    top_indices = np.argsort(-similarities)[:top_k]

    results = [(int(idx), float(similarities[idx])) for idx in top_indices]

    return results