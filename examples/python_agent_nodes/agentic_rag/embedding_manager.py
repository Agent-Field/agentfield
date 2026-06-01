"""
Embedding Manager - FastEmbed wrapper for lazy, efficient embeddings
Handles embedding generation, caching, and similarity computation
"""

from typing import List, Optional
import numpy as np
from fastembed import TextEmbedding


class EmbeddingManager:
    """
    Manages embeddings with lazy loading and caching
    Uses FastEmbed for fast, lightweight embeddings
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        """
        Initialize embedding manager with lazy model loading

        Args:
            model_name: FastEmbed model to use (default: bge-small-en-v1.5)
        """
        self.model_name = model_name
        self._model: Optional[TextEmbedding] = None

    @property
    def model(self) -> TextEmbedding:
        """Lazy load the embedding model"""
        if self._model is None:
            self._model = TextEmbedding(model_name=self.model_name)
        return self._model

    def embed_text(self, text: str) -> np.ndarray:
        """
        Embed a single text string

        Args:
            text: Text to embed

        Returns:
            Embedding vector as numpy array
        """
        embeddings = list(self.model.embed([text]))
        return np.array(embeddings[0])

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        Embed multiple texts efficiently in batch

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        embeddings = list(self.model.embed(texts))
        return [np.array(emb) for emb in embeddings]

    @staticmethod
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors

        Args:
            vec1: First embedding vector
            vec2: Second embedding vector

        Returns:
            Cosine similarity score (0.0 to 1.0)
        """
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    @staticmethod
    def batch_cosine_similarity(
        query_vec: np.ndarray, doc_vecs: List[np.ndarray]
    ) -> List[float]:
        """
        Compute cosine similarity between query and multiple documents

        Args:
            query_vec: Query embedding vector
            doc_vecs: List of document embedding vectors

        Returns:
            List of similarity scores
        """
        scores = []
        for doc_vec in doc_vecs:
            score = EmbeddingManager.cosine_similarity(query_vec, doc_vec)
            scores.append(score)
        return scores


# Global singleton instance
_embedding_manager: Optional[EmbeddingManager] = None


def get_embedding_manager() -> EmbeddingManager:
    """Get or create the global embedding manager instance"""
    global _embedding_manager
    if _embedding_manager is None:
        _embedding_manager = EmbeddingManager()
    return _embedding_manager
