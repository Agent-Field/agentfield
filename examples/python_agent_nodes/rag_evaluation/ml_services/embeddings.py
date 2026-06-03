"""
Sentence Embedding Service

Uses all-MiniLM-L6-v2 for fast, CPU-friendly semantic similarity.
- 384-dimensional embeddings
- ~20ms per inference
- 22M parameters
"""

from typing import List, Union, Optional
from functools import lru_cache
import numpy as np

# Lazy loading to avoid import-time overhead
_model = None
_tokenizer = None


def _get_model():
    """Lazy load the embedding model."""
    global _model, _tokenizer
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer('all-MiniLM-L6-v2')
        except ImportError:
            raise ImportError(
                "sentence-transformers required. Install with: "
                "pip install sentence-transformers"
            )
    return _model


class EmbeddingService:
    """
    Sentence embedding service for semantic similarity.

    Usage:
        service = EmbeddingService()
        similarity = service.similarity("Hello world", "Hi there")
    """

    def __init__(self):
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = _get_model()
        return self._model

    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for text(s).

        Args:
            texts: Single text or list of texts

        Returns:
            Numpy array of embeddings (N x 384 dimensions)
        """
        if isinstance(texts, str):
            texts = [texts]
        return self.model.encode(texts, normalize_embeddings=True)

    def similarity(self, text1: str, text2: str) -> float:
        """
        Compute cosine similarity between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score between 0 and 1
        """
        emb1 = self.embed(text1)
        emb2 = self.embed(text2)
        return float(np.dot(emb1.flatten(), emb2.flatten()))

    def batch_similarity(self, query: str, candidates: List[str]) -> List[float]:
        """
        Compute similarity between query and multiple candidates.

        Args:
            query: Query text
            candidates: List of candidate texts

        Returns:
            List of similarity scores
        """
        query_emb = self.embed(query)
        candidate_embs = self.embed(candidates)
        scores = np.dot(candidate_embs, query_emb.T).flatten()
        return scores.tolist()

    def find_most_similar(
        self,
        query: str,
        candidates: List[str],
        top_k: int = 3,
        threshold: float = 0.5
    ) -> List[tuple]:
        """
        Find most similar candidates to query.

        Args:
            query: Query text
            candidates: List of candidate texts
            top_k: Number of top results to return
            threshold: Minimum similarity threshold

        Returns:
            List of (index, score, text) tuples sorted by similarity
        """
        scores = self.batch_similarity(query, candidates)

        # Filter by threshold and sort
        results = [
            (i, score, candidates[i])
            for i, score in enumerate(scores)
            if score >= threshold
        ]
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:top_k]


# Singleton instance for reuse
_service_instance: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """Get singleton embedding service instance."""
    global _service_instance
    if _service_instance is None:
        _service_instance = EmbeddingService()
    return _service_instance
