"""
ML Services for RAG Evaluation

Provides lightweight CPU-based ML models for cost-efficient evaluation:
- Sentence embeddings (all-MiniLM-L6-v2)
- NLI entailment checking (DeBERTa-MNLI)
- Named Entity Recognition (BERT-NER)
"""

from .embeddings import EmbeddingService
from .nli import NLIService
from .ner import NERService

__all__ = ["EmbeddingService", "NLIService", "NERService"]
