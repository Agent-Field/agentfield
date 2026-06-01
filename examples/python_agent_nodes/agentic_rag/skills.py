"""
Deterministic skills for Agentic RAG
These are non-AI functions for data processing
"""

import hashlib
import re
from typing import List, Dict
import numpy as np
from embedding_manager import get_embedding_manager


def load_document(file_path: str) -> str:
    """Load document from file path"""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def create_chunk_id(text: str, index: int) -> str:
    """Generate unique chunk ID"""
    text_hash = hashlib.md5(text.encode()).hexdigest()[:8]
    return f"chunk_{index}_{text_hash}"


def simple_chunk_text(
    text: str, chunk_size: int = 500, overlap: int = 50
) -> List[Dict]:
    """
    Simple text chunking with overlap
    Returns list of dicts with chunk info
    """
    chunks = []
    start = 0
    index = 0

    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end]

        # Try to break at sentence boundary
        if end < len(text):
            last_period = chunk_text.rfind(".")
            last_newline = chunk_text.rfind("\n")
            break_point = max(last_period, last_newline)

            if break_point > chunk_size * 0.5:  # At least 50% of chunk
                end = start + break_point + 1
                chunk_text = text[start:end]

        chunks.append(
            {
                "id": create_chunk_id(chunk_text, index),
                "text": chunk_text.strip(),
                "metadata": {"start_char": start, "end_char": end, "index": index},
            }
        )

        start = end - overlap
        index += 1

    return chunks


def extract_keywords(text: str, top_n: int = 10) -> List[str]:
    """Extract key terms from text (simple frequency-based)"""
    # Remove common words
    stop_words = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "as",
        "is",
        "was",
        "are",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "should",
        "could",
        "may",
        "might",
        "must",
        "can",
        "this",
        "that",
        "these",
        "those",
        "i",
        "you",
        "he",
        "she",
        "it",
        "we",
        "they",
    }

    # Tokenize and count
    words = re.findall(r"\b[a-z]{3,}\b", text.lower())
    word_freq = {}

    for word in words:
        if word not in stop_words:
            word_freq[word] = word_freq.get(word, 0) + 1

    # Sort by frequency
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [word for word, _ in sorted_words[:top_n]]


def keyword_match_score(query_keywords: List[str], chunk_text: str) -> float:
    """Calculate keyword match score"""
    chunk_lower = chunk_text.lower()
    matches = sum(1 for kw in query_keywords if kw.lower() in chunk_lower)
    return matches / len(query_keywords) if query_keywords else 0.0


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors"""
    import math

    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(a * a for a in vec1))
    magnitude2 = math.sqrt(sum(b * b for b in vec2))

    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0

    return dot_product / (magnitude1 * magnitude2)


def find_quote_in_chunk(claim: str, chunk_text: str, min_length: int = 20) -> str:
    """
    Find best matching quote in chunk for a claim
    Returns the most relevant sentence/phrase
    """
    # Split chunk into sentences
    sentences = re.split(r"[.!?]+", chunk_text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > min_length]

    if not sentences:
        return ""

    # Find sentence with most word overlap with claim
    claim_words = set(claim.lower().split())
    best_sentence = ""
    best_score = 0

    for sentence in sentences:
        sentence_words = set(sentence.lower().split())
        overlap = len(claim_words & sentence_words)

        if overlap > best_score:
            best_score = overlap
            best_sentence = sentence

    return best_sentence


def deduplicate_chunks(
    chunks: List[Dict], similarity_threshold: float = 0.9
) -> List[Dict]:
    """Remove duplicate chunks based on text similarity"""
    unique_chunks = []
    seen_texts = set()

    for chunk in chunks:
        text_normalized = " ".join(chunk["text"].lower().split())

        # Simple deduplication based on normalized text
        if text_normalized not in seen_texts:
            unique_chunks.append(chunk)
            seen_texts.add(text_normalized)

    return unique_chunks


# ============= EMBEDDING FUNCTIONS =============


def embed_text(text: str) -> List[float]:
    """
    Embed a single text using FastEmbed
    Returns embedding as list for JSON serialization
    """
    emb_manager = get_embedding_manager()
    embedding = emb_manager.embed_text(text)
    return embedding.tolist()


def embed_batch(texts: List[str]) -> List[List[float]]:
    """
    Embed multiple texts efficiently in batch
    Returns embeddings as lists for JSON serialization
    """
    emb_manager = get_embedding_manager()
    embeddings = emb_manager.embed_batch(texts)
    return [emb.tolist() for emb in embeddings]


def compute_similarity(emb1: List[float], emb2: List[float]) -> float:
    """
    Compute cosine similarity between two embeddings
    """
    emb_manager = get_embedding_manager()
    vec1 = np.array(emb1)
    vec2 = np.array(emb2)
    return emb_manager.cosine_similarity(vec1, vec2)


def rank_by_similarity(
    query_embedding: List[float],
    chunk_embeddings: List[List[float]],
    chunk_ids: List[str],
    top_k: int = 10,
) -> List[Dict]:
    """
    Rank chunks by similarity to query
    Returns list of {chunk_id, score} dicts
    """
    emb_manager = get_embedding_manager()
    query_vec = np.array(query_embedding)
    doc_vecs = [np.array(emb) for emb in chunk_embeddings]

    scores = emb_manager.batch_cosine_similarity(query_vec, doc_vecs)

    # Combine with chunk IDs and sort
    ranked = [
        {"chunk_id": chunk_id, "score": score}
        for chunk_id, score in zip(chunk_ids, scores)
    ]
    ranked.sort(key=lambda x: x["score"], reverse=True)

    return ranked[:top_k]
