"""Reusable helpers for documentation chatbot reasoners."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Sequence

from agentfield.logger import log_info

from schemas import Citation, DocumentContext, RetrievalResult


def alpha_key(index: int) -> str:
    """Convert index to alphabetic key (0->A, 1->B, ..., 26->AA)."""
    if index < 0:
        raise ValueError("Index must be non-negative")

    letters: List[str] = []
    current = index
    while True:
        current, remainder = divmod(current, 26)
        letters.append(chr(ord("A") + remainder))
        if current == 0:
            break
        current -= 1
    return "".join(reversed(letters))


def filter_hits(
    hits: Sequence[Dict],
    *,
    namespace: str,
    min_score: float,
) -> List[Dict]:
    """Filter vector search hits by namespace and minimum score."""
    filtered: List[Dict] = []
    for hit in hits:
        metadata = hit.get("metadata", {})
        if metadata.get("namespace") != namespace:
            continue
        if hit.get("score", 0.0) < min_score:
            continue
        filtered.append(hit)
    return filtered


def deduplicate_results(results: List[RetrievalResult]) -> List[RetrievalResult]:
    """Deduplicate by source, keeping highest score per unique chunk."""
    by_source: Dict[str, RetrievalResult] = {}

    for result in results:
        if result.source in by_source:
            if by_source[result.source].score >= result.score:
                continue
        by_source[result.source] = result

    return sorted(by_source.values(), key=lambda r: r.score, reverse=True)


def build_citations(results: Sequence[RetrievalResult]) -> List[Citation]:
    """Create citation objects from retrieval results."""
    citations: List[Citation] = []
    for idx, result in enumerate(results):
        key = alpha_key(idx)
        path, lines = (
            result.source.split(":") if ":" in result.source else (result.source, "0-0")
        )
        start_line, end_line = lines.split("-")

        citations.append(
            Citation(
                key=key,
                relative_path=path,
                start_line=int(start_line),
                end_line=int(end_line),
                section=result.metadata.get("section"),
                preview=result.text[:200],
                score=result.score,
            )
        )
    return citations


def format_context_for_synthesis(results: Sequence[RetrievalResult]) -> str:
    """Format retrieval results for the synthesizer prompt."""
    if not results:
        return "(no context available)"

    blocks: List[str] = []
    for idx, result in enumerate(results):
        key = alpha_key(idx)
        blocks.append(
            f"=== CHUNK [{key}] ===\n"
            f"Source: {result.source}\n"
            f"Score: {result.score:.3f}\n"
            f"Text:\n{result.text}\n"
        )

    return "\n".join(blocks)


def calculate_document_score(chunks: Iterable[RetrievalResult]) -> float:
    """Weighted score based on chunk scores and coverage."""
    chunk_list = list(chunks)
    if not chunk_list:
        return 0.0

    average_score = sum(chunk.score for chunk in chunk_list) / len(chunk_list)
    coverage_boost = min(len(chunk_list), 5) * 0.05
    return average_score + coverage_boost


async def aggregate_chunks_to_documents(
    global_memory,
    chunks: List[RetrievalResult],
    top_n: int = 5,
) -> List[DocumentContext]:
    """
    Group chunks by document, fetch full documents, and rank by relevance.
    """
    by_document: Dict[str, List[RetrievalResult]] = defaultdict(list)
    for chunk in chunks:
        doc_key = chunk.metadata.get("document_key")
        if doc_key:
            by_document[doc_key].append(chunk)

    if not by_document:
        log_info("[aggregate_chunks_to_documents] No document keys found in chunks")
        return []

    log_info(
        f"[aggregate_chunks_to_documents] Found {len(by_document)} unique documents"
    )

    document_contexts: List[DocumentContext] = []
    for doc_key, doc_chunks in by_document.items():
        doc_data = await global_memory.get(key=doc_key)
        if not doc_data:
            log_info(f"[aggregate_chunks_to_documents] Document not found: {doc_key}")
            continue

        relevance_score = calculate_document_score(doc_chunks)

        matched_sections = [
            chunk.metadata.get("section")
            for chunk in doc_chunks
            if chunk.metadata.get("section")
        ]
        seen = set()
        unique_sections = []
        for section in matched_sections:
            if section not in seen:
                seen.add(section)
                unique_sections.append(section)

        document_contexts.append(
            DocumentContext(
                document_key=doc_key,
                full_text=doc_data.get("full_text", ""),
                relative_path=doc_data.get("relative_path", "unknown"),
                matching_chunks=len(doc_chunks),
                relevance_score=relevance_score,
                matched_sections=unique_sections,
            )
        )

    ranked_documents = sorted(
        document_contexts, key=lambda x: x.relevance_score, reverse=True
    )[:top_n]

    log_info(
        f"[aggregate_chunks_to_documents] Returning top {len(ranked_documents)} documents "
        f"(scores: {[f'{d.relevance_score:.3f}' for d in ranked_documents]})"
    )

    return ranked_documents


def format_documents_for_synthesis(documents: Sequence[DocumentContext]) -> str:
    """Format full documents with minimal metadata for better AI comprehension."""
    if not documents:
        return "(no documents available)"

    blocks: List[str] = []
    for idx, doc in enumerate(documents):
        key = alpha_key(idx)
        header = f"=== DOCUMENT [{key}]: {doc.relative_path} ==="
        blocks.append(f"{header}\n\n{doc.full_text}\n")

    return "\n".join(blocks)


def build_citations_from_documents(
    documents: Sequence[DocumentContext],
) -> List[Citation]:
    """Convert document contexts to citation objects."""
    citations: List[Citation] = []

    for idx, doc in enumerate(documents):
        key = alpha_key(idx)
        citation = Citation(
            key=key,
            relative_path=doc.relative_path,
            start_line=0,
            end_line=0,
            section=", ".join(doc.matched_sections) if doc.matched_sections else None,
            preview=doc.full_text[:200],
            score=doc.relevance_score,
        )
        citations.append(citation)

    return citations
