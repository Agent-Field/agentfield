"""Simplified Pydantic schemas for the Documentation Chatbot - optimized for .ai compatibility."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ========================= Ingestion Schemas =========================


class DocumentChunk(BaseModel):
    """A single chunk of documentation tied to file + line metadata."""

    chunk_id: str
    namespace: str = Field(default="documentation", description="Logical corpus")
    relative_path: str = Field(description="Path relative to ingestion root")
    section: Optional[str] = Field(default=None, description="Markdown heading")
    text: str
    start_line: int
    end_line: int


class IngestReport(BaseModel):
    """Summary returned after ingesting a folder of docs."""

    namespace: str
    file_count: int
    chunk_count: int
    skipped_files: List[str] = Field(default_factory=list)


# ========================= Query Planning Schemas (Simple: 2 attributes) =========================


class QueryPlan(BaseModel):
    """Diverse search queries generated from user question."""

    queries: List[str] = Field(
        description="3-5 semantically diverse search queries covering different angles"
    )
    strategy: str = Field(
        description="Query diversity approach: 'broad', 'specific', or 'mixed'"
    )


# ========================= Retrieval Schemas =========================


class RetrievalResult(BaseModel):
    """Single retrieved chunk with minimal metadata."""

    text: str = Field(description="Chunk content")
    source: str = Field(description="File path and line range, e.g., 'file.md:10-20'")
    score: float = Field(description="Similarity score")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata including document_key reference",
    )


class DocumentContext(BaseModel):
    """Full document context aggregated from matching chunks."""

    document_key: str = Field(description="Unique document identifier")
    full_text: str = Field(description="Complete document text")
    relative_path: str = Field(description="File path relative to ingestion root")
    matching_chunks: int = Field(description="Number of chunks that matched queries")
    relevance_score: float = Field(description="Aggregated relevance score")
    matched_sections: List[str] = Field(
        default_factory=list, description="Section headings where chunks matched"
    )


# ========================= Citation Schema (Kept for frontend compatibility) =========================


class Citation(BaseModel):
    """Citation metadata for rendering inline references."""

    key: str = Field(
        description="Single letter key without brackets (e.g., 'A', 'B', 'AA'). Do NOT include square brackets."
    )
    relative_path: str
    start_line: int
    end_line: int
    section: Optional[str] = None
    preview: str
    score: float


# ========================= Answer Schemas =========================


class DocAnswer(BaseModel):
    """Final response from the QA system - maintains frontend contract."""

    answer: str = Field(description="Markdown answer with inline citations like [A][B]")
    citations: List[Citation] = Field(
        default_factory=list,
        description="Leave empty - citations are injected by the system",
    )
    confidence: str = Field(
        description="Answer confidence: 'high', 'partial', or 'insufficient'"
    )
    needs_more: bool = Field(
        default=False,
        description="True if more retrieval needed to fully answer question",
    )
    missing_topics: List[str] = Field(
        default_factory=list,
        description="Specific topics/info needed if needs_more=True",
    )
