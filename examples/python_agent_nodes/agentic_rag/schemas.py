"""
Simple Pydantic schemas for Agentic RAG
Each schema has 2-4 fields max for simplicity
"""

from pydantic import BaseModel
from typing import List, Optional


# ============= DOCUMENT & CHUNKING =============


class Chunk(BaseModel):
    """Single document chunk"""

    id: str
    text: str
    metadata: dict  # {start_char, end_char, page_num}


class ChunkList(BaseModel):
    """List of chunks from document"""

    chunks: List[Chunk]
    total_count: int


# ============= QUERY ANALYSIS =============


class QueryAnalysis(BaseModel):
    """Analysis of query complexity"""

    complexity_score: float  # 0.0-1.0
    query_type: str  # factual, analytical, comparative
    needs_decomposition: bool


class SubQuestion(BaseModel):
    """Decomposed sub-question"""

    question: str
    priority: int  # 1=highest


# ============= RETRIEVAL =============


class RankedChunk(BaseModel):
    """Chunk with relevance score"""

    chunk_id: str
    score: float
    text: str


class SearchTerms(BaseModel):
    """Structured list of key terms for targeted retrieval"""

    terms: List[str]


class RetrievalResult(BaseModel):
    """Results from retrieval strategy"""

    chunks: List[RankedChunk]
    strategy: str  # semantic, keyword, hybrid


# ============= ANSWER SYNTHESIS =============


class DraftAnswer(BaseModel):
    """Draft answer with confidence"""

    text: str
    confidence: float  # 0.0-1.0
    gaps: List[str]  # Missing information


class Gap(BaseModel):
    """Information gap identified"""

    description: str
    priority: int


class GapList(BaseModel):
    """Wrapper returned when listing gaps"""

    gaps: List[Gap]


# ============= CLAIM VERIFICATION =============


class Claim(BaseModel):
    """Atomic factual claim"""

    id: str
    text: str
    claim_type: str  # fact, inference, opinion


class ClaimList(BaseModel):
    """Wrapper for extracted claims"""

    claims: List[Claim]


class VerificationResult(BaseModel):
    """Verification status for a claim"""

    claim_id: str
    is_verified: bool
    confidence: float
    quote_ids: List[str]


class Citation(BaseModel):
    """Citation with source reference"""

    chunk_id: str
    quote: str
    page_num: Optional[int] = None


# ============= QUERY DECOMPOSITION =============


class SubQuestions(BaseModel):
    """Decomposed sub-questions"""

    questions: List[SubQuestion]


# ============= GAPS & REFINEMENT =============


class RefinementQuery(BaseModel):
    """Query for refinement iteration"""

    query: str
    focus_area: str


class RefinementQueryList(BaseModel):
    """List of refinement queries"""

    queries: List[RefinementQuery]


# ============= FINAL OUTPUT =============


class VerifiedAnswer(BaseModel):
    """Final verified answer with full provenance"""

    answer: str
    citations: List[Citation]
    confidence_score: float
    verification_summary: dict  # {verified: int, uncertain: int, removed: int}
    completeness_score: float
    gaps: List[str]


class CompletenessScore(BaseModel):
    """Completeness assessment wrapper"""

    score: float


# ============= META-COGNITIVE SCHEMAS =============


class QuerySemantics(BaseModel):
    """Query understanding for meta-reasoning"""

    question_type: str  # who_is, what_is, explain, compare, when, why
    subject_type: str  # person, concept, event, organization, place
    critical_terms: List[str]  # Must-match terms
    flexibility: str  # exact_match_required, moderate, broad


class PrecisionRequirements(BaseModel):
    """Match precision needed for this query"""

    precision_level: str  # exact, high, moderate, broad
    reasoning: str  # Why this precision (context for later stages)


class RetrievalPlan(BaseModel):
    """Dynamically composed retrieval strategy"""

    primary_strategy: str  # term_verification, semantic, hybrid
    validation_required: bool
    strategy_reasoning: str


class MatchQuality(BaseModel):
    """Self-aware assessment of chunk-query match"""

    is_relevant: bool
    quality_score: float  # 0.0-1.0
    mismatch_reason: str  # Why NOT relevant (or empty if relevant)


class DraftAnswerWithEntities(BaseModel):
    """Draft answer with extracted entities"""

    answer_text: str
    mentioned_entities: List[str]  # AI extracts entities from answer
    answer_confidence: float


class DriftAnalysis(BaseModel):
    """Semantic drift detection between query and answer"""

    has_entity_substitution: bool
    substitution_details: str  # "Query: X → Answer: Y"
    is_answer_valid: bool


class AlignmentVerdict(BaseModel):
    """Final validation: does answer actually answer query?"""

    is_aligned: bool
    should_return_answer: bool  # or return "no info found"
    verdict_reasoning: str


class FinalConfidence(BaseModel):
    """Adaptive confidence calibration"""

    confidence_score: float
    confidence_reasoning: str
