"""
Data models for RAG Evaluation Agent Node

Minimal schemas (3-4 fields) optimized for smaller, cheaper LLMs.
Each schema represents a focused output from a specialist reasoner.
"""

from pydantic import BaseModel, Field
from typing import Literal, Optional, List, Dict, Any


# ============================================
# INPUT MODELS
# ============================================

class EvaluationInput(BaseModel):
    """Input for RAG evaluation."""
    question: str = Field(description="The user's question")
    context: str = Field(description="Retrieved context from RAG system")
    response: str = Field(description="RAG-generated response to evaluate")
    sources: Optional[List[str]] = Field(None, description="Optional source documents")
    model: Optional[str] = Field(None, description="Model to use for evaluation (e.g., 'openrouter/deepseek/deepseek-chat-v3-0324')")


# ============================================
# FAITHFULNESS MODELS (Adversarial Debate)
# ============================================

class AtomicClaim(BaseModel):
    """A single factual claim extracted from response."""
    claim_text: str = Field(description="The factual assertion")
    claim_type: Literal["factual", "inferential", "opinion"]
    importance: Literal["critical", "supporting", "minor"]


class ClaimExtraction(BaseModel):
    """Output from claim extractor."""
    claims: List[AtomicClaim]
    total_count: int = Field(description="Number of claims extracted")


class ProsecutorAttack(BaseModel):
    """Prosecutor's attack on a claim."""
    claim_index: int = Field(description="Which claim is being attacked")
    attack_type: Literal["unsupported", "contradicted", "exaggerated", "out_of_context"]
    evidence: str = Field(description="Why this claim is unfaithful")
    severity: Literal["critical", "major", "minor"]


class ProsecutorAnalysis(BaseModel):
    """Full prosecutor analysis output."""
    attacks: List[ProsecutorAttack]
    prosecution_summary: str = Field(description="Overall prosecution case")


class DefenderDefense(BaseModel):
    """Defender's defense of a claim."""
    claim_index: int = Field(description="Which claim is being defended")
    defense_type: Literal["direct_support", "implicit_support", "reasonable_inference", "acknowledged_issue"]
    evidence: str = Field(description="Evidence supporting the claim")
    strength: float = Field(ge=0.0, le=1.0, description="Strength of defense")


class DefenderAnalysis(BaseModel):
    """Full defender analysis output."""
    defenses: List[DefenderDefense]
    defense_summary: str = Field(description="Overall defense case")


class FaithfulnessVerdict(BaseModel):
    """Final faithfulness judgment from judge."""
    score: float = Field(ge=0.0, le=1.0, description="Faithfulness score")
    unfaithful_claims: List[str] = Field(description="Claims ruled unfaithful")
    debate_summary: str = Field(description="Key debate points")
    reasoning: str = Field(description="Judicial reasoning")


# ============================================
# RELEVANCE MODELS (Multi-Jury Consensus)
# ============================================

class QuestionIntent(BaseModel):
    """Question analysis output."""
    primary_intent: str = Field(description="What user really wants to know")
    sub_questions: List[str] = Field(description="Implicit sub-questions")
    expected_type: Literal["factual", "explanation", "list", "comparison", "procedure", "opinion"]


class JurorVote(BaseModel):
    """Single juror's vote."""
    dimension: Literal["literal", "intent", "scope"]
    score: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(description="Why this score")
    confidence: float = Field(ge=0.0, le=1.0)


class RelevanceVerdict(BaseModel):
    """Final relevance verdict from foreman."""
    overall_score: float = Field(ge=0.0, le=1.0)
    literal_score: float = Field(ge=0.0, le=1.0)
    intent_score: float = Field(ge=0.0, le=1.0)
    scope_score: float = Field(ge=0.0, le=1.0)
    disagreement_level: float = Field(ge=0.0, le=1.0, description="How much jurors disagreed")
    verdict: str = Field(description="Summary verdict")


# ============================================
# HALLUCINATION MODELS (Hybrid ML+LLM)
# ============================================

class StatementExtraction(BaseModel):
    """Extracted factual statements for verification."""
    statements: List[str] = Field(description="Factual statements to verify")
    entity_count: int = Field(description="Number of named entities found")


class MLVerificationResult(BaseModel):
    """ML-based verification result for a statement."""
    statement_index: int
    verification_status: Literal["verified", "uncertain", "unverified"]
    confidence: float = Field(ge=0.0, le=1.0)
    method: Literal["embedding_similarity", "nli_entailment", "exact_match"]


class LLMVerificationResult(BaseModel):
    """LLM verification for uncertain statements."""
    statement_index: int
    is_hallucination: bool
    explanation: str = Field(description="Why this is/isn't a hallucination")
    confidence: float = Field(ge=0.0, le=1.0)


class HallucinationReport(BaseModel):
    """Final hallucination detection report."""
    score: float = Field(ge=0.0, le=1.0, description="Groundedness score (1=fully grounded)")
    fabrications: List[str] = Field(description="Fabricated statements")
    contradictions: List[str] = Field(description="Statements contradicting context")
    ml_handled_percent: float = Field(ge=0.0, le=100.0, description="% handled by ML")
    total_statements: int


# ============================================
# CONSTITUTIONAL MODELS (Principles-Based)
# ============================================

class PrincipleViolation(BaseModel):
    """A single principle violation."""
    principle_id: str
    principle_name: str
    violation: str = Field(description="What violated the principle")
    severity: Literal["critical", "major", "minor"]


class PrincipleCheck(BaseModel):
    """Result of checking one principle."""
    principle_id: str
    score: float = Field(ge=0.0, le=1.0)
    passed: bool
    violations: List[PrincipleViolation]
    reasoning: str


class ConstitutionalReport(BaseModel):
    """Full constitutional compliance report."""
    overall_score: float = Field(ge=0.0, le=1.0)
    compliance_status: Literal["compliant", "minor_issues", "major_issues", "non_compliant"]
    critical_violations: List[PrincipleViolation]
    principle_scores: Dict[str, float]
    improvement_needed: List[str]


# ============================================
# UNIFIED OUTPUT MODELS
# ============================================

class RAGEvaluationResult(BaseModel):
    """Complete RAG evaluation result."""
    # Core metric scores
    faithfulness: FaithfulnessVerdict
    relevance: RelevanceVerdict
    hallucination: HallucinationReport
    constitutional: ConstitutionalReport

    # Aggregate assessment
    overall_score: float = Field(ge=0.0, le=1.0)
    quality_tier: Literal["excellent", "good", "acceptable", "poor", "critical"]

    # Metadata
    evaluation_mode: Literal["quick", "standard", "thorough"]
    ai_calls_made: int
    requires_human_review: bool = Field(default=False)

    # Actionable insights
    critical_issues: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)


# ============================================
# SIMPLIFIED SINGLE-REASONER MODELS
# (For quick mode evaluation)
# ============================================

class QuickFaithfulness(BaseModel):
    """Simplified faithfulness for quick mode."""
    score: float = Field(ge=0.0, le=1.0)
    issues: List[str]
    reasoning: str


class QuickRelevance(BaseModel):
    """Simplified relevance for quick mode."""
    score: float = Field(ge=0.0, le=1.0)
    addresses_question: bool
    reasoning: str


class QuickHallucination(BaseModel):
    """Simplified hallucination for quick mode."""
    score: float = Field(ge=0.0, le=1.0)
    fabrications_found: List[str]
    reasoning: str


class QuickConstitutional(BaseModel):
    """Simplified constitutional for quick mode."""
    score: float = Field(ge=0.0, le=1.0)
    violations: List[str]
    reasoning: str


# ============================================
# CONFIGURATION MODELS
# ============================================

class EvaluationConfig(BaseModel):
    """Configuration for evaluation run."""
    mode: Literal["quick", "standard", "thorough"] = Field(default="standard")
    enable_ml: bool = Field(default=True, description="Use ML models for hallucination detection")
    constitution_file: Optional[str] = Field(None, description="Path to custom constitution")
    domain: Optional[Literal["general", "medical", "legal", "financial"]] = Field(default="general")
    faithfulness_debate_mode: Literal["full", "simplified"] = Field(default="full")
    model: Optional[str] = Field(None, description="Model to use for all AI calls (e.g., 'openrouter/deepseek/deepseek-chat-v3-0324')")
