"""
RAG Evaluation Python SDK Client

A lightweight client for the RAG Evaluation AgentField node.
Provides both sync and async interfaces for all evaluation metrics.

Usage:
    from rag_eval_client import RAGEvaluator

    evaluator = RAGEvaluator("http://localhost:8080")
    result = evaluator.evaluate(question, context, response)
"""

import httpx
from typing import Optional, Literal, Any
from dataclasses import dataclass, field
from enum import Enum
import random


class EvaluationMode(str, Enum):
    QUICK = "quick"
    STANDARD = "standard"
    THOROUGH = "thorough"


class Domain(str, Enum):
    GENERAL = "general"
    MEDICAL = "medical"
    LEGAL = "legal"
    FINANCIAL = "financial"


@dataclass
class FaithfulnessResult:
    """Result from adversarial debate faithfulness evaluation"""
    score: float
    unfaithful_claims: list[str]
    debate_summary: str
    reasoning: str


@dataclass
class RelevanceResult:
    """Result from multi-jury relevance evaluation"""
    overall_score: float
    literal_score: float
    intent_score: float
    scope_score: float
    disagreement_level: float
    verdict: str


@dataclass
class HallucinationResult:
    """Result from hybrid ML+LLM hallucination detection"""
    score: float
    fabrications: list[str]
    contradictions: list[str]
    ml_handled_percent: float
    total_statements: int


@dataclass
class ConstitutionalResult:
    """Result from principles-based constitutional evaluation"""
    overall_score: float
    compliance_status: str
    critical_violations: list[str]
    principle_scores: dict[str, float]
    improvement_needed: list[str] = field(default_factory=list)


@dataclass
class RAGEvaluationResult:
    """Complete RAG evaluation result with all metrics"""
    overall_score: float
    quality_tier: str
    evaluation_mode: str
    ai_calls_made: int
    requires_human_review: bool
    critical_issues: list[str]
    recommendations: list[str]
    faithfulness: FaithfulnessResult
    relevance: RelevanceResult
    hallucination: HallucinationResult
    constitutional: ConstitutionalResult
    execution_id: str
    duration_ms: int


class RAGEvaluator:
    """
    Python SDK client for RAG Evaluation AgentField node.

    Provides convenient methods for evaluating RAG responses using
    multi-reasoner architectures including adversarial debate,
    multi-jury consensus, and hybrid ML+LLM verification.

    Example:
        >>> evaluator = RAGEvaluator("http://localhost:8080")
        >>> result = evaluator.evaluate(
        ...     question="What is photosynthesis?",
        ...     context="Photosynthesis is the process...",
        ...     response="Photosynthesis converts sunlight..."
        ... )
        >>> print(f"Score: {result.overall_score}")
    """

    def __init__(
        self,
        agentfield_server: str = "http://localhost:8080",
        timeout: float = 60.0,
        agent_id: str = "rag-evaluation"
    ):
        """
        Initialize the RAG Evaluator client.

        Args:
            agentfield_server: URL of the AgentField control plane
            timeout: Request timeout in seconds
            agent_id: ID of the RAG evaluation agent node
        """
        self.base_url = agentfield_server.rstrip("/")
        self.timeout = timeout
        self.agent_id = agent_id
        self._client = httpx.Client(timeout=timeout)

    def _execute(self, reasoner: str, input_data: dict) -> dict:
        """Execute a reasoner via the control plane"""
        url = f"{self.base_url}/api/v1/execute/{self.agent_id}.{reasoner}"
        response = self._client.post(url, json={"input": input_data})
        response.raise_for_status()
        return response.json()

    # ==================== Full Evaluation ====================

    def evaluate(
        self,
        question: str,
        context: str,
        response: str,
        mode: EvaluationMode = EvaluationMode.STANDARD,
        domain: Domain = Domain.GENERAL
    ) -> RAGEvaluationResult:
        """
        Perform full RAG evaluation with all 4 metrics.

        This runs:
        - Faithfulness (adversarial debate)
        - Relevance (multi-jury consensus)
        - Hallucination (hybrid ML+LLM)
        - Constitutional (principles-based)

        Args:
            question: The user's original question
            context: Retrieved context/documents
            response: RAG-generated response to evaluate
            mode: Evaluation depth (quick/standard/thorough)
            domain: Domain for constitutional principles

        Returns:
            RAGEvaluationResult with all metrics and recommendations
        """
        result = self._execute("evaluate_rag_response", {
            "question": question,
            "context": context,
            "response": response,
            "mode": mode.value,
            "domain": domain.value
        })

        return self._parse_full_result(result)

    def evaluate_quick(
        self,
        question: str,
        context: str,
        response: str,
        domain: Domain = Domain.GENERAL
    ) -> RAGEvaluationResult:
        """Quick evaluation (~4 AI calls) for real-time validation"""
        return self.evaluate(question, context, response, EvaluationMode.QUICK, domain)

    def evaluate_standard(
        self,
        question: str,
        context: str,
        response: str,
        domain: Domain = Domain.GENERAL
    ) -> RAGEvaluationResult:
        """Standard evaluation (~14 AI calls) for production use"""
        return self.evaluate(question, context, response, EvaluationMode.STANDARD, domain)

    def evaluate_thorough(
        self,
        question: str,
        context: str,
        response: str,
        domain: Domain = Domain.GENERAL
    ) -> RAGEvaluationResult:
        """Thorough evaluation (~20+ AI calls) for audits/compliance"""
        return self.evaluate(question, context, response, EvaluationMode.THOROUGH, domain)

    # ==================== Individual Metrics ====================

    def faithfulness(
        self,
        response: str,
        context: str,
        mode: Literal["quick", "full"] = "full"
    ) -> FaithfulnessResult:
        """
        Evaluate faithfulness using adversarial debate.

        A Prosecutor attacks claims, a Defender defends them,
        and a Judge synthesizes the final verdict.

        Args:
            response: RAG-generated response
            context: Source context/documents
            mode: "quick" (single reasoner) or "full" (adversarial debate)
        """
        result = self._execute("evaluate_faithfulness_only", {
            "response": response,
            "context": context,
            "mode": mode
        })
        return self._parse_faithfulness(result["result"])

    def relevance(
        self,
        question: str,
        response: str,
        mode: Literal["quick", "full"] = "full"
    ) -> RelevanceResult:
        """
        Evaluate relevance using multi-jury consensus.

        Three jurors vote on different aspects:
        - Literal: Does response literally answer the question?
        - Intent: Does it address the underlying user need?
        - Scope: Is the response appropriately scoped?

        Args:
            question: Original question
            response: RAG-generated response
            mode: "quick" (single reasoner) or "full" (multi-jury)
        """
        result = self._execute("evaluate_relevance_only", {
            "question": question,
            "response": response,
            "mode": mode
        })
        return self._parse_relevance(result["result"])

    def hallucination(
        self,
        response: str,
        context: str,
        mode: Literal["quick", "full"] = "full"
    ) -> HallucinationResult:
        """
        Detect hallucinations using hybrid ML+LLM approach.

        ML models (embeddings, NLI) filter obvious cases,
        LLM escalation handles uncertain statements.

        Args:
            response: RAG-generated response
            context: Source context/documents
            mode: "quick" (single reasoner) or "full" (hybrid ML+LLM)
        """
        result = self._execute("evaluate_hallucination_only", {
            "response": response,
            "context": context,
            "mode": mode
        })
        return self._parse_hallucination(result["result"])

    def constitutional(
        self,
        question: str,
        response: str,
        context: str,
        domain: Domain = Domain.GENERAL,
        mode: Literal["quick", "full"] = "full"
    ) -> ConstitutionalResult:
        """
        Evaluate constitutional compliance using principles.

        Checks against configurable principles:
        - No fabrication
        - Accurate attribution
        - Completeness
        - Safety
        - Uncertainty expression

        Args:
            question: Original question
            response: RAG-generated response
            context: Source context
            domain: Domain preset (general/medical/legal/financial)
            mode: "quick" (single reasoner) or "full" (parallel principle checks)
        """
        result = self._execute("evaluate_constitutional_only", {
            "question": question,
            "response": response,
            "context": context,
            "domain": domain.value,
            "mode": mode
        })
        return self._parse_constitutional(result["result"])

    # ==================== Sampling & Batch Evaluation ====================

    def sample_and_evaluate(
        self,
        rag_logs: list[dict],
        sample_size: int = 100,
        sample_rate: float = None,
        mode: EvaluationMode = EvaluationMode.QUICK,
        domain: Domain = Domain.GENERAL
    ) -> dict:
        """
        Sample RAG logs and evaluate for quality metrics.

        Useful for building evaluation pipelines that randomly
        sample production traffic for quality monitoring.

        Args:
            rag_logs: List of dicts with 'question', 'context', 'response'
            sample_size: Number of samples to evaluate (if sample_rate not set)
            sample_rate: Fraction of logs to sample (0.0-1.0)
            mode: Evaluation depth
            domain: Domain preset

        Returns:
            Dict with aggregate statistics and individual results
        """
        # Sample the logs
        if sample_rate is not None:
            sample_size = max(1, int(len(rag_logs) * sample_rate))

        samples = random.sample(rag_logs, min(sample_size, len(rag_logs)))

        # Evaluate each sample
        results = []
        for log in samples:
            try:
                result = self.evaluate(
                    question=log["question"],
                    context=log["context"],
                    response=log["response"],
                    mode=mode,
                    domain=domain
                )
                results.append({
                    "input": log,
                    "result": result,
                    "success": True
                })
            except Exception as e:
                results.append({
                    "input": log,
                    "error": str(e),
                    "success": False
                })

        # Compute aggregate statistics
        successful = [r for r in results if r["success"]]

        if successful:
            stats = {
                "sample_size": len(samples),
                "successful_evaluations": len(successful),
                "failed_evaluations": len(results) - len(successful),
                "avg_overall_score": sum(r["result"].overall_score for r in successful) / len(successful),
                "avg_faithfulness": sum(r["result"].faithfulness.score for r in successful) / len(successful),
                "avg_relevance": sum(r["result"].relevance.overall_score for r in successful) / len(successful),
                "avg_hallucination": sum(r["result"].hallucination.score for r in successful) / len(successful),
                "avg_constitutional": sum(r["result"].constitutional.overall_score for r in successful) / len(successful),
                "quality_tier_distribution": self._count_tiers(successful),
                "critical_issues_count": sum(len(r["result"].critical_issues) for r in successful),
                "requires_human_review_count": sum(1 for r in successful if r["result"].requires_human_review),
            }
        else:
            stats = {
                "sample_size": len(samples),
                "successful_evaluations": 0,
                "failed_evaluations": len(results),
            }

        return {
            "statistics": stats,
            "results": results
        }

    def _count_tiers(self, results: list) -> dict:
        """Count quality tier distribution"""
        tiers = {}
        for r in results:
            tier = r["result"].quality_tier
            tiers[tier] = tiers.get(tier, 0) + 1
        return tiers

    # ==================== Result Parsing ====================

    def _parse_full_result(self, api_result: dict) -> RAGEvaluationResult:
        """Parse API response into typed result"""
        r = api_result["result"]
        return RAGEvaluationResult(
            overall_score=r.get("overall_score", 0),
            quality_tier=r.get("quality_tier", "unknown"),
            evaluation_mode=r.get("evaluation_mode", "unknown"),
            ai_calls_made=r.get("ai_calls_made", 0),
            requires_human_review=r.get("requires_human_review", False),
            critical_issues=r.get("critical_issues", []),
            recommendations=r.get("recommendations", []),
            faithfulness=self._parse_faithfulness(r.get("faithfulness", {})),
            relevance=self._parse_relevance(r.get("relevance", {})),
            hallucination=self._parse_hallucination(r.get("hallucination", {})),
            constitutional=self._parse_constitutional(r.get("constitutional", {})),
            execution_id=api_result.get("execution_id", ""),
            duration_ms=api_result.get("duration_ms", 0)
        )

    def _parse_faithfulness(self, data: dict) -> FaithfulnessResult:
        return FaithfulnessResult(
            score=data.get("score", 0),
            unfaithful_claims=data.get("unfaithful_claims", []),
            debate_summary=data.get("debate_summary", ""),
            reasoning=data.get("reasoning", "")
        )

    def _parse_relevance(self, data: dict) -> RelevanceResult:
        return RelevanceResult(
            overall_score=data.get("overall_score", 0),
            literal_score=data.get("literal_score", 0),
            intent_score=data.get("intent_score", 0),
            scope_score=data.get("scope_score", 0),
            disagreement_level=data.get("disagreement_level", 0),
            verdict=data.get("verdict", "")
        )

    def _parse_hallucination(self, data: dict) -> HallucinationResult:
        return HallucinationResult(
            score=data.get("score", 0),
            fabrications=data.get("fabrications", []),
            contradictions=data.get("contradictions", []),
            ml_handled_percent=data.get("ml_handled_percent", 0),
            total_statements=data.get("total_statements", 0)
        )

    def _parse_constitutional(self, data: dict) -> ConstitutionalResult:
        return ConstitutionalResult(
            overall_score=data.get("overall_score", 0),
            compliance_status=data.get("compliance_status", "unknown"),
            critical_violations=data.get("critical_violations", []),
            principle_scores=data.get("principle_scores", {}),
            improvement_needed=data.get("improvement_needed", [])
        )

    def close(self):
        """Close the HTTP client"""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ==================== Async Client ====================

class AsyncRAGEvaluator:
    """
    Async Python SDK client for RAG Evaluation.

    Same interface as RAGEvaluator but with async methods
    for use in async applications.

    Example:
        >>> async with AsyncRAGEvaluator("http://localhost:8080") as evaluator:
        ...     result = await evaluator.evaluate(question, context, response)
    """

    def __init__(
        self,
        agentfield_server: str = "http://localhost:8080",
        timeout: float = 60.0,
        agent_id: str = "rag-evaluation"
    ):
        self.base_url = agentfield_server.rstrip("/")
        self.timeout = timeout
        self.agent_id = agent_id
        self._client = httpx.AsyncClient(timeout=timeout)

    async def _execute(self, reasoner: str, input_data: dict) -> dict:
        """Execute a reasoner via the control plane"""
        url = f"{self.base_url}/api/v1/execute/{self.agent_id}.{reasoner}"
        response = await self._client.post(url, json={"input": input_data})
        response.raise_for_status()
        return response.json()

    async def evaluate(
        self,
        question: str,
        context: str,
        response: str,
        mode: EvaluationMode = EvaluationMode.STANDARD,
        domain: Domain = Domain.GENERAL
    ) -> RAGEvaluationResult:
        """Async full RAG evaluation with all 4 metrics"""
        result = await self._execute("evaluate_rag_response", {
            "question": question,
            "context": context,
            "response": response,
            "mode": mode.value,
            "domain": domain.value
        })
        return RAGEvaluator._parse_full_result(None, result)

    async def faithfulness(
        self,
        response: str,
        context: str,
        mode: Literal["quick", "full"] = "full"
    ) -> FaithfulnessResult:
        """Async faithfulness evaluation"""
        result = await self._execute("evaluate_faithfulness_only", {
            "response": response,
            "context": context,
            "mode": mode
        })
        return RAGEvaluator._parse_faithfulness(None, result["result"])

    async def relevance(
        self,
        question: str,
        response: str,
        mode: Literal["quick", "full"] = "full"
    ) -> RelevanceResult:
        """Async relevance evaluation"""
        result = await self._execute("evaluate_relevance_only", {
            "question": question,
            "response": response,
            "mode": mode
        })
        return RAGEvaluator._parse_relevance(None, result["result"])

    async def hallucination(
        self,
        response: str,
        context: str,
        mode: Literal["quick", "full"] = "full"
    ) -> HallucinationResult:
        """Async hallucination detection"""
        result = await self._execute("evaluate_hallucination_only", {
            "response": response,
            "context": context,
            "mode": mode
        })
        return RAGEvaluator._parse_hallucination(None, result["result"])

    async def constitutional(
        self,
        question: str,
        response: str,
        context: str,
        domain: Domain = Domain.GENERAL,
        mode: Literal["quick", "full"] = "full"
    ) -> ConstitutionalResult:
        """Async constitutional evaluation"""
        result = await self._execute("evaluate_constitutional_only", {
            "question": question,
            "response": response,
            "context": context,
            "domain": domain.value,
            "mode": mode
        })
        return RAGEvaluator._parse_constitutional(None, result["result"])

    async def close(self):
        """Close the async HTTP client"""
        await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()


# ==================== Convenience Functions ====================

def evaluate_rag(
    question: str,
    context: str,
    response: str,
    server: str = "http://localhost:8080",
    mode: str = "standard",
    domain: str = "general"
) -> RAGEvaluationResult:
    """
    One-liner RAG evaluation.

    Example:
        >>> from rag_eval_client import evaluate_rag
        >>> result = evaluate_rag("What is AI?", context, response)
        >>> print(result.overall_score)
    """
    with RAGEvaluator(server) as evaluator:
        return evaluator.evaluate(
            question, context, response,
            EvaluationMode(mode),
            Domain(domain)
        )


if __name__ == "__main__":
    # Quick test
    evaluator = RAGEvaluator("http://localhost:8080")

    result = evaluator.evaluate_quick(
        question="What is the capital of France?",
        context="France is a country in Europe. Paris is the capital city of France.",
        response="The capital of France is Paris."
    )

    print(f"Overall Score: {result.overall_score}")
    print(f"Quality Tier: {result.quality_tier}")
    print(f"Faithfulness: {result.faithfulness.score}")
    print(f"Relevance: {result.relevance.overall_score}")

    evaluator.close()
