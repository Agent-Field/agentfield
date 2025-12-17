"""
Master Orchestrator for RAG Evaluation

Provides adaptive depth routing for evaluation:
- Quick mode: 4-6 AI calls (single reasoner per metric)
- Standard mode: 10-14 AI calls (multi-reasoner patterns)
- Thorough mode: 18+ AI calls (full debate + jury + all principles)

Creates impressive master workflow graph showing all metric evaluations.
"""

import asyncio
from typing import Dict, Any, Optional, Literal, Union
from models import (
    RAGEvaluationResult,
    EvaluationConfig,
    FaithfulnessVerdict,
    RelevanceVerdict,
    HallucinationReport,
    ConstitutionalReport,
)


def register_orchestrator_reasoners(router):
    """Register orchestrator reasoners with the router."""

    # ============================================
    # MAIN EVALUATION ENTRY POINT
    # ============================================

    @router.reasoner()
    async def evaluate_rag_response(
        question: str,
        context: str,
        response: str,
        mode: Literal["quick", "standard", "thorough"] = "standard",
        domain: str = "general",
        faithfulness_debate_mode: Literal["full", "simplified"] = "full",
        model: Optional[str] = None
    ) -> dict:
        """
        Master RAG evaluation orchestrator.

        Routes to appropriate evaluation depth based on mode:
        - quick: Fast single-reasoner evaluation (4-6 AI calls)
        - standard: Multi-reasoner patterns (10-14 AI calls)
        - thorough: Full adversarial + jury + all principles (18+ AI calls)

        Creates master workflow graph:
        evaluate_rag -> [faithfulness, relevance, hallucination, constitutional] -> aggregate

        Each metric branch shows its internal architecture:
        - Faithfulness: debate pattern (prosecutor/defender/judge)
        - Relevance: jury pattern (literal/intent/scope/foreman)
        - Hallucination: hybrid pattern (ml_verify/llm_escalate/synthesize)
        - Constitutional: principles pattern (check_P1/P2/.../aggregate)
        """

        router.note(f"Starting {mode} evaluation for domain: {domain}",
                   tags=["orchestration", "start", mode])

        # Store evaluation context in workflow memory
        await router.memory.set("evaluation_input", {
            "question": question,
            "context_length": len(context),
            "response_length": len(response),
            "mode": mode,
            "domain": domain
        })

        if mode == "quick":
            result = await evaluate_quick(
                question=question,
                context=context,
                response=response,
                domain=domain,
                model=model
            )
        elif mode == "thorough":
            result = await evaluate_thorough(
                question=question,
                context=context,
                response=response,
                domain=domain,
                faithfulness_debate_mode=faithfulness_debate_mode,
                model=model
            )
        else:  # standard
            result = await evaluate_standard(
                question=question,
                context=context,
                response=response,
                domain=domain,
                faithfulness_debate_mode=faithfulness_debate_mode,
                model=model
            )

        router.note(f"Evaluation complete: {result['quality_tier']} (score: {result['overall_score']:.2f})",
                   tags=["orchestration", "complete"])

        return result

    # ============================================
    # QUICK MODE (Minimal, fast)
    # ============================================

    @router.reasoner()
    async def evaluate_quick(
        question: str,
        context: str,
        response: str,
        domain: str = "general",
        model: Optional[str] = None
    ) -> dict:
        """
        Quick evaluation mode: Single reasoner per metric.

        Workflow: 4 parallel single-reasoner evaluations
        AI calls: 4 total
        Latency: ~1-2s

        Good for: High-volume validation, real-time checks
        """

        router.note("Quick mode: 4 parallel single-reasoner evaluations",
                   tags=["orchestration", "quick", "parallel"])

        # Launch all 4 evaluations in parallel
        faithfulness_task = router.app.call(
            "rag-evaluation.evaluate_faithfulness_quick",
            response=response,
            context=context,
            model=model
        )

        relevance_task = router.app.call(
            "rag-evaluation.evaluate_relevance_quick",
            question=question,
            response=response,
            model=model
        )

        hallucination_task = router.app.call(
            "rag-evaluation.evaluate_hallucination_quick",
            response=response,
            context=context,
            model=model
        )

        constitutional_task = router.app.call(
            "rag-evaluation.evaluate_constitutional_quick",
            question=question,
            response=response,
            context=context,
            model=model
        )

        # Wait for all
        faithfulness, relevance, hallucination, constitutional = await asyncio.gather(
            faithfulness_task, relevance_task, hallucination_task, constitutional_task
        )

        router.note("Quick evaluations complete", tags=["orchestration", "quick"])

        # Aggregate
        return aggregate_results(
            faithfulness=faithfulness,
            relevance=relevance,
            hallucination=hallucination,
            constitutional=constitutional,
            mode="quick",
            ai_calls=4
        )

    # ============================================
    # STANDARD MODE (Balanced)
    # ============================================

    @router.reasoner()
    async def evaluate_standard(
        question: str,
        context: str,
        response: str,
        domain: str = "general",
        faithfulness_debate_mode: str = "full",
        model: Optional[str] = None
    ) -> dict:
        """
        Standard evaluation mode: Multi-reasoner patterns.

        Workflow: 4 parallel metric evaluations, each with internal multi-reasoner
        AI calls: 10-14 total
        Latency: ~3-4s

        Good for: Production evaluation, quality assurance
        """

        router.note("Standard mode: Multi-reasoner patterns",
                   tags=["orchestration", "standard", "parallel"])

        # Choose faithfulness depth
        if faithfulness_debate_mode == "full":
            faithfulness_task = router.app.call(
                "rag-evaluation.evaluate_faithfulness_full",
                response=response,
                context=context,
                model=model
            )
        else:
            faithfulness_task = router.app.call(
                "rag-evaluation.evaluate_faithfulness_quick",
                response=response,
                context=context,
                model=model
            )

        # Multi-jury relevance
        relevance_task = router.app.call(
            "rag-evaluation.evaluate_relevance_full",
            question=question,
            response=response,
            model=model
        )

        # Hybrid hallucination
        hallucination_task = router.app.call(
            "rag-evaluation.evaluate_hallucination_full",
            response=response,
            context=context,
            model=model
        )

        # Constitutional with parallel principle checks
        constitutional_task = router.app.call(
            "rag-evaluation.evaluate_constitutional_full",
            question=question,
            response=response,
            context=context,
            domain=domain,
            model=model
        )

        # Wait for all
        faithfulness, relevance, hallucination, constitutional = await asyncio.gather(
            faithfulness_task, relevance_task, hallucination_task, constitutional_task
        )

        router.note("Standard evaluations complete", tags=["orchestration", "standard"])

        # Estimate AI calls
        ai_calls = 4 if faithfulness_debate_mode != "full" else 4 + 3  # Base + debate
        ai_calls += 4  # Relevance jury
        ai_calls += 2 + hallucination.get("uncertain_count", 2)  # Hybrid
        ai_calls += 6  # Constitutional principles

        return aggregate_results(
            faithfulness=faithfulness,
            relevance=relevance,
            hallucination=hallucination,
            constitutional=constitutional,
            mode="standard",
            ai_calls=ai_calls
        )

    # ============================================
    # THOROUGH MODE (Maximum depth)
    # ============================================

    @router.reasoner()
    async def evaluate_thorough(
        question: str,
        context: str,
        response: str,
        domain: str = "general",
        faithfulness_debate_mode: str = "full",
        model: Optional[str] = None
    ) -> dict:
        """
        Thorough evaluation mode: Maximum depth on all metrics.

        Workflow: Full adversarial + full jury + full hybrid + all principles
        AI calls: 18+ total
        Latency: ~6-8s

        Good for: High-stakes evaluation, audits, compliance checks
        """

        router.note("Thorough mode: Maximum depth evaluation",
                   tags=["orchestration", "thorough", "parallel"])

        # All metrics at full depth, in parallel
        faithfulness_task = router.app.call(
            "rag-evaluation.evaluate_faithfulness_full",
            response=response,
            context=context,
            model=model
        )

        relevance_task = router.app.call(
            "rag-evaluation.evaluate_relevance_full",
            question=question,
            response=response,
            model=model
        )

        hallucination_task = router.app.call(
            "rag-evaluation.evaluate_hallucination_full",
            response=response,
            context=context,
            model=model
        )

        constitutional_task = router.app.call(
            "rag-evaluation.evaluate_constitutional_full",
            question=question,
            response=response,
            context=context,
            domain=domain,
            model=model
        )

        # Wait for all
        faithfulness, relevance, hallucination, constitutional = await asyncio.gather(
            faithfulness_task, relevance_task, hallucination_task, constitutional_task
        )

        router.note("Thorough evaluations complete", tags=["orchestration", "thorough"])

        return aggregate_results(
            faithfulness=faithfulness,
            relevance=relevance,
            hallucination=hallucination,
            constitutional=constitutional,
            mode="thorough",
            ai_calls=20  # Approximate
        )

    # ============================================
    # INDIVIDUAL METRIC ENDPOINTS
    # ============================================

    @router.reasoner()
    async def evaluate_faithfulness_only(
        response: str,
        context: str,
        mode: str = "full",
        model: Optional[str] = None
    ) -> dict:
        """
        Evaluate faithfulness only.

        Standalone endpoint for focused faithfulness evaluation.
        """
        router.note("Faithfulness-only evaluation", tags=["faithfulness", "standalone"])

        if mode == "full":
            return await router.app.call(
                "rag-evaluation.evaluate_faithfulness_full",
                response=response,
                context=context,
                model=model
            )
        else:
            return await router.app.call(
                "rag-evaluation.evaluate_faithfulness_quick",
                response=response,
                context=context,
                model=model
            )

    @router.reasoner()
    async def evaluate_relevance_only(
        question: str,
        response: str,
        mode: str = "full",
        model: Optional[str] = None
    ) -> dict:
        """
        Evaluate relevance only.

        Standalone endpoint for focused relevance evaluation.
        """
        router.note("Relevance-only evaluation", tags=["relevance", "standalone"])

        if mode == "full":
            return await router.app.call(
                "rag-evaluation.evaluate_relevance_full",
                question=question,
                response=response,
                model=model
            )
        else:
            return await router.app.call(
                "rag-evaluation.evaluate_relevance_quick",
                question=question,
                response=response,
                model=model
            )

    @router.reasoner()
    async def evaluate_hallucination_only(
        response: str,
        context: str,
        mode: str = "full",
        model: Optional[str] = None
    ) -> dict:
        """
        Evaluate hallucination only.

        Standalone endpoint for focused hallucination detection.
        """
        router.note("Hallucination-only evaluation", tags=["hallucination", "standalone"])

        if mode == "full":
            return await router.app.call(
                "rag-evaluation.evaluate_hallucination_full",
                response=response,
                context=context,
                model=model
            )
        else:
            return await router.app.call(
                "rag-evaluation.evaluate_hallucination_quick",
                response=response,
                context=context,
                model=model
            )

    @router.reasoner()
    async def evaluate_constitutional_only(
        question: str,
        response: str,
        context: str,
        domain: str = "general",
        mode: str = "full",
        model: Optional[str] = None
    ) -> dict:
        """
        Evaluate constitutional compliance only.

        Standalone endpoint for focused constitutional evaluation.
        """
        router.note("Constitutional-only evaluation", tags=["constitutional", "standalone"])

        if mode == "full":
            return await router.app.call(
                "rag-evaluation.evaluate_constitutional_full",
                question=question,
                response=response,
                context=context,
                domain=domain,
                model=model
            )
        else:
            return await router.app.call(
                "rag-evaluation.evaluate_constitutional_quick",
                question=question,
                response=response,
                context=context,
                model=model
            )

    # ============================================
    # HELPER FUNCTIONS
    # ============================================

    def aggregate_results(
        faithfulness: Dict,
        relevance: Dict,
        hallucination: Dict,
        constitutional: Dict,
        mode: str,
        ai_calls: int
    ) -> dict:
        """
        Aggregate metric results into unified evaluation report.
        """
        # Extract scores
        f_score = faithfulness.get("score", 0.5)
        r_score = relevance.get("overall_score", 0.5)
        h_score = hallucination.get("score", 0.5)
        c_score = constitutional.get("overall_score", 0.5)

        # Weighted average (faithfulness and hallucination weighted higher)
        overall = (
            f_score * 0.30 +
            r_score * 0.20 +
            h_score * 0.30 +
            c_score * 0.20
        )

        # Determine quality tier
        if overall >= 0.9:
            tier = "excellent"
        elif overall >= 0.75:
            tier = "good"
        elif overall >= 0.6:
            tier = "acceptable"
        elif overall >= 0.4:
            tier = "poor"
        else:
            tier = "critical"

        # Collect critical issues
        critical_issues = []
        if f_score < 0.5:
            critical_issues.append("Low faithfulness - response may not be grounded in context")
        if h_score < 0.5:
            critical_issues.append("Significant hallucinations detected")
        if constitutional.get("compliance_status") == "non_compliant":
            critical_issues.append("Constitutional violations found")

        # Human review needed?
        needs_review = (
            overall < 0.5 or
            len(critical_issues) > 0 or
            relevance.get("disagreement_level", 0) > 0.3
        )

        # Recommendations
        recommendations = []
        if f_score < 0.7:
            recommendations.append("Improve grounding in source material")
        if r_score < 0.7:
            recommendations.append("Better address the user's question")
        if h_score < 0.7:
            recommendations.append("Remove unsupported claims")
        if c_score < 0.7:
            recommendations.append("Review against evaluation principles")

        return RAGEvaluationResult(
            faithfulness=FaithfulnessVerdict(**faithfulness),
            relevance=RelevanceVerdict(**relevance),
            hallucination=HallucinationReport(**hallucination),
            constitutional=ConstitutionalReport(**constitutional),
            overall_score=overall,
            quality_tier=tier,
            evaluation_mode=mode,
            ai_calls_made=ai_calls,
            requires_human_review=needs_review,
            critical_issues=critical_issues,
            recommendations=recommendations
        ).model_dump()
