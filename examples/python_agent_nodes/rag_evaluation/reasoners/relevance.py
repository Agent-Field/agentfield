"""
Relevance Evaluation Reasoners

Implements a multi-jury consensus pattern:
1. question_analyst - Extract question intent and sub-questions
2. literal_juror - Does response literally answer the question?
3. intent_juror - Does response address underlying user need?
4. scope_juror - Is response appropriately scoped?
5. jury_foreman - Aggregate votes, handle disagreements

This pattern captures multiple dimensions of relevance.
"""

import asyncio
from typing import Dict, Any, List, Optional
from models import (
    QuestionIntent,
    JurorVote,
    RelevanceVerdict,
    QuickRelevance,
)


def register_relevance_reasoners(router):
    """Register all relevance-related reasoners with the router."""

    # ============================================
    # QUESTION ANALYSIS
    # ============================================

    @router.reasoner()
    async def analyze_question(
        question: str,
        model: Optional[str] = None
    ) -> dict:
        """
        Question analyst: Extract intent and requirements.

        Analyzes the user's question to understand:
        - Primary intent (what they really want to know)
        - Implicit sub-questions
        - Expected answer type (factual, explanation, list, etc.)

        This analysis guides the jury's evaluation.
        """

        router.note("Analyzing question intent...", tags=["relevance", "analysis"])

        result = await router.ai(
            f"""Analyze this user question to understand what they're really asking.

QUESTION:
{question}

Identify:
1. primary_intent: What does the user REALLY want to know? (beyond literal words)
2. sub_questions: What implicit questions are embedded in this? (list of strings)
3. expected_type: What type of answer is expected?
   - factual: specific fact/answer
   - explanation: understanding of how/why
   - list: enumeration of items
   - comparison: contrasting options
   - procedure: step-by-step instructions
   - opinion: subjective assessment""",
            schema=QuestionIntent,
            model=model
        )

        # Include the identified intent
        router.note(f"Question intent: \"{result.primary_intent[:70]}...\"",
                   tags=["relevance", "analysis"])

        return result.model_dump()

    # ============================================
    # LITERAL JUROR
    # ============================================

    @router.reasoner()
    async def vote_literal_relevance(
        question: str,
        response: str,
        question_analysis: Dict,
        model: Optional[str] = None
    ) -> dict:
        """
        Literal juror: Does response literally answer the question?

        Focuses on surface-level match between question and answer:
        - Are the question words/entities addressed?
        - Does the response directly answer what was asked?
        - Is there a clear correspondence?

        This juror catches responses that miss the literal question.
        """

        router.note("Literal juror evaluating...", tags=["relevance", "literal"])

        result = await router.ai(
            f"""You are the LITERAL JUROR. Evaluate if the response directly answers the question.

QUESTION:
{question}

RESPONSE:
{response}

QUESTION ANALYSIS:
Expected type: {question_analysis.get('expected_type', 'unknown')}
Sub-questions: {question_analysis.get('sub_questions', [])}

Evaluate LITERALLY:
- Does the response address the exact words/terms in the question?
- Is there a direct answer to the literal question asked?
- Are the key entities from the question discussed?

Provide:
- dimension: "literal" (this is your focus)
- score: 0.0-1.0 (1.0 = perfectly answers literal question)
- reasoning: Why this score?
- confidence: 0.0-1.0 how confident in your assessment""",
            schema=JurorVote,
            model=model
        )

        # Include reasoning snippet
        reason = result.reasoning[:60] if result.reasoning else "Evaluated"
        router.note(f"Literal juror ({result.score:.0%}): {reason}...",
                   tags=["relevance", "jury", "literal"])

        return result.model_dump()

    # ============================================
    # INTENT JUROR
    # ============================================

    @router.reasoner()
    async def vote_intent_relevance(
        question: str,
        response: str,
        question_analysis: Dict,
        model: Optional[str] = None
    ) -> dict:
        """
        Intent juror: Does response address underlying user need?

        Focuses on deeper intent behind the question:
        - What is the user trying to accomplish?
        - Does the response help them achieve that goal?
        - Would the user be satisfied with this answer?

        This juror catches technically correct but unhelpful responses.
        """

        router.note("Intent juror evaluating...", tags=["relevance", "intent"])

        result = await router.ai(
            f"""You are the INTENT JUROR. Evaluate if the response addresses the user's TRUE NEED.

QUESTION:
{question}

RESPONSE:
{response}

PRIMARY INTENT IDENTIFIED:
{question_analysis.get('primary_intent', 'Unknown')}

Evaluate INTENT:
- Does the response help the user achieve their actual goal?
- Would a reasonable user be satisfied with this answer?
- Does it address the "why" behind asking this question?

Provide:
- dimension: "intent" (this is your focus)
- score: 0.0-1.0 (1.0 = perfectly addresses intent)
- reasoning: Why this score?
- confidence: 0.0-1.0 how confident in your assessment""",
            schema=JurorVote,
            model=model
        )

        # Include reasoning snippet
        reason = result.reasoning[:60] if result.reasoning else "Evaluated"
        router.note(f"Intent juror ({result.score:.0%}): {reason}...",
                   tags=["relevance", "jury", "intent"])

        return result.model_dump()

    # ============================================
    # SCOPE JUROR
    # ============================================

    @router.reasoner()
    async def vote_scope_relevance(
        question: str,
        response: str,
        question_analysis: Dict,
        model: Optional[str] = None
    ) -> dict:
        """
        Scope juror: Is response appropriately scoped?

        Focuses on whether the response is:
        - Not too narrow (missing important aspects)
        - Not too broad (including irrelevant information)
        - Appropriately detailed for the question

        This juror catches scope mismatches.
        """

        router.note("Scope juror evaluating...", tags=["relevance", "scope"])

        result = await router.ai(
            f"""You are the SCOPE JUROR. Evaluate if the response has appropriate scope.

QUESTION:
{question}

RESPONSE:
{response}

SUB-QUESTIONS TO ADDRESS:
{question_analysis.get('sub_questions', [])}

Evaluate SCOPE:
- Is the response too narrow? (missing important aspects)
- Is the response too broad? (including irrelevant tangents)
- Is the level of detail appropriate for the question?
- Are all sub-questions addressed?

Provide:
- dimension: "scope" (this is your focus)
- score: 0.0-1.0 (1.0 = perfectly scoped)
- reasoning: Why this score?
- confidence: 0.0-1.0 how confident in your assessment""",
            schema=JurorVote,
            model=model
        )

        # Include reasoning snippet
        reason = result.reasoning[:60] if result.reasoning else "Evaluated"
        router.note(f"Scope juror ({result.score:.0%}): {reason}...",
                   tags=["relevance", "jury", "scope"])

        return result.model_dump()

    # ============================================
    # JURY FOREMAN (SYNTHESIS)
    # ============================================

    @router.reasoner()
    async def synthesize_relevance_verdict(
        question: str,
        response: str,
        literal_vote: Dict,
        intent_vote: Dict,
        scope_vote: Dict,
        model: Optional[str] = None
    ) -> dict:
        """
        Jury foreman: Synthesize all juror votes into final verdict.

        Responsibilities:
        - Aggregate scores with appropriate weighting
        - Identify and resolve significant disagreements
        - Produce explained final verdict
        - Flag high-disagreement cases for human review
        """

        router.note("Jury foreman synthesizing votes...", tags=["relevance", "synthesis"])

        # Calculate disagreement
        scores = [
            literal_vote.get("score", 0.5),
            intent_vote.get("score", 0.5),
            scope_vote.get("score", 0.5)
        ]
        max_score = max(scores)
        min_score = min(scores)
        disagreement = max_score - min_score

        # Weight by confidence
        literal_conf = literal_vote.get("confidence", 0.5)
        intent_conf = intent_vote.get("confidence", 0.5)
        scope_conf = scope_vote.get("confidence", 0.5)

        total_conf = literal_conf + intent_conf + scope_conf
        if total_conf > 0:
            weighted_score = (
                literal_vote.get("score", 0.5) * literal_conf +
                intent_vote.get("score", 0.5) * intent_conf +
                scope_vote.get("score", 0.5) * scope_conf
            ) / total_conf
        else:
            weighted_score = sum(scores) / 3

        # Generate verdict summary
        result = await router.ai(
            f"""You are the JURY FOREMAN. Synthesize the votes into a final verdict.

QUESTION: {question}
RESPONSE: {response}

JURY VOTES:
- Literal Juror: {literal_vote.get('score', 0):.2f} - {literal_vote.get('reasoning', 'No reason')}
- Intent Juror: {intent_vote.get('score', 0):.2f} - {intent_vote.get('reasoning', 'No reason')}
- Scope Juror: {scope_vote.get('score', 0):.2f} - {scope_vote.get('reasoning', 'No reason')}

DISAGREEMENT LEVEL: {disagreement:.2f}

Provide a summary verdict explaining the jury's decision.""",
            schema=None,  # Just get text summary
            model=model
        )

        verdict = RelevanceVerdict(
            overall_score=weighted_score,
            literal_score=literal_vote.get("score", 0.5),
            intent_score=intent_vote.get("score", 0.5),
            scope_score=scope_vote.get("score", 0.5),
            disagreement_level=disagreement,
            verdict=str(result) if result else "Jury has reached a verdict."
        )

        # Include the verdict summary
        verdict_snippet = verdict.verdict[:70] if verdict.verdict else "Jury has reached consensus"
        router.note(f"Jury verdict ({verdict.overall_score:.0%}): {verdict_snippet}...",
                   tags=["relevance", "jury", "verdict"])

        return verdict.model_dump()

    # ============================================
    # FULL JURY ORCHESTRATOR
    # ============================================

    @router.reasoner()
    async def evaluate_relevance_full(
        question: str,
        response: str,
        model: Optional[str] = None
    ) -> dict:
        """
        Full multi-jury orchestrator for relevance.

        Workflow:
        1. Analyze question intent
        2. Three jurors vote in parallel (literal, intent, scope)
        3. Foreman synthesizes final verdict

        Creates workflow graph:
        analyze_question -> [literal, intent, scope] -> foreman
        """

        router.note("Convening jury for relevance assessment...", tags=["relevance", "orchestration", "jury"])

        # Step 1: Analyze the question
        question_analysis = await router.app.call(
            "rag-evaluation.analyze_question",
            question=question,
            model=model
        )

        router.note("Jury deliberating in parallel...", tags=["relevance", "parallel"])

        # Step 2: All three jurors vote in parallel
        literal_task = router.app.call(
            "rag-evaluation.vote_literal_relevance",
            question=question,
            response=response,
            question_analysis=question_analysis,
            model=model
        )

        intent_task = router.app.call(
            "rag-evaluation.vote_intent_relevance",
            question=question,
            response=response,
            question_analysis=question_analysis,
            model=model
        )

        scope_task = router.app.call(
            "rag-evaluation.vote_scope_relevance",
            question=question,
            response=response,
            question_analysis=question_analysis,
            model=model
        )

        # Wait for all jurors
        literal_vote, intent_vote, scope_vote = await asyncio.gather(
            literal_task, intent_task, scope_task
        )

        router.note("All jurors have voted", tags=["relevance", "parallel"])

        # Step 3: Foreman synthesizes
        verdict = await router.app.call(
            "rag-evaluation.synthesize_relevance_verdict",
            question=question,
            response=response,
            literal_vote=literal_vote,
            intent_vote=intent_vote,
            scope_vote=scope_vote,
            model=model
        )

        router.note(f"Jury complete: {verdict['overall_score']:.2f} relevance",
                   tags=["relevance", "complete"])

        return verdict

    # ============================================
    # QUICK SINGLE-REASONER VERSION
    # ============================================

    @router.reasoner()
    async def evaluate_relevance_quick(
        question: str,
        response: str,
        model: Optional[str] = None
    ) -> dict:
        """
        Quick single-reasoner relevance evaluation.

        Used in 'quick' mode for faster, cheaper evaluation.
        """

        router.note("Quick relevance check...", tags=["relevance", "quick"])

        result = await router.ai(
            f"""Evaluate the relevance of this response to the question.

QUESTION:
{question}

RESPONSE:
{response}

Check:
- Does the response answer the question asked?
- Is it helpful for the user's needs?
- Is it appropriately scoped?

Provide:
- score: 0.0-1.0 (1.0 = perfectly relevant)
- addresses_question: true/false
- reasoning: Brief explanation""",
            schema=QuickRelevance,
            model=model
        )

        router.note(f"Quick relevance: {result.score:.2f}",
                   tags=["relevance", "quick"])

        # Convert to full verdict format
        return RelevanceVerdict(
            overall_score=result.score,
            literal_score=result.score,
            intent_score=result.score,
            scope_score=result.score,
            disagreement_level=0.0,
            verdict=result.reasoning
        ).model_dump()
