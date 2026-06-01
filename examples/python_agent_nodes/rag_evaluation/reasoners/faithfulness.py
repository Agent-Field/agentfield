"""
Faithfulness Evaluation Reasoners

Implements an adversarial debate pattern:
1. claim_extractor - Decompose response into atomic claims
2. prosecutor_reasoner - Find unsupported/contradicted claims (ATTACK)
3. defender_reasoner - Find supporting evidence for claims (DEFEND)
4. faithfulness_judge - Weigh arguments, produce final verdict

This pattern reduces bias by forcing consideration of both perspectives.
"""

import asyncio
from typing import Dict, Any, List, Optional
from models import (
    ClaimExtraction,
    AtomicClaim,
    ProsecutorAnalysis,
    ProsecutorAttack,
    DefenderAnalysis,
    DefenderDefense,
    FaithfulnessVerdict,
    QuickFaithfulness,
)


def register_faithfulness_reasoners(router):
    """Register all faithfulness-related reasoners with the router."""

    # ============================================
    # CLAIM EXTRACTION
    # ============================================

    @router.reasoner()
    async def extract_claims(
        response: str,
        context: str,
        model: Optional[str] = None
    ) -> dict:
        """
        Claim extractor: Decompose response into atomic claims.

        Analyzes the RAG response and breaks it down into individual
        factual assertions that can be independently verified against
        the context. Each claim is classified by type and importance.

        Creates the foundation for the adversarial debate.
        """

        router.note("Extracting atomic claims from response for verification...", tags=["faithfulness", "extraction"])

        result = await router.ai(
            f"""Analyze this RAG response and extract all factual claims.

Response to analyze:
{response}

For each claim, identify:
1. The exact factual assertion
2. Type: factual (direct fact), inferential (derived conclusion), opinion (subjective)
3. Importance: critical (key info), supporting (helpful detail), minor (trivial)

Extract EVERY verifiable statement as a separate claim.""",
            schema=ClaimExtraction,
            model=model
        )

        # Include first claim text as dynamic content
        first_claim = result.claims[0].claim_text if result.claims else "none found"
        router.note(f"Found {result.total_count} claims to verify, starting with: \"{first_claim[:60]}...\"",
                   tags=["faithfulness", "extraction"])

        return result.model_dump()

    # ============================================
    # PROSECUTOR (ATTACK)
    # ============================================

    @router.reasoner()
    async def prosecute_claims(
        claims: List[Dict],
        context: str,
        response: str,
        model: Optional[str] = None
    ) -> dict:
        """
        Prosecutor: Adversarial attack on claims.

        Acts as an aggressive prosecutor seeking to find:
        - Unsupported claims (no evidence in context)
        - Contradicted claims (context says otherwise)
        - Exaggerated claims (overstated relative to source)
        - Out-of-context claims (misusing source information)

        The prosecutor's job is to find ALL possible issues.
        """

        router.note("Prosecutor examining claims for issues...",
                   tags=["faithfulness", "prosecutor"])

        claims_text = "\n".join([
            f"{i+1}. [{c.get('claim_type', 'unknown')}] {c.get('claim_text', c)}"
            for i, c in enumerate(claims)
        ])

        result = await router.ai(
            f"""You are an aggressive PROSECUTOR. Your job is to find EVERY possible issue
with these claims relative to the source context.

CLAIMS TO ATTACK:
{claims_text}

SOURCE CONTEXT:
{context}

ORIGINAL RESPONSE:
{response}

For each problematic claim, identify:
- claim_index: Which claim (0-indexed)
- attack_type: unsupported, contradicted, exaggerated, or out_of_context
- evidence: Specific reason this claim is problematic
- severity: critical (major falsehood), major (significant issue), minor (small problem)

Be AGGRESSIVE - find every possible issue. Better to over-attack than miss problems.
The defender will have a chance to respond.""",
            schema=ProsecutorAnalysis,
            model=model
        )

        # Include summary of attack types
        attack_types = [a.attack_type for a in result.attacks] if result.attacks else []
        summary = result.prosecution_summary[:80] if result.prosecution_summary else "No issues found"
        router.note(f"Prosecutor: {summary}...",
                   tags=["faithfulness", "prosecution"])

        return result.model_dump()

    # ============================================
    # DEFENDER (DEFENSE)
    # ============================================

    @router.reasoner()
    async def defend_claims(
        claims: List[Dict],
        context: str,
        response: str,
        attacks: List[Dict],
        model: Optional[str] = None
    ) -> dict:
        """
        Defender: Evidence-based defense of claims.

        Acts as a defense attorney seeking to find:
        - Direct support (explicit evidence in context)
        - Implicit support (reasonable interpretation)
        - Valid inference (logical conclusion from context)
        - Acknowledged issues (accepts valid attacks)

        The defender must be fair - acknowledge real problems.
        """

        router.note("Defender building case for claims...",
                   tags=["faithfulness", "defender"])

        claims_text = "\n".join([
            f"{i+1}. {c.get('claim_text', c)}"
            for i, c in enumerate(claims)
        ])

        attacks_text = "\n".join([
            f"- Claim {a['claim_index']}: {a['attack_type']} - {a['evidence']}"
            for a in attacks
        ]) if attacks else "No attacks to defend against."

        result = await router.ai(
            f"""You are a DEFENSE ATTORNEY. Your job is to find evidence supporting these claims
AND honestly address the prosecutor's attacks.

CLAIMS TO DEFEND:
{claims_text}

SOURCE CONTEXT:
{context}

PROSECUTOR'S ATTACKS:
{attacks_text}

For each claim (especially attacked ones), provide:
- claim_index: Which claim (0-indexed)
- defense_type: direct_support, implicit_support, reasonable_inference, or acknowledged_issue
- evidence: Quote or reasoning from context that supports the claim
- strength: 0.0-1.0 how strong is this defense

Be FAIR - if a claim truly has no support, acknowledge it as 'acknowledged_issue'.
Your credibility depends on honesty.""",
            schema=DefenderAnalysis,
            model=model
        )

        # Include defense summary
        summary = result.defense_summary[:80] if result.defense_summary else "Defense arguments presented"
        router.note(f"Defender: {summary}...",
                   tags=["faithfulness", "defense"])

        return result.model_dump()

    # ============================================
    # JUDGE (VERDICT)
    # ============================================

    @router.reasoner()
    async def judge_faithfulness(
        claims: List[Dict],
        prosecution: Dict,
        defense: Dict,
        context: str,
        model: Optional[str] = None
    ) -> dict:
        """
        Judge: Impartial verdict on faithfulness.

        Weighs prosecutor's attacks against defender's evidence to
        produce a fair final verdict. The judge:
        - Considers both sides' arguments
        - Evaluates evidence quality
        - Produces explained ruling for each disputed claim
        - Calculates overall faithfulness score
        """

        router.note("Judge deliberating on evidence...",
                   tags=["faithfulness", "judge"])

        claims_summary = "\n".join([
            f"{i+1}. {c.get('claim_text', c)}"
            for i, c in enumerate(claims)
        ])

        attacks_summary = "\n".join([
            f"- Attack on claim {a['claim_index']}: {a['attack_type']} ({a['severity']}) - {a['evidence']}"
            for a in prosecution.get("attacks", [])
        ]) if prosecution.get("attacks") else "No attacks."

        defenses_summary = "\n".join([
            f"- Defense of claim {d['claim_index']}: {d['defense_type']} (strength {d['strength']:.2f}) - {d['evidence']}"
            for d in defense.get("defenses", [])
        ]) if defense.get("defenses") else "No defenses."

        result = await router.ai(
            f"""You are an IMPARTIAL JUDGE. Weigh the evidence and render a verdict on faithfulness.

CLAIMS AT ISSUE:
{claims_summary}

PROSECUTION CASE:
{prosecution.get("prosecution_summary", "No summary")}
{attacks_summary}

DEFENSE CASE:
{defense.get("defense_summary", "No summary")}
{defenses_summary}

SOURCE CONTEXT:
{context}

Render your verdict:
1. score: 0.0-1.0 overall faithfulness (1.0 = fully faithful)
2. unfaithful_claims: List the actual TEXT of claims you rule unfaithful
3. debate_summary: Key points from the debate that influenced your decision
4. reasoning: Your judicial reasoning for the verdict

Be FAIR and IMPARTIAL. Only rule claims unfaithful if the prosecution proved its case.""",
            schema=FaithfulnessVerdict,
            model=model
        )

        # Include key reasoning from judge
        reason_snippet = result.reasoning[:80] if result.reasoning else "Verdict reached"
        router.note(f"Judge ruling ({result.score:.0%} faithful): {reason_snippet}...",
                   tags=["faithfulness", "judge", "verdict"])

        return result.model_dump()

    # ============================================
    # FULL DEBATE ORCHESTRATOR
    # ============================================

    @router.reasoner()
    async def evaluate_faithfulness_full(
        response: str,
        context: str,
        model: Optional[str] = None
    ) -> dict:
        """
        Full adversarial debate orchestrator for faithfulness.

        Workflow:
        1. Extract claims from response
        2. Prosecutor attacks claims (parallel)
        3. Defender defends claims (parallel, needs attacks)
        4. Judge renders final verdict

        Creates impressive workflow graph:
        extract_claims -> [prosecutor, defender] -> judge
        """

        router.note("Starting adversarial debate: prosecutor vs defender...",
                   tags=["faithfulness", "orchestration"])

        # Step 1: Extract claims
        claims_result = await router.app.call(
            "rag-evaluation.extract_claims",
            response=response,
            context=context,
            model=model
        )
        claims = claims_result.get("claims", [])

        if not claims:
            router.note("No claims found to evaluate", tags=["faithfulness"])
            return FaithfulnessVerdict(
                score=1.0,
                unfaithful_claims=[],
                debate_summary="No factual claims found in response",
                reasoning="Response contains no verifiable claims"
            ).model_dump()

        # Step 2: Prosecutor attacks (can start immediately)
        prosecution_task = router.app.call(
            "rag-evaluation.prosecute_claims",
            claims=claims,
            context=context,
            response=response,
            model=model
        )

        # Wait for prosecution first (defender needs attacks)
        prosecution = await prosecution_task

        # Step 3: Defender responds to attacks
        defense = await router.app.call(
            "rag-evaluation.defend_claims",
            claims=claims,
            context=context,
            response=response,
            attacks=prosecution.get("attacks", []),
            model=model
        )

        # Step 4: Judge renders verdict
        verdict = await router.app.call(
            "rag-evaluation.judge_faithfulness",
            claims=claims,
            prosecution=prosecution,
            defense=defense,
            context=context,
            model=model
        )

        router.note(f"Debate complete: {verdict['score']:.2f} faithfulness",
                   tags=["faithfulness", "complete"])

        return verdict

    # ============================================
    # SIMPLIFIED SINGLE-REASONER VERSION
    # ============================================

    @router.reasoner()
    async def evaluate_faithfulness_quick(
        response: str,
        context: str,
        model: Optional[str] = None
    ) -> dict:
        """
        Quick single-reasoner faithfulness evaluation.

        Used in 'quick' mode for faster, cheaper evaluation.
        Trades depth for speed - single LLM call instead of debate.
        """

        router.note("Quick faithfulness check...", tags=["faithfulness", "quick"])

        result = await router.ai(
            f"""Evaluate the faithfulness of this response to the given context.

RESPONSE:
{response}

CONTEXT:
{context}

Check if the response is grounded in the context:
- Are all claims supported by the context?
- Are there any fabrications or hallucinations?
- Does the response contradict the context?

Provide:
- score: 0.0-1.0 (1.0 = fully faithful)
- issues: List of specific faithfulness issues found
- reasoning: Brief explanation""",
            schema=QuickFaithfulness,
            model=model
        )

        router.note(f"Quick faithfulness: {result.score:.2f}",
                   tags=["faithfulness", "quick"])

        # Convert to full verdict format for consistency
        return FaithfulnessVerdict(
            score=result.score,
            unfaithful_claims=result.issues,
            debate_summary="Quick single-reasoner evaluation (no debate)",
            reasoning=result.reasoning
        ).model_dump()
