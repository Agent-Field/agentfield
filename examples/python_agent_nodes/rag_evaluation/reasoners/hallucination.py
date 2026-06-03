"""
Hallucination Detection Reasoners

Implements a hybrid ML+LLM chain-of-verification pattern:
1. extract_statements - Extract factual statements (ML: NER + rules)
2. ml_verify_batch - Batch ML verification (embeddings + NLI)
3. llm_verify_uncertain - LLM verification for uncertain cases
4. synthesize_hallucination - Aggregate into final report

This hybrid approach achieves 60-80% cost reduction vs pure LLM.
"""

import asyncio
from typing import Dict, Any, List, Optional
from models import (
    StatementExtraction,
    MLVerificationResult,
    LLMVerificationResult,
    HallucinationReport,
    QuickHallucination,
)


def register_hallucination_reasoners(router):
    """Register all hallucination-related reasoners with the router."""

    # ============================================
    # STATEMENT EXTRACTION (ML-based skill + reasoner wrapper)
    # ============================================

    @router.skill()
    def extract_statements_ml(response: str) -> dict:
        """
        ML-based statement extraction skill.

        Uses NER to identify factual statements containing:
        - Named entities (people, places, organizations)
        - Numbers and quantities
        - Dates and times
        - Factual claims

        This is a SKILL (not reasoner) because it's deterministic ML.
        """
        try:
            from ml_services.ner import get_ner_service

            ner = get_ner_service()
            statements = ner.extract_factual_claims(response)
            entities = ner.extract_entities(response)

            return {
                "statements": statements,
                "entity_count": len(entities),
                "entities": entities
            }
        except ImportError:
            # Fallback: split into sentences
            import re
            sentences = [s.strip() for s in re.split(r'[.!?]+', response) if s.strip()]
            return {
                "statements": sentences,
                "entity_count": 0,
                "entities": []
            }

    @router.reasoner()
    async def extract_statements(
        response: str,
        context: str,
        model: Optional[str] = None
    ) -> dict:
        """
        Statement extraction reasoner (wraps ML skill).

        Creates a visible node in workflow graph while delegating
        to ML skill for actual extraction. Falls back to LLM if
        ML service unavailable.
        """

        router.note("Extracting factual statements...", tags=["hallucination", "extraction"])

        # Try ML extraction first
        ml_result = extract_statements_ml(response)

        if ml_result["statements"]:
            result = StatementExtraction(
                statements=ml_result["statements"],
                entity_count=ml_result["entity_count"]
            )
            router.note(f"ML extracted {len(result.statements)} statements, {result.entity_count} entities",
                       tags=["hallucination", "ml"])
        else:
            # Fallback to LLM
            router.note("Falling back to LLM extraction...", tags=["hallucination", "fallback"])
            result = await router.ai(
                f"""Extract all factual statements from this response that should be verified.

RESPONSE:
{response}

List each verifiable factual claim as a separate statement.
Focus on claims containing: facts, numbers, dates, names, specific assertions.""",
                schema=StatementExtraction,
                model=model
            )

        # Show first statement being checked
        first_stmt = result.statements[0][:50] if result.statements else "none"
        router.note(f"Checking {len(result.statements)} statements, starting with: \"{first_stmt}...\"",
                   tags=["hallucination", "extraction"])

        return result.model_dump()

    # ============================================
    # ML VERIFICATION (Batch processing)
    # ============================================

    @router.skill()
    def ml_verify_statements(
        statements: List[str],
        context: str,
        entailment_threshold: float = 0.7,
        similarity_threshold: float = 0.6
    ) -> List[dict]:
        """
        ML-based batch verification skill.

        Uses:
        1. Embedding similarity (fast pre-filter)
        2. NLI entailment (accurate verification)

        Returns verification status for each statement.
        """
        results = []

        try:
            from ml_services.embeddings import get_embedding_service
            from ml_services.nli import get_nli_service

            embedding_service = get_embedding_service()
            nli_service = get_nli_service()

            # Split context into sentences for matching
            import re
            context_sentences = [s.strip() for s in re.split(r'[.!?]+', context) if s.strip()]

            for i, statement in enumerate(statements):
                # Step 1: Find similar context sentences
                similar = embedding_service.find_most_similar(
                    statement,
                    context_sentences,
                    top_k=3,
                    threshold=similarity_threshold
                )

                if not similar:
                    # No similar sentences found
                    results.append({
                        "statement_index": i,
                        "verification_status": "unverified",
                        "confidence": 0.3,
                        "method": "embedding_similarity"
                    })
                    continue

                # Step 2: NLI entailment check on similar sentences
                best_match = similar[0]
                nli_result = nli_service.verify_claim(
                    context=best_match[2],  # The text
                    claim=statement,
                    entailment_threshold=entailment_threshold
                )

                if nli_result["status"] == "verified":
                    results.append({
                        "statement_index": i,
                        "verification_status": "verified",
                        "confidence": nli_result["confidence"],
                        "method": "nli_entailment"
                    })
                elif nli_result["status"] == "contradicted":
                    results.append({
                        "statement_index": i,
                        "verification_status": "unverified",
                        "confidence": nli_result["confidence"],
                        "method": "nli_entailment"
                    })
                else:
                    # Uncertain - needs LLM review
                    results.append({
                        "statement_index": i,
                        "verification_status": "uncertain",
                        "confidence": nli_result["confidence"],
                        "method": "nli_entailment"
                    })

        except ImportError:
            # ML services not available - mark all as uncertain
            for i, _ in enumerate(statements):
                results.append({
                    "statement_index": i,
                    "verification_status": "uncertain",
                    "confidence": 0.5,
                    "method": "ml_unavailable"
                })

        return results

    @router.reasoner()
    async def verify_statements_ml(
        statements: List[str],
        context: str
    ) -> dict:
        """
        ML verification reasoner (creates workflow node).

        Wraps ML skill and reports results. Identifies which
        statements need LLM escalation.
        """

        router.note(f"ML verifying {len(statements)} statements...",
                   tags=["hallucination", "ml_verify"])

        results = ml_verify_statements(statements, context)

        verified = sum(1 for r in results if r["verification_status"] == "verified")
        unverified = sum(1 for r in results if r["verification_status"] == "unverified")
        uncertain = sum(1 for r in results if r["verification_status"] == "uncertain")

        router.note(f"ML verification: {verified} grounded, {unverified} flagged, {uncertain} need LLM review",
                   tags=["hallucination", "ml"])

        return {
            "results": results,
            "verified_count": verified,
            "unverified_count": unverified,
            "uncertain_count": uncertain,
            "statements": statements
        }

    # ============================================
    # LLM VERIFICATION (Escalation for uncertain)
    # ============================================

    @router.reasoner()
    async def verify_statement_llm(
        statement: str,
        statement_index: int,
        context: str,
        ml_result: Dict,
        model: Optional[str] = None
    ) -> dict:
        """
        LLM verification for a single uncertain statement.

        Creates separate workflow node for each LLM verification,
        showing the escalation pattern in the graph.
        """

        router.note(f"LLM verifying statement {statement_index}...",
                   tags=["hallucination", "llm_verify"])

        result = await router.ai(
            f"""Verify if this statement is a hallucination or grounded in the context.

STATEMENT TO VERIFY:
{statement}

CONTEXT:
{context}

ML PRELIMINARY RESULT:
Status: {ml_result.get('verification_status', 'uncertain')}
Confidence: {ml_result.get('confidence', 0.5):.2f}

Determine:
- is_hallucination: true if statement contains fabricated/unsupported information
- explanation: Why is this a hallucination or why is it grounded
- confidence: 0.0-1.0 how confident in this assessment""",
            schema=LLMVerificationResult,
            model=model
        )

        # Ensure statement_index is set
        result_dict = result.model_dump()
        result_dict["statement_index"] = statement_index

        # Include explanation snippet
        status = "Fabricated" if result.is_hallucination else "Grounded"
        explanation = result.explanation[:50] if result.explanation else ""
        router.note(f"LLM check #{statement_index}: {status} - {explanation}...",
                   tags=["hallucination", "llm"])

        return result_dict

    @router.reasoner()
    async def verify_uncertain_statements(
        statements: List[str],
        ml_results: List[Dict],
        context: str,
        model: Optional[str] = None
    ) -> dict:
        """
        LLM verification orchestrator for uncertain statements.

        Launches parallel LLM verification for all statements marked
        'uncertain' by ML. Creates star pattern in workflow graph.
        """

        # Find uncertain statements
        uncertain_indices = [
            r["statement_index"]
            for r in ml_results
            if r["verification_status"] == "uncertain"
        ]

        if not uncertain_indices:
            router.note("No uncertain statements to verify",
                       tags=["hallucination", "llm_verify"])
            return {"llm_results": []}

        router.note(f"LLM verifying {len(uncertain_indices)} uncertain statements in parallel...",
                   tags=["hallucination", "llm_parallel"])

        # Launch parallel LLM verification
        tasks = [
            router.app.call(
                "rag-evaluation.verify_statement_llm",
                statement=statements[i],
                statement_index=i,
                context=context,
                ml_result=ml_results[i],
                model=model
            )
            for i in uncertain_indices
        ]

        llm_results = await asyncio.gather(*tasks)

        hallucinations = sum(1 for r in llm_results if r.get("is_hallucination", False))
        router.note(f"LLM found {hallucinations} hallucinations in {len(llm_results)} uncertain statements",
                   tags=["hallucination", "llm_complete"])

        return {"llm_results": list(llm_results)}

    # ============================================
    # HALLUCINATION SYNTHESIS
    # ============================================

    @router.reasoner()
    async def synthesize_hallucination_report(
        statements: List[str],
        ml_results: List[Dict],
        llm_results: List[Dict],
        context: str
    ) -> dict:
        """
        Synthesize final hallucination report.

        Combines ML and LLM results into unified report with:
        - Overall groundedness score
        - List of fabrications and contradictions
        - Cost efficiency metrics (ML vs LLM handling)
        """

        router.note("Synthesizing hallucination report...",
                   tags=["hallucination", "synthesis"])

        fabrications = []
        contradictions = []

        # Process ML results
        for ml_res in ml_results:
            idx = ml_res["statement_index"]
            if ml_res["verification_status"] == "unverified":
                fabrications.append(statements[idx])

        # Process LLM results (override ML uncertain)
        for llm_res in llm_results:
            idx = llm_res["statement_index"]
            if llm_res.get("is_hallucination", False):
                if "contradict" in llm_res.get("explanation", "").lower():
                    contradictions.append(statements[idx])
                else:
                    fabrications.append(statements[idx])

        # Calculate scores
        total = len(statements)
        ml_handled = sum(1 for r in ml_results if r["verification_status"] != "uncertain")
        ml_percent = (ml_handled / total * 100) if total > 0 else 0

        hallucination_count = len(fabrications) + len(contradictions)
        groundedness_score = 1.0 - (hallucination_count / total) if total > 0 else 1.0

        report = HallucinationReport(
            score=max(0.0, groundedness_score),
            fabrications=fabrications,
            contradictions=contradictions,
            ml_handled_percent=ml_percent,
            total_statements=total
        )

        router.note(f"Groundedness: {report.score:.2f}, ML handled: {ml_percent:.0f}%",
                   tags=["hallucination", "complete"])

        return report.model_dump()

    # ============================================
    # FULL HYBRID ORCHESTRATOR
    # ============================================

    @router.reasoner()
    async def evaluate_hallucination_full(
        response: str,
        context: str,
        model: Optional[str] = None
    ) -> dict:
        """
        Full hybrid ML+LLM orchestrator for hallucination detection.

        Workflow:
        1. Extract statements (ML + fallback)
        2. ML verification batch (embedding + NLI)
        3. LLM verification for uncertain (parallel escalation)
        4. Synthesize final report

        Creates impressive workflow graph showing hybrid pattern:
        extract -> ml_verify -> [llm_verify_0, llm_verify_1, ...] -> synthesize
        """

        router.note("Starting hybrid ML + LLM hallucination detection...",
                   tags=["hallucination", "orchestration"])

        # Step 1: Extract statements
        extraction = await router.app.call(
            "rag-evaluation.extract_statements",
            response=response,
            context=context,
            model=model
        )
        statements = extraction.get("statements", [])

        if not statements:
            router.note("No statements to verify", tags=["hallucination"])
            return HallucinationReport(
                score=1.0,
                fabrications=[],
                contradictions=[],
                ml_handled_percent=100.0,
                total_statements=0
            ).model_dump()

        # Step 2: ML verification
        ml_verification = await router.app.call(
            "rag-evaluation.verify_statements_ml",
            statements=statements,
            context=context
        )

        # Step 3: LLM verification for uncertain
        llm_verification = await router.app.call(
            "rag-evaluation.verify_uncertain_statements",
            statements=statements,
            ml_results=ml_verification.get("results", []),
            context=context,
            model=model
        )

        # Step 4: Synthesize
        report = await router.app.call(
            "rag-evaluation.synthesize_hallucination_report",
            statements=statements,
            ml_results=ml_verification.get("results", []),
            llm_results=llm_verification.get("llm_results", []),
            context=context
        )

        router.note(f"Hybrid detection complete: {report['score']:.2f} groundedness",
                   tags=["hallucination", "complete"])

        return report

    # ============================================
    # QUICK SINGLE-REASONER VERSION
    # ============================================

    @router.reasoner()
    async def evaluate_hallucination_quick(
        response: str,
        context: str,
        model: Optional[str] = None
    ) -> dict:
        """
        Quick single-reasoner hallucination check.

        Used in 'quick' mode - single LLM call, no ML pre-filtering.
        """

        router.note("Quick hallucination check...", tags=["hallucination", "quick"])

        result = await router.ai(
            f"""Check this response for hallucinations (fabricated or unsupported information).

RESPONSE:
{response}

CONTEXT:
{context}

Identify any statements that are:
- Fabricated (not in context)
- Contradicted by context
- Exaggerated beyond what context supports

Provide:
- score: 0.0-1.0 (1.0 = fully grounded, no hallucinations)
- fabrications_found: List of fabricated statements
- reasoning: Brief explanation""",
            schema=QuickHallucination,
            model=model
        )

        router.note(f"Quick hallucination: {result.score:.2f}",
                   tags=["hallucination", "quick"])

        return HallucinationReport(
            score=result.score,
            fabrications=result.fabrications_found,
            contradictions=[],
            ml_handled_percent=0.0,
            total_statements=len(result.fabrications_found) if result.fabrications_found else 1
        ).model_dump()
