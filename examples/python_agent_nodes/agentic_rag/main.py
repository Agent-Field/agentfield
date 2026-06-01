"""
Meta-Cognitive Agentic RAG - Self-Reasoning Multi-Agent System

Architecture Philosophy:
- Composed 0.3 LLMs → 0.8 system capability
- No hardcoded rules - AI reasons about requirements dynamically
- Simple schemas (2-4 fields) - One reasoner, one job
- Intelligence emerges from clever composition
- String for reasoning → passed as context
- Structured for decisions → programmatic routing

12-Phase Meta-Cognitive Pipeline:
1. Document chunking
2. Query introspection (AI analyzes query semantics)
3. Precision reasoning (AI decides match requirements)
4. Strategy composition (AI designs retrieval plan)
5. Ensemble retrieval (semantic + keyword + type-specific)
6. Self-aware match quality (AI evaluates WHY chunks matched)
7. Answer synthesis with entity extraction
8. Semantic drift detection (catches entity substitutions)
9. Answer-query alignment verification (final gate)
10. Adaptive confidence calibration (AI-based, not thresholds)
11. Conditional routing (return answer or "no info found")
12. Citation building

Key Innovation: Hallucination Prevention Through Meta-Cognition
- AI detects its own entity mismatches
- "Yann Klegal" ≠ "Yann LeCun" caught by drift detection
- High confidence (0.95) in negative answers when appropriate
"""

import os
import asyncio
from typing import List, Dict
from agentfield import Agent, AIConfig
from schemas import (
    SubQuestion,
    SubQuestions,
    DraftAnswer,
    Claim,
    VerificationResult,
    Citation,
    VerifiedAnswer,
    RankedChunk,
    ChunkList,
    Chunk,
    Gap,
    RefinementQuery,
    SearchTerms,
    GapList,
    RefinementQueryList,
    ClaimList,
    CompletenessScore,
    # Meta-cognitive schemas
    QuerySemantics,
    PrecisionRequirements,
    RetrievalPlan,
    MatchQuality,
    DraftAnswerWithEntities,
    DriftAnalysis,
    AlignmentVerdict,
    FinalConfidence,
)
from skills import (
    load_document,
    simple_chunk_text,
    extract_keywords,
    keyword_match_score,
    find_quote_in_chunk,
    deduplicate_chunks,
    embed_text,
    embed_batch,
)

# Initialize agent
app = Agent(
    node_id="agentic-rag",
    agentfield_server=os.getenv("AGENTFIELD_SERVER", "http://localhost:8080"),
    ai_config=AIConfig(
        model=os.getenv("AI_MODEL", "openai/gpt-4.1-mini"), temperature=0.3
    ),
)


# ============= PHASE 1: SMART CHUNKING =============


@app.skill()
async def chunk_document(file_path: str) -> ChunkList:
    """Load and intelligently chunk document"""
    content = load_document(file_path)
    chunk_dicts = simple_chunk_text(content, chunk_size=500, overlap=50)

    chunks = [
        Chunk(id=c["id"], text=c["text"], metadata=c["metadata"]) for c in chunk_dicts
    ]

    # Store in memory (PostgreSQL-backed)
    await app.memory.set("document_chunks", [c.model_dump() for c in chunks])
    await app.memory.set("document_path", file_path)

    # Persist embeddings through unified vector memory
    embeddings = embed_batch([c["text"] for c in chunk_dicts])
    for chunk, embedding in zip(chunks, embeddings):
        metadata = {
            "text": chunk.text,
            "source": file_path,
            **chunk.metadata,
        }
        await app.memory.set_vector(chunk.id, embedding, metadata=metadata)

    return ChunkList(chunks=chunks, total_count=len(chunks))


# ============= PHASE 2: META-COGNITIVE QUERY UNDERSTANDING =============


@app.reasoner()
async def introspect_query_semantics(question: str) -> QuerySemantics:
    """
    Meta-reasoning: AI analyzes what KIND of question this is
    No hardcoded rules - AI decides dynamically
    """
    return await app.ai(
        system="You are a query introspection specialist. Analyze what the user is asking for.",
        user=f"""Question: "{question}"

Analyze this query's semantic structure:

1. question_type: Identify the question pattern
   - who_is: Asking about a specific person
   - what_is: Asking about a concept/thing
   - explain: Asking for explanation
   - compare: Asking to compare things
   - when: Asking about time/events
   - why: Asking for reasoning/causes
   - how: Asking for process/method

2. subject_type: What is being asked about?
   - person: A specific individual
   - concept: An idea or theory
   - event: Something that happened
   - organization: A company/institution
   - place: A location
   - process: How something works

3. critical_terms: Extract terms that MUST appear in answers
   - For "Who is Yann Klegal?": ["Yann Klegal"]
   - For "What is deep learning?": ["deep learning"]
   - These are terms where exact or near-exact match matters

4. flexibility: How flexible can matching be?
   - exact_match_required: Specific entities (people, places, orgs)
   - moderate: Concepts where synonyms ok
   - broad: General topics where semantic similarity ok
""",
        schema=QuerySemantics,
    )


@app.reasoner()
async def reason_about_precision(
    semantics: QuerySemantics, question: str
) -> PrecisionRequirements:
    """
    Meta-reasoning: AI decides what precision is needed
    Based on query semantics, not hardcoded rules
    """
    return await app.ai(
        system="You determine what match precision is needed for retrieval.",
        user=f"""Question: "{question}"

Query Analysis:
- Type: {semantics.question_type}
- Subject: {semantics.subject_type}
- Critical terms: {semantics.critical_terms}
- Current flexibility: {semantics.flexibility}

Decide the precision level needed:

1. precision_level:
   - exact: Names of specific people/organizations/places must match exactly
          Example: "Who is Yann Klegal?" needs exact name match
   - high: Specific concepts where close match needed
          Example: "What is ResNet?" needs ResNet specifically
   - moderate: Concepts where variations/synonyms acceptable
          Example: "What is AI?" could match "artificial intelligence"
   - broad: General topics where semantic similarity is fine
          Example: "Tell me about neural networks" can be broad

2. reasoning: Explain WHY this precision is needed (1-2 sentences)
   This reasoning will be passed to later stages as context.

Think carefully: If asking about a specific person by name, that name MUST match exactly.
Different people with similar names are different entities.
""",
        schema=PrecisionRequirements,
    )


@app.reasoner()
async def decompose_query(question: str) -> List[SubQuestion]:
    """Break complex query into atomic sub-questions"""
    result = await app.ai(
        system="You decompose complex questions into atomic sub-questions.",
        user=f"""Question: "{question}"

Break into 3-5 atomic sub-questions.
Each should be:
- Self-contained
- Answerable independently
- Prioritized (1=highest)
""",
        schema=SubQuestions,
    )
    return result.questions


# ============= PHASE 3: DYNAMIC STRATEGY COMPOSITION =============


@app.reasoner()
async def compose_retrieval_plan(
    precision: PrecisionRequirements, critical_terms: List[str]
) -> RetrievalPlan:
    """
    Meta-reasoning: AI dynamically composes retrieval strategy
    No hardcoded if-else logic - AI decides based on requirements
    """
    return await app.ai(
        system="You design retrieval strategies based on precision requirements.",
        user=f"""Precision Requirements:
- Level: {precision.precision_level}
- Reasoning: {precision.reasoning}
- Critical terms: {critical_terms}

Design a retrieval strategy:

1. primary_strategy: Choose the main approach
   - term_verification: When exact term matching is critical
   - semantic: When meaning matters more than exact words
   - hybrid: Combine both approaches

2. validation_required: Should retrieved chunks be validated?
   - true: For exact/high precision (verify critical terms present)
   - false: For moderate/broad precision

3. strategy_reasoning: Explain your strategy choice (1-2 sentences)
   This will guide the retrieval execution.

Example: For "Who is A" with exact precision:
- primary_strategy: "hybrid" (semantic to find candidates, then validate)
- validation_required: true (must verify "A" appears exactly)
- reasoning: "Person name requires exact match validation after semantic retrieval"
""",
        schema=RetrievalPlan,
    )


# ============= PHASE 4: ENSEMBLE RETRIEVAL =============


@app.skill()
async def lazy_semantic_retrieval(question: str, top_k: int = 10) -> List[RankedChunk]:
    """
    Semantic retrieval powered by AgentField's unified vector memory.
    """
    query_embedding = embed_text(question)
    results = await app.memory.similarity_search(query_embedding, top_k=top_k)

    ranked_chunks: List[RankedChunk] = []
    for result in results:
        metadata = result.get("metadata") or {}
        ranked_chunks.append(
            RankedChunk(
                chunk_id=result.get("key", metadata.get("chunk_id", "")),
                score=result.get("score", 0.0),
                text=metadata.get("text", ""),
            )
        )

    return ranked_chunks


@app.skill()
def keyword_retrieval(
    question: str, chunks: List[Dict], top_k: int = 10
) -> List[RankedChunk]:
    """Keyword-based retrieval (deterministic, fast)"""
    keywords = extract_keywords(question, top_n=5)

    scored_chunks = []
    for chunk in chunks:
        score = keyword_match_score(keywords, chunk["text"])
        if score > 0:
            scored_chunks.append(
                {"chunk_id": chunk["id"], "score": score, "text": chunk["text"]}
            )

    scored_chunks.sort(key=lambda x: x["score"], reverse=True)
    return [RankedChunk(**c) for c in scored_chunks[:top_k]]


@app.reasoner()
async def type_specific_retrieval(
    question: str, query_type: str, chunks: List[Dict], top_k: int = 10
) -> List[RankedChunk]:
    """Retrieval strategy based on question type"""
    # Use AI to identify key entities/concepts for this type
    result = await app.ai(
        system=f"You identify key {query_type} elements for targeted retrieval.",
        user=f"""Question: "{question}"
Type: {query_type}

Identify 3-5 key search terms specific to this question type.
For factual: entities, names, dates
For analytical: concepts, relationships
For comparative: items being compared
For temporal: time periods, sequences
""",
        schema=SearchTerms,
    )

    # Use identified terms for targeted search
    scored_chunks = []
    for chunk in chunks:
        score = (
            sum(1 for term in result.terms if term.lower() in chunk["text"].lower())
            / len(result.terms)
            if result.terms
            else 0
        )

        if score > 0:
            scored_chunks.append(
                {"chunk_id": chunk["id"], "score": score, "text": chunk["text"]}
            )

    scored_chunks.sort(key=lambda x: x["score"], reverse=True)
    return [RankedChunk(**c) for c in scored_chunks[:top_k]]


@app.reasoner()
async def ensemble_retrieval(
    question: str, query_type: str, top_k: int = 10
) -> List[RankedChunk]:
    """
    Ensemble retrieval: Run 3 strategies in parallel, merge results
    """
    chunks = await app.memory.get("document_chunks", [])

    # Run all strategies in parallel
    semantic_task = lazy_semantic_retrieval(question, top_k)
    keyword_task = asyncio.to_thread(keyword_retrieval, question, chunks, top_k)
    type_task = type_specific_retrieval(question, query_type, chunks, top_k)

    semantic_results, keyword_results, type_results = await asyncio.gather(
        semantic_task, keyword_task, type_task
    )

    # Merge with score boosting for agreement
    all_chunks = {}
    for chunk in semantic_results:
        all_chunks[chunk.chunk_id] = chunk

    for chunk in keyword_results:
        if chunk.chunk_id in all_chunks:
            # Boost if found by multiple strategies
            all_chunks[chunk.chunk_id].score = (
                (all_chunks[chunk.chunk_id].score + chunk.score) / 2 * 1.3
            )
        else:
            all_chunks[chunk.chunk_id] = chunk

    for chunk in type_results:
        if chunk.chunk_id in all_chunks:
            all_chunks[chunk.chunk_id].score = (
                (all_chunks[chunk.chunk_id].score + chunk.score) / 2 * 1.5
            )  # Highest boost for type-specific match
        else:
            all_chunks[chunk.chunk_id] = chunk

    # Sort and return top_k
    ranked = sorted(all_chunks.values(), key=lambda x: x.score, reverse=True)
    return ranked[:top_k]


# ============= PHASE 5: SELF-AWARE MATCH QUALITY ASSESSMENT =============


@app.reasoner()
async def evaluate_chunk_match_quality(
    chunk_text: str,
    question: str,
    critical_terms: List[str],
    precision_level: str,
    precision_reasoning: str,
) -> MatchQuality:
    """
    Meta-reasoning: AI evaluates WHY a chunk matched and if it's actually relevant
    Detects entity mismatches, substitutions, and semantic drift
    """
    return await app.ai(
        system="You evaluate whether a retrieved chunk truly matches the query requirements.",
        user=f"""Question: "{question}"

Critical Terms Required: {critical_terms}
Precision Level: {precision_level}
Why This Precision: {precision_reasoning}

Retrieved Chunk:
---
{chunk_text}
---

Evaluate this match:

1. is_relevant: Does this chunk actually answer the question?
   - Check if critical terms are present (especially for exact/high precision)
   - For person names: "Yann Klegal" ≠ "Yann LeCun" (different people!)
   - For concepts: Check if the right concept is discussed

2. quality_score: Rate match quality (0.0-1.0)
   - 1.0: Perfect match, all critical terms present, directly relevant
   - 0.7: Good match, mostly relevant
   - 0.4: Weak match, partial relevance
   - 0.0: No match, different entity/concept

3. mismatch_reason: If NOT relevant, explain why (or empty string if relevant)
   - "Chunk mentions 'Yann LeCun' but query asks 'Yann Klegal' - different people"
   - "Chunk discusses X but query asks about Y"
   - "Critical term 'Z' not found in chunk"

Be strict: If the query asks about a specific person/entity and the chunk mentions
a different person/entity, mark as NOT relevant even if semantically similar.
""",
        schema=MatchQuality,
    )


# ============= PHASE 6: ITERATIVE REFINEMENT =============


@app.reasoner()
async def synthesize_draft_answer(
    question: str, chunks: List[RankedChunk]
) -> DraftAnswer:
    """Generate draft answer from retrieved chunks"""
    context = "\n\n".join([f"[{c.chunk_id}] {c.text}" for c in chunks])

    return await app.ai(
        system="You synthesize precise answers from provided context only.",
        user=f"""Question: "{question}"

Context:
{context}

Provide:
1. text: Answer based ONLY on context
2. confidence: 0.0-1.0 (how well context answers question)
3. gaps: List missing information needed for complete answer
""",
        schema=DraftAnswer,
    )


@app.reasoner()
async def identify_gaps(draft: DraftAnswer) -> List[Gap]:
    """Identify specific information gaps"""
    if not draft.gaps:
        return []

    result = await app.ai(
        system="You identify and prioritize information gaps.",
        user=f"""Draft answer gaps: {draft.gaps}

For each gap, provide:
- description: What specific information is missing
- priority: 1-3 (1=critical, 3=nice-to-have)
""",
        schema=GapList,
    )
    return result.gaps


@app.reasoner()
async def generate_refinement_queries(gaps: List[Gap]) -> List[RefinementQuery]:
    """Generate targeted queries to fill gaps"""
    gap_descriptions = [g.description for g in gaps]

    result = await app.ai(
        system="You generate targeted search queries to fill information gaps.",
        user=f"""Information gaps: {gap_descriptions}

For each gap, generate a focused search query.
""",
        schema=RefinementQueryList,
    )
    return result.queries


@app.reasoner()
async def iterative_refinement(
    question: str, query_type: str, max_iterations: int = 3
) -> DraftAnswer:
    """
    Iteratively refine answer with confidence-driven routing
    """
    current_chunks = await ensemble_retrieval(question, query_type, top_k=5)
    iteration = 0
    draft = None

    while iteration < max_iterations:
        draft = await synthesize_draft_answer(question, current_chunks)

        app.note(
            f"Iteration {iteration + 1}: confidence={draft.confidence:.2f}",
            ["refinement"],
        )

        # Confidence-driven routing
        if draft.confidence > 0.85:
            app.note("High confidence - early exit", ["refinement"])
            break

        if iteration < max_iterations - 1:
            # Identify gaps
            gaps = await identify_gaps(draft)

            if gaps:
                # Generate refinement queries
                refinement_queries = await generate_refinement_queries(gaps)

                # Expand retrieval for each gap
                additional_chunks = []
                for ref_query in refinement_queries:
                    new_chunks = await ensemble_retrieval(
                        ref_query.query, query_type, top_k=3
                    )
                    additional_chunks.extend(new_chunks)

                # Merge and deduplicate
                all_chunk_dicts = [
                    {"chunk_id": c.chunk_id, "score": c.score, "text": c.text}
                    for c in current_chunks
                ]
                for chunk in additional_chunks:
                    all_chunk_dicts.append(
                        {
                            "chunk_id": chunk.chunk_id,
                            "score": chunk.score,
                            "text": chunk.text,
                        }
                    )

                unique_dicts = deduplicate_chunks(all_chunk_dicts)
                current_chunks = [RankedChunk(**c) for c in unique_dicts]

        iteration += 1

    if draft is None:
        draft = await synthesize_draft_answer(question, current_chunks)

    return draft


# ============= PHASE 7: ANSWER SYNTHESIS WITH ENTITY EXTRACTION =============


@app.reasoner()
async def synthesize_answer_with_entities(
    question: str, chunks: List[RankedChunk]
) -> DraftAnswerWithEntities:
    """Generate answer AND extract what entities it mentions"""
    context = "\n\n".join([f"[{c.chunk_id}] {c.text}" for c in chunks])

    return await app.ai(
        system="You synthesize answers from context and extract entities mentioned.",
        user=f"""Question: "{question}"

Context:
{context}

Provide:
1. answer_text: Answer based ONLY on context
2. mentioned_entities: List ALL specific entities (people, orgs, places) mentioned in your answer
   - For "Yann LeCun developed CNNs": ["Yann LeCun", "CNNs"]
   - For "Deep learning is a subset of AI": ["Deep learning", "AI"]
3. answer_confidence: 0.0-1.0 (how well context answers question)
""",
        schema=DraftAnswerWithEntities,
    )


# ============= PHASE 8: SEMANTIC DRIFT DETECTION =============


@app.reasoner()
async def detect_entity_drift(
    query_critical_terms: List[str],
    answer_mentioned_entities: List[str],
    original_question: str,
    answer_text: str,
) -> DriftAnalysis:
    """
    Meta-reasoning: Detect if answer substituted entities
    Critical for preventing "Yann Klegal" → "Yann LeCun" hallucinations
    """
    return await app.ai(
        system="You detect entity substitution and semantic drift between query and answer.",
        user=f"""Original Question: "{original_question}"
Query Critical Terms: {query_critical_terms}

Generated Answer: "{answer_text}"
Answer Mentioned Entities: {answer_mentioned_entities}

Analyze for drift:

1. has_entity_substitution: Did the answer substitute a different entity?
   - Query asks "Yann Klegal" but answer discusses "Yann LeCun" → TRUE
   - Query asks "deep learning" and answer discusses "deep learning" → FALSE
   - Look for critical terms from query that are MISSING or REPLACED in answer

2. substitution_details: Describe any substitution
   - "Query asks about 'Yann Klegal' but answer discusses 'Yann LeCun' - different people"
   - "Query term 'X' replaced with 'Y' in answer"
   - Leave empty if no substitution

3. is_answer_valid: Is this answer valid for the original question?
   - FALSE if entities were substituted
   - FALSE if answer discusses wrong topic
   - TRUE only if answer actually addresses what was asked

Think carefully: If query mentions a specific person name and answer mentions a different
person name, that's entity substitution even if the names are similar.
""",
        schema=DriftAnalysis,
    )


# ============= PHASE 9: ANSWER-QUERY ALIGNMENT VERIFICATION =============


@app.reasoner()
async def verify_answer_alignment(
    original_question: str,
    answer_text: str,
    drift_analysis: DriftAnalysis,
    precision_requirements: PrecisionRequirements,
) -> AlignmentVerdict:
    """
    Final gate: Does answer actually answer the query?
    Uses drift analysis + precision requirements to make decision
    """
    return await app.ai(
        system="You verify if an answer actually answers the original question.",
        user=f"""Original Question: "{original_question}"

Generated Answer: "{answer_text}"

Precision Required: {precision_requirements.precision_level}
Precision Reasoning: {precision_requirements.reasoning}

Drift Analysis:
- Has entity substitution: {drift_analysis.has_entity_substitution}
- Details: {drift_analysis.substitution_details}
- Is valid: {drift_analysis.is_answer_valid}

Make final verdict:

1. is_aligned: Does the answer align with what was asked?
   - FALSE if entity substitution detected
   - FALSE if answer discusses wrong topic
   - FALSE if precision requirements not met
   - TRUE only if answer truly addresses the question

2. should_return_answer: Should we return this answer to user?
   - FALSE if not aligned (return "no information found" instead)
   - TRUE if aligned and valid

3. verdict_reasoning: Explain your verdict (1-2 sentences)
   Why is this answer valid/invalid for the query?

Be strict: If precision is "exact" and there's entity substitution, answer should NOT be returned.
""",
        schema=AlignmentVerdict,
    )


# ============= PHASE 10: ADAPTIVE CONFIDENCE CALIBRATION =============


@app.reasoner()
async def calibrate_confidence(
    match_qualities: List[MatchQuality],
    drift_analysis: DriftAnalysis,
    alignment_verdict: AlignmentVerdict,
    precision_requirements: PrecisionRequirements,
) -> FinalConfidence:
    """
    Meta-reasoning: AI computes final confidence considering all quality signals
    No fixed thresholds - AI reasons about confidence
    """
    # Summary stats for context
    avg_quality = (
        sum(q.quality_score for q in match_qualities) / len(match_qualities)
        if match_qualities
        else 0.0
    )
    relevant_count = sum(1 for q in match_qualities if q.is_relevant)

    return await app.ai(
        system="You calibrate confidence based on all quality signals.",
        user=f"""Precision Requirements:
- Level: {precision_requirements.precision_level}
- Reasoning: {precision_requirements.reasoning}

Match Quality Summary:
- Relevant chunks: {relevant_count}/{len(match_qualities)}
- Average quality score: {avg_quality:.2f}

Drift Analysis:
- Entity substitution: {drift_analysis.has_entity_substitution}
- Valid answer: {drift_analysis.is_answer_valid}

Alignment Verdict:
- Is aligned: {alignment_verdict.is_aligned}
- Should return: {alignment_verdict.should_return_answer}
- Reasoning: {alignment_verdict.verdict_reasoning}

Calculate final confidence:

1. confidence_score: Overall confidence (0.0-1.0)
   - Consider match quality
   - Heavy penalty if entity substitution detected
   - Heavy penalty if alignment failed
   - High confidence (0.9+) for valid negative answers ("no info found")

2. confidence_reasoning: Explain the confidence (2-3 sentences)
   - Why this confidence level?
   - What signals contributed?
   - Any concerns?

Note: High confidence in "no information found" is GOOD - it means we're certain
we don't have info about the queried entity.
""",
        schema=FinalConfidence,
    )


# ============= PHASE 11: CLAIM VERIFICATION (LEGACY - kept for completeness) =============


@app.reasoner()
async def decompose_into_claims(answer_text: str) -> List[Claim]:
    """Break answer into atomic verifiable claims"""
    result = await app.ai(
        system="You extract atomic factual claims from text.",
        user=f"""Text: "{answer_text}"

Extract individual claims. Each should be:
- Atomic (one fact)
- Verifiable
- Self-contained

Classify as: fact, inference, or opinion
""",
        schema=ClaimList,
    )
    return result.claims


@app.reasoner()
async def verify_claim(claim: Claim) -> VerificationResult:
    """Verify single claim against source chunks"""
    chunks = await app.memory.get("document_chunks", [])

    # Find supporting chunks
    supporting_chunks = []
    for chunk in chunks:
        if any(
            word in chunk["text"].lower() for word in claim.text.lower().split()[:5]
        ):
            supporting_chunks.append(chunk)

    if not supporting_chunks:
        return VerificationResult(
            claim_id=claim.id, is_verified=False, confidence=0.0, quote_ids=[]
        )

    # Extract exact quotes
    quote_chunks = []
    for chunk in supporting_chunks[:5]:
        quote = find_quote_in_chunk(claim.text, chunk["text"])
        if quote:
            quote_chunks.append((chunk["id"], quote))

    if not quote_chunks:
        return VerificationResult(
            claim_id=claim.id, is_verified=False, confidence=0.0, quote_ids=[]
        )

    # Verify with AI
    context = "\n\n".join([f"[{chunk_id}] {quote}" for chunk_id, quote in quote_chunks])

    result = await app.ai(
        system="You verify claims against source quotes. Be strict.",
        user=f"""Claim: "{claim.text}"

Source quotes:
{context}

Is this claim directly supported?
""",
        schema=VerificationResult,
    )

    result.claim_id = claim.id
    return result


# ============= PHASE 6: FINAL SYNTHESIS =============


@app.reasoner()
async def build_verified_answer(
    draft: DraftAnswer, verifications: List[VerificationResult]
) -> VerifiedAnswer:
    """Rebuild answer using only verified claims"""
    chunks = await app.memory.get("document_chunks", [])
    chunk_map = {c["id"]: c for c in chunks}

    # Filter claims
    verified = [v for v in verifications if v.is_verified and v.confidence > 0.7]
    uncertain = [v for v in verifications if 0.4 < v.confidence <= 0.7]
    removed = [v for v in verifications if v.confidence <= 0.4]

    # Build citations
    citations = []
    for v in verified:
        for chunk_id in v.quote_ids:
            if chunk_id in chunk_map:
                chunk = chunk_map[chunk_id]
                quote = chunk["text"][:200] + "..."

                citations.append(
                    Citation(
                        chunk_id=chunk_id,
                        quote=quote,
                        page_num=chunk["metadata"].get("index"),
                    )
                )

    # Calculate confidence
    avg_confidence = (
        sum(v.confidence for v in verified) / len(verifications)
        if verifications
        else 0.0
    )

    return VerifiedAnswer(
        answer=draft.text,
        citations=citations,
        confidence_score=avg_confidence,
        verification_summary={
            "verified": len(verified),
            "uncertain": len(uncertain),
            "removed": len(removed),
        },
        completeness_score=draft.confidence,
        gaps=draft.gaps,
    )


# ============= PHASE 7: QUALITY CHECK =============


@app.reasoner()
async def completeness_check(answer: VerifiedAnswer, original_question: str) -> float:
    """Check if answer fully addresses the question"""
    result = await app.ai(
        system="You assess answer completeness.",
        user=f"""Question: "{original_question}"

Answer: "{answer.answer}"

Gaps: {answer.gaps}

Rate completeness (0.0-1.0):
- 1.0: Fully answers question
- 0.7: Mostly complete, minor gaps
- 0.4: Partial answer
- 0.0: Doesn't answer question
""",
        schema=CompletenessScore,
    )
    return result.score


# ============= MAIN ENTRY POINT: META-COGNITIVE ORCHESTRATOR =============


@app.reasoner()
async def query_document(file_path: str, question: str) -> VerifiedAnswer:
    """
    Meta-cognitive orchestrator - Composes micro-reasoners dynamically
    Each reasoner has ONE job with 2-4 field schemas
    Intelligence emerges from clever composition
    """
    try:
        # ===== PHASE 1: DOCUMENT CHUNKING =====
        app.note("Phase 1: Chunking document", ["pipeline"])
        chunk_list = await chunk_document(file_path)
        app.note(f"Created {chunk_list.total_count} chunks", ["chunking"])

        chunks = await app.memory.get("document_chunks", [])

        # ===== PHASE 2: META-REASONING (Query Introspection) =====
        app.note("Phase 2: Meta-reasoning - Query introspection", ["pipeline"])

        # Step 1: AI analyzes query semantics
        semantics = await introspect_query_semantics(question)
        app.note(
            f"Query type: {semantics.question_type}, "
            f"Subject: {semantics.subject_type}, "
            f"Critical terms: {semantics.critical_terms}",
            ["introspection"],
        )

        # Step 2: AI reasons about precision needed
        precision = await reason_about_precision(semantics, question)
        app.note(
            f"Precision: {precision.precision_level} - {precision.reasoning}",
            ["precision"],
        )

        # ===== PHASE 3: DYNAMIC STRATEGY COMPOSITION =====
        app.note("Phase 3: Composing retrieval strategy", ["pipeline"])

        # AI dynamically composes retrieval plan
        plan = await compose_retrieval_plan(precision, semantics.critical_terms)
        app.note(
            f"Strategy: {plan.primary_strategy}, "
            f"Validation required: {plan.validation_required}",
            ["strategy"],
        )

        # ===== PHASE 4: ENSEMBLE RETRIEVAL =====
        app.note("Phase 4: Ensemble retrieval", ["pipeline"])

        # Execute ensemble retrieval (existing code - works well)
        # Use factual as default query type for ensemble
        candidate_chunks = await ensemble_retrieval(
            question,
            (
                semantics.question_type
                if semantics.question_type
                in ["factual", "analytical", "comparative", "temporal"]
                else "factual"
            ),
            top_k=20,  # Get more candidates for quality assessment
        )
        app.note(f"Retrieved {len(candidate_chunks)} candidate chunks", ["retrieval"])

        # ===== PHASE 5: SELF-AWARE MATCH QUALITY ASSESSMENT (PARALLEL) =====
        app.note("Phase 5: Evaluating match quality (parallel)", ["pipeline"])

        # Run quality assessment in parallel for all candidates
        quality_tasks = [
            evaluate_chunk_match_quality(
                chunk.text,
                question,
                semantics.critical_terms,
                precision.precision_level,
                precision.reasoning,
            )
            for chunk in candidate_chunks
        ]
        match_qualities = await asyncio.gather(*quality_tasks)

        # Filter to only relevant chunks
        high_quality_chunks = [
            chunk
            for chunk, quality in zip(candidate_chunks, match_qualities)
            if quality.is_relevant and quality.quality_score > 0.5
        ]

        relevant_count = len(high_quality_chunks)
        app.note(
            f"Quality filtering: {relevant_count}/{len(candidate_chunks)} chunks relevant",
            ["quality"],
        )

        # Log mismatch reasons for debugging
        for chunk, quality in zip(candidate_chunks, match_qualities):
            if not quality.is_relevant and quality.mismatch_reason:
                app.note(
                    f"Filtered: {quality.mismatch_reason}",
                    ["quality", "debug"],
                )

        # ===== PHASE 6: CONDITIONAL - Check if we have relevant chunks =====
        if not high_quality_chunks:
            app.note(
                "No relevant chunks found - returning negative answer", ["pipeline"]
            )

            # Return high-confidence negative answer
            return VerifiedAnswer(
                answer=f"No information found in the document about {', '.join(semantics.critical_terms)}.",
                citations=[],
                confidence_score=0.95,  # High confidence in negative answer
                verification_summary={"verified": 0, "uncertain": 0, "removed": 0},
                completeness_score=1.0,  # Complete negative answer
                gaps=[],
            )

        # ===== PHASE 7: ANSWER SYNTHESIS WITH ENTITY EXTRACTION =====
        app.note("Phase 7: Synthesizing answer with entity extraction", ["pipeline"])

        draft_with_entities = await synthesize_answer_with_entities(
            question, high_quality_chunks
        )
        app.note(
            f"Draft confidence: {draft_with_entities.answer_confidence:.2f}, "
            f"Entities: {draft_with_entities.mentioned_entities}",
            ["synthesis"],
        )

        # ===== PHASE 8: SEMANTIC DRIFT DETECTION =====
        app.note("Phase 8: Detecting semantic drift", ["pipeline"])

        drift_analysis = await detect_entity_drift(
            semantics.critical_terms,
            draft_with_entities.mentioned_entities,
            question,
            draft_with_entities.answer_text,
        )
        app.note(
            f"Drift check: substitution={drift_analysis.has_entity_substitution}, "
            f"valid={drift_analysis.is_answer_valid}",
            ["drift"],
        )

        if drift_analysis.has_entity_substitution:
            app.note(
                f"DRIFT DETECTED: {drift_analysis.substitution_details}",
                ["drift", "warning"],
            )

        # ===== PHASE 9: ANSWER-QUERY ALIGNMENT VERIFICATION =====
        app.note("Phase 9: Verifying answer alignment", ["pipeline"])

        alignment = await verify_answer_alignment(
            question,
            draft_with_entities.answer_text,
            drift_analysis,
            precision,
        )
        app.note(
            f"Alignment: {alignment.is_aligned}, "
            f"Should return: {alignment.should_return_answer}",
            ["alignment"],
        )
        app.note(f"Verdict: {alignment.verdict_reasoning}", ["alignment"])

        # ===== PHASE 10: CONDITIONAL - Should we return this answer? =====
        if not alignment.should_return_answer:
            app.note(
                "Answer failed alignment check - returning negative answer",
                ["pipeline"],
            )

            # Return high-confidence negative answer
            return VerifiedAnswer(
                answer=f"No information found in the document about {', '.join(semantics.critical_terms)}.",
                citations=[],
                confidence_score=0.95,  # High confidence in negative answer
                verification_summary={"verified": 0, "uncertain": 0, "removed": 0},
                completeness_score=1.0,
                gaps=[alignment.verdict_reasoning],  # Explain why we couldn't answer
            )

        # ===== PHASE 11: ADAPTIVE CONFIDENCE CALIBRATION =====
        app.note("Phase 11: Calibrating confidence", ["pipeline"])

        final_confidence = await calibrate_confidence(
            match_qualities,
            drift_analysis,
            alignment,
            precision,
        )
        app.note(
            f"Final confidence: {final_confidence.confidence_score:.2f}",
            ["confidence"],
        )
        app.note(f"Reasoning: {final_confidence.confidence_reasoning}", ["confidence"])

        # ===== PHASE 12: BUILD FINAL ANSWER =====
        app.note("Phase 12: Building final answer with citations", ["pipeline"])

        # Build citations from high-quality chunks
        chunk_ids_used = [c.chunk_id for c in high_quality_chunks[:5]]  # Top 5
        all_chunks_map = {c["id"]: c for c in chunks}

        citations = []
        for chunk_id in chunk_ids_used:
            if chunk_id in all_chunks_map:
                chunk_data = all_chunks_map[chunk_id]
                citations.append(
                    Citation(
                        chunk_id=chunk_id,
                        quote=chunk_data["text"][:200] + "...",
                        page_num=chunk_data["metadata"].get("index"),
                    )
                )

        final_answer = VerifiedAnswer(
            answer=draft_with_entities.answer_text,
            citations=citations,
            confidence_score=final_confidence.confidence_score,
            verification_summary={
                "verified": relevant_count,
                "uncertain": 0,
                "removed": len(candidate_chunks) - relevant_count,
            },
            completeness_score=draft_with_entities.answer_confidence,
            gaps=[],
        )

        app.note(
            f"✅ Complete! Confidence: {final_confidence.confidence_score:.2f}, "
            f"Relevant chunks: {relevant_count}",
            ["complete"],
        )

        return final_answer

    finally:
        # CRITICAL: Memory cleanup
        app.note("Cleaning up memory", ["cleanup"])
        await app.memory.delete("document_chunks")
        await app.memory.delete("document_path")


if __name__ == "__main__":
    print("🧠 Meta-Cognitive Agentic RAG - Self-Reasoning Multi-Agent System")
    print("📍 Node: agentic-rag")
    print("🌐 Control Plane: http://localhost:8080")
    print("\n🎯 Philosophy: Composed 0.3 LLMs → 0.8 System Capability")
    print("\n12-Phase Meta-Cognitive Pipeline:")
    print("  1️⃣  Query introspection (AI analyzes semantics)")
    print("  2️⃣  Precision reasoning (AI decides requirements)")
    print("  3️⃣  Strategy composition (AI designs retrieval)")
    print("  4️⃣  Ensemble retrieval (3 parallel strategies)")
    print("  5️⃣  Self-aware match quality (AI evaluates chunks)")
    print("  6️⃣  Answer synthesis with entity extraction")
    print("  7️⃣  Semantic drift detection (catches substitutions)")
    print("  8️⃣  Answer-query alignment (final verification)")
    print("  9️⃣  Adaptive confidence calibration")
    print("\n✨ Key Features:")
    print("  ✅ No hardcoded rules - AI reasons dynamically")
    print("  ✅ Simple schemas (2-4 fields) per reasoner")
    print("  ✅ Parallel quality assessment (20+ chunks)")
    print("  ✅ Entity mismatch detection")
    print("  ✅ High-confidence negative answers (0.95)")
    print("  ✅ FastEmbed lazy semantic search")
    print("  ✅ PostgreSQL-backed memory")
    print("\n🚫 Hallucination Prevention:")
    print("  → 'Yann Klegal' ≠ 'Yann LeCun' (different people)")
    print("  → Returns 'No information found' when appropriate")
    print("  → AI self-validates at multiple layers")

    app.run(auto_port=True)
