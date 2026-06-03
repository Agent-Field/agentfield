"""QA orchestration reasoners."""

from __future__ import annotations

from typing import List

from agentfield import AgentRouter
from agentfield.logger import log_info

from pipeline_utils import (
    aggregate_chunks_to_documents,
    build_citations,
    build_citations_from_documents,
    deduplicate_results,
    format_context_for_synthesis,
    format_documents_for_synthesis,
)
from product_context import PRODUCT_CONTEXT
from routers.query_planning import plan_queries
from routers.retrieval import parallel_retrieve
from schemas import DocAnswer, RetrievalResult

qa_router = AgentRouter(tags=["qa"])


@qa_router.reasoner()
async def synthesize_answer(
    question: str,
    results: List[RetrievalResult],
    is_refinement: bool = False,
) -> DocAnswer:
    """Generate answer with self-assessment of completeness."""

    if not results:
        return DocAnswer(
            answer="I could not find any relevant documentation to answer this question.",
            citations=[],
            confidence="insufficient",
            needs_more=False,
            missing_topics=["No documentation found for this topic"],
        )

    context_text = format_context_for_synthesis(results)
    citations = build_citations(results)

    # Build citation key reference for the prompt
    citation_keys = [c.key for c in citations]
    key_map = "\n".join(
        [f"  {c.key}: {c.relative_path}:{c.start_line}-{c.end_line}" for c in citations]
    )

    system_prompt = f"""You are a knowledgeable documentation assistant helping users understand and use this product effectively. Your goal is to provide accurate, helpful answers that empower users to accomplish their tasks.

## PRODUCT CONTEXT

{PRODUCT_CONTEXT}

Use this context to understand the product's architecture, terminology, and common use cases. This helps you provide more accurate answers and explain technical concepts correctly.

## CITATION FORMAT (CRITICAL)

You have access to these source keys: {', '.join(citation_keys)}

**How to cite in your answer text:**
- Write the citation key wrapped in square brackets: [A], [B], [C], etc.
- Place citations immediately after the relevant claim
- You can combine multiple citations: [A][B] or [A][B][C]

**Example of correct inline citations:**
"Agentfield uses DIDs for identity [A]. The control plane manages orchestration [B]. You can deploy agents independently [A][B]."

**IMPORTANT:** Leave the `citations` field empty in your response (return `[]`). The system will inject citation metadata automatically. You only need to use [A], [B], etc. in the answer text.

## Core Principles

**Accuracy & Trust:**
- Base every statement on the provided documentation
- Cite sources using inline references like [A] or [B][C] after each factual claim
- If information isn't in the docs, clearly state: 'The documentation doesn't cover this yet'
- Never invent API names, commands, configuration values, or examples

**Clarity & Usefulness:**
- Start with a direct answer to the user's question
- Provide specific, actionable information (actual commands, file paths, step-by-step instructions)
- Use code blocks for commands, configuration, and code examples
- Structure complex answers with headings, bullets, or numbered steps
- Adapt your detail level to the question's complexity

**Tone & Style:**
- Be professional yet approachable—like a helpful colleague
- Use clear, concise language without unnecessary jargon
- When technical terms are needed, briefly explain them
- Be encouraging and supportive, especially for setup/troubleshooting questions

## Answer Format

**Structure your response as:**
1. **Direct answer** - Address the question immediately
2. **Key details** - Provide specific information, commands, or steps
3. **Context** (if helpful) - Add relevant background or related information
4. **Next steps** (if applicable) - Guide users on what to do next

**Formatting guidelines:**
- Use GitHub-flavored Markdown
- Format code with backticks: `inline code` or ```language blocks```
- Use bullets for lists, numbers for sequential steps
- Keep paragraphs focused (2-4 sentences each)
- Add inline citations [A], [B], etc. after each factual claim

## Self-Assessment

After generating your answer, honestly evaluate its completeness:

**Set `confidence='high'` and `needs_more=False` when:**
- You found specific, detailed information that fully answers the question
- All key aspects of the question are addressed with concrete details
- The user can take action based on your answer

**Set `confidence='partial'` and `needs_more=True` when:**
- You found some relevant information but it's incomplete
- Key details are missing (e.g., has steps 1-2 but not step 3)
- Specify exactly what's missing in `missing_topics` (e.g., ['configuration options', 'error handling'])

**Set `confidence='insufficient'` and `needs_more=True` when:**
- After thoroughly reading all documentation, the requested information isn't present
- The question asks about features/topics not covered in the docs
- Specify what information would be needed in `missing_topics`

{"**Refinement Mode:** This is a second retrieval attempt. If you have useful information—even if not complete—provide it and set `needs_more=False` to avoid retrieval loops." if is_refinement else ""}"""

    user_prompt = f"""Question: {question}

## Available Sources (use these keys for citations)

{key_map}

## Context Chunks

{context_text}

---

Generate a concise markdown answer with inline citations [A], [B], etc. after each factual claim.
Leave the `citations` array empty in your response - the system will inject citation metadata automatically.
Then self-assess and set confidence, needs_more, and missing_topics accordingly."""

    response = await qa_router.ai(
        system=system_prompt,
        user=user_prompt,
        schema=DocAnswer,
    )

    if isinstance(response, DocAnswer):
        if not response.citations:
            response.citations = citations
        return response

    response_dict = response if isinstance(response, dict) else response.model_dump()
    response_dict["citations"] = citations
    return DocAnswer.model_validate(response_dict)


@qa_router.reasoner()
async def qa_answer(
    question: str,
    namespace: str = "website-docs",
    top_k: int = 6,
    min_score: float = 0.35,
) -> DocAnswer:
    """
    Main QA orchestrator with parallel retrieval and optional refinement.
    """

    log_info(f"[qa_answer] Processing question: {question}")

    plan = await plan_queries(question)
    log_info(
        f"[qa_answer] Generated {len(plan.queries)} queries with strategy: {plan.strategy}"
    )

    results = await parallel_retrieve(
        queries=plan.queries,
        namespace=namespace,
        top_k=top_k,
        min_score=min_score,
    )

    answer = await synthesize_answer(question, results, is_refinement=False)

    log_info(
        f"[qa_answer] First synthesis: confidence={answer.confidence}, "
        f"needs_more={answer.needs_more}, citations={len(answer.citations)}"
    )

    if answer.needs_more and answer.missing_topics:
        log_info(f"[qa_answer] Refinement needed for: {answer.missing_topics}")

        refinement_queries = []
        for topic in answer.missing_topics[:3]:
            refinement_queries.append(f"{question} {topic}")
            refinement_queries.append(topic)

        additional_results = await parallel_retrieve(
            queries=refinement_queries,
            namespace=namespace,
            top_k=top_k,
            min_score=min_score,
        )

        all_results = results + additional_results
        merged_results = deduplicate_results(all_results)

        log_info(
            f"[qa_answer] Refinement retrieved {len(additional_results)} new chunks, "
            f"merged to {len(merged_results)} total"
        )

        answer = await synthesize_answer(question, merged_results, is_refinement=True)

        log_info(
            f"[qa_answer] Refined synthesis: confidence={answer.confidence}, "
            f"needs_more={answer.needs_more}, citations={len(answer.citations)}"
        )

    return answer


@qa_router.reasoner()
async def qa_answer_with_documents(
    question: str,
    namespace: str = "website-docs",
    top_k: int = 6,
    min_score: float = 0.35,
    top_documents: int = 5,
) -> DocAnswer:
    """
    Document-aware QA orchestrator that retrieves full documents instead of chunks.
    """

    log_info(f"[qa_answer_with_documents] Processing question: {question}")

    plan = await plan_queries(question)
    log_info(
        f"[qa_answer_with_documents] Generated {len(plan.queries)} queries with strategy: {plan.strategy}"
    )

    chunk_results = await parallel_retrieve(
        queries=plan.queries,
        namespace=namespace,
        top_k=top_k,
        min_score=min_score,
    )

    global_memory = qa_router.memory.global_scope
    documents = await aggregate_chunks_to_documents(
        global_memory, chunk_results, top_n=top_documents
    )

    if not documents:
        return DocAnswer(
            answer="I could not find any relevant documentation to answer this question.",
            citations=[],
            confidence="insufficient",
            needs_more=False,
            missing_topics=["No documentation found for this topic"],
        )

    context_text = format_documents_for_synthesis(documents)
    citations = build_citations_from_documents(documents)

    # Build citation key reference for the prompt
    citation_keys = [c.key for c in citations]
    key_map = "\n".join([f"  {c.key}: {c.relative_path}" for c in citations])

    system_prompt = f"""You are a knowledgeable documentation assistant helping users understand and use this product effectively. Your goal is to provide accurate, helpful answers by thoroughly reading and comprehending the full documentation pages provided.

## PRODUCT CONTEXT

{PRODUCT_CONTEXT}

Use this context to understand the product's architecture, terminology, and common use cases. This helps you provide more accurate answers and explain technical concepts correctly. For example, when users ask about 'identity', you know they're asking about DIDs and VCs. When they ask about 'functions', you understand they might mean reasoners or skills.

## CITATION FORMAT (CRITICAL)

You have access to these source keys: {', '.join(citation_keys)}

**How to cite in your answer text:**
- Write the citation key wrapped in square brackets: [A], [B], [C], etc.
- Place citations immediately after the relevant claim
- You can combine multiple citations: [A][B] or [A][B][C]

**Example of correct inline citations:**
"To get started, run `af init my-project` [A]. The control plane handles orchestration automatically [B]. You can deploy agents independently [A][B]."

**IMPORTANT:** Leave the `citations` field empty in your response (return `[]`). The system will inject citation metadata automatically. You only need to use [A], [B], etc. in the answer text.

## Core Principles

**Accuracy & Trust:**
- Base every statement on the provided documentation pages
- Cite sources using inline references like [A] or [B][C] after each factual claim
- If information isn't in the docs, clearly state: 'The documentation doesn't cover this yet'
- Never invent API names, commands, configuration values, or examples

**Clarity & Usefulness:**
- Start with a direct answer to the user's question
- Extract and present SPECIFIC details from the documentation: actual commands, file paths, configuration values, step-by-step instructions
- Use code blocks for commands, configuration, and code examples
- Structure complex answers with headings, bullets, or numbered steps
- Be concrete and actionable—give users what they need to accomplish their task

**Tone & Style:**
- Be professional yet approachable—like a helpful colleague
- Use clear, concise language without unnecessary jargon
- When technical terms are needed, briefly explain them
- Be encouraging and supportive, especially for setup/troubleshooting questions

## Reading Instructions

**How to use the documentation:**
1. Read the full documentation pages carefully and thoroughly
2. Find the specific information that directly answers the user's question
3. Extract and present the actual details, steps, commands, or explanations
4. Quote or paraphrase directly from the documentation—be specific
5. If the answer requires multiple steps or details, extract ALL of them

**Important:** Don't just say 'the documentation mentions X'—tell users exactly what it says. Don't be vague or generic—extract specific information. You are reading the documentation FOR the user.

## Answer Format

**Structure your response as:**
1. **Direct answer** - Address the question immediately with specific details
2. **Key details** - Provide actual commands, file paths, configuration values, or step-by-step instructions
3. **Context** (if helpful) - Add relevant background or related information
4. **Next steps** (if applicable) - Guide users on what to do next

**Formatting guidelines:**
- Use GitHub-flavored Markdown
- Format code with backticks: `inline code` or ```language blocks```
- Use bullets for lists, numbers for sequential steps
- Keep paragraphs focused (2-4 sentences each)
- Add inline citations [A], [B], etc. after each factual claim

## Examples

**Question:** 'How do I get started?'
**Good Answer:**
To get started with AgentField:

1. Install the CLI: `npm install -g agentfield` [A]
2. Initialize a new project: `af init my-project` [A]
3. Configure your agent in the generated `agent.yaml` file [A]

The initialization creates a basic project structure with example agents you can customize [A].

**Question:** 'How is IAM treated?'
**Good Answer:**
AgentField uses Decentralized Identifiers (DIDs) for identity management [A]. Each agent receives a unique, cryptographically verifiable DID when registered [A]. You can configure IAM policies in the control plane settings under `config/agentfield.yaml` in the `security` section [B].

## Self-Assessment

After generating your answer, honestly evaluate its completeness:

**Set `confidence='high'` and `needs_more=False` when:**
- You found specific, detailed information that fully answers the question
- All key aspects are addressed with concrete details from the documentation
- The user can take action based on your answer
- Note: If the answer requires combining info from multiple paragraphs or sections, that's still a complete answer

**Set `confidence='partial'` and `needs_more=True` when:**
- You found some relevant information but it's incomplete
- Key details are missing (e.g., has steps 1-2 but not step 3)
- Specify exactly what's missing in `missing_topics` (e.g., ['configuration options', 'error handling'])

**Set `confidence='insufficient'` and `needs_more=True` when:**
- After thoroughly reading all documentation pages, the requested information isn't present
- The question asks about features/topics not covered in the docs
- Specify what information would be needed in `missing_topics`"""

    user_prompt = f"""Question: {question}

## Available Sources (use these keys for citations)

{key_map}

## Full Documentation Pages

{context_text}

---

Generate a concise markdown answer with inline citations [A], [B], etc. after each factual claim.
Leave the `citations` array empty in your response - the system will inject citation metadata automatically.
Then self-assess and set confidence, needs_more, and missing_topics accordingly."""

    response = await qa_router.ai(
        system=system_prompt,
        user=user_prompt,
        schema=DocAnswer,
    )

    if isinstance(response, DocAnswer):
        if not response.citations:
            response.citations = citations
        answer = response
    else:
        response_dict = (
            response if isinstance(response, dict) else response.model_dump()
        )
        response_dict["citations"] = citations
        answer = DocAnswer.model_validate(response_dict)

    log_info(
        f"[qa_answer_with_documents] First synthesis: confidence={answer.confidence}, "
        f"needs_more={answer.needs_more}, documents_used={len(documents)}"
    )

    if answer.needs_more and answer.missing_topics:
        log_info(
            f"[qa_answer_with_documents] Refinement needed for: {answer.missing_topics}"
        )

        refinement_queries = []
        for topic in answer.missing_topics[:3]:
            refinement_queries.append(f"{question} {topic}")
            refinement_queries.append(topic)

        additional_chunks = await parallel_retrieve(
            queries=refinement_queries,
            namespace=namespace,
            top_k=top_k,
            min_score=min_score,
        )

        all_chunks = chunk_results + additional_chunks
        merged_documents = await aggregate_chunks_to_documents(
            global_memory, all_chunks, top_n=top_documents
        )

        log_info(
            f"[qa_answer_with_documents] Refinement found {len(merged_documents)} total documents"
        )

        context_text = format_documents_for_synthesis(merged_documents)
        citations = build_citations_from_documents(merged_documents)
        citation_keys = [c.key for c in citations]
        key_map = "\n".join([f"  {c.key}: {c.relative_path}" for c in citations])

        system_prompt_refined = (
            system_prompt
            + "\n\n**REFINEMENT MODE:** This is a second retrieval attempt. If you have useful information—even if not complete—provide it and set `needs_more=False` to avoid retrieval loops."
        )

        user_prompt_refined = f"""Question: {question}

## Available Sources (use these keys for citations)

{key_map}

## Full Documentation Pages

{context_text}

---

Generate a concise markdown answer with inline citations [A], [B], etc. after each factual claim.
Leave the `citations` array empty in your response - the system will inject citation metadata automatically.
Then self-assess and set confidence, needs_more, and missing_topics accordingly."""

        response = await qa_router.ai(
            system=system_prompt_refined,
            user=user_prompt_refined,
            schema=DocAnswer,
        )

        if isinstance(response, DocAnswer):
            if not response.citations:
                response.citations = citations
            answer = response
        else:
            response_dict = (
                response if isinstance(response, dict) else response.model_dump()
            )
            response_dict["citations"] = citations
            answer = DocAnswer.model_validate(response_dict)

        log_info(
            f"[qa_answer_with_documents] Refined synthesis: confidence={answer.confidence}, "
            f"needs_more={answer.needs_more}, documents_used={len(merged_documents)}"
        )

    return answer
