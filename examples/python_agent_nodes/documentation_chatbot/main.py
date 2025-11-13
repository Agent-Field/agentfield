"""Simplified Documentation chatbot with parallel retrieval and self-aware synthesis."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Sequence

from agentfield import AIConfig, Agent
from agentfield.logger import log_info

if __package__ in (None, ""):
    current_dir = Path(__file__).resolve().parent
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))

from chunking import chunk_markdown_text, is_supported_file, read_text
from embedding import embed_query, embed_texts
from schemas import (
    Citation,
    DocAnswer,
    DocumentChunk,
    DocumentContext,
    IngestReport,
    QueryPlan,
    RetrievalResult,
)

# ========================= Product Context (Customizable) =========================
# This section provides domain-specific context to improve search and answer quality.
# Replace this with your own product information when adapting this chatbot.

PRODUCT_CONTEXT = """
## Product Overview
AgentField is a Kubernetes-style control plane with IAM for building next generation of autonomous software. It provides production infrastructure
for deploying, orchestrating, and observing multi-agent systems with cryptographic identity and audit trails.

**Architecture**: Distributed control plane + independent agent nodes. Think "Kubernetes for AI agents."

**Positioning**: AgentField is infrastructure, not an application framework. While agent frameworks help you build
single AI applications, AgentField provides the orchestration layer for
deploying and managing distributed multi-agent systems in production (like Kubernetes orchestrates containers).

## Design Philosophy

**Infrastructure-First Approach:**
- Control plane handles routing, identity, memory, and observability centrally
- Agents run as independent microservices, not embedded libraries
- Teams deploy agents independently without coordination
- Every agent function becomes a REST API automatically
- Stateless control plane enables horizontal scaling

**Production-Grade Guarantees:**
- Cryptographic identity for every agent and execution (W3C DID standard)
- Tamper-proof audit trails via Verifiable Credentials
- Zero-config distributed state management
- No timeout limits for async workflows (hours or days)
- Observable by default (workflow DAGs, execution traces, agent notes)

**Built for Multi-Team Scale:**
- Independent agent deployment (no monolithic coordination)
- Service discovery through control plane
- Shared memory fabric across distributed agents
- Cross-agent communication via REST APIs
- Works with any tech stack (Python, Go, React, mobile, .NET, etc.)

## Core Concepts & Terminology

**Agent Primitives:**
- **Reasoners**: AI-guided decision making functions (use LLMs for judgment)
- **Skills**: Deterministic functions (reliable execution, no AI)
- **Agent Nodes**: Independent services that register with the control plane
- **Control Plane**: Central orchestration server (handles routing, memory, identity)

**Identity & Trust:**
- **DIDs** (Decentralized Identifiers): Cryptographic identity for agents (W3C standard)
- **VCs** (Verifiable Credentials): Tamper-proof execution records (W3C standard)
- **Workflow DAGs**: Visual representation of agent execution chains
- **Security Model**: Every execution cryptographically signed and attributable
- **Audit Compliance**: Exportable proof chains for regulatory/compliance requirements
- **Zero-Trust Architecture**: Agents authenticate via DIDs, not shared secrets

**State Management:**
- **Memory Scopes**: Hierarchical state sharing (global, actor, session, workflow)
- **Zero-config memory**: Automatic state synchronization across distributed agents
- **Memory events**: Real-time reactive patterns (on_change listeners)

**Execution Patterns:**
- **Sync execution**: `/api/v1/execute/` (90 second timeout)
- **Async execution**: `/api/v1/execute/async/` (no timeout limits, hours/days)
- **Webhooks**: Callback URLs for async results
- **Cross-agent calls**: `app.call("agent.function")` for agent-to-agent communication

**Scalability & Production Architecture:**
- **Stateless Control Plane**: No session affinity, horizontal scaling to billions of requests
- **Independent Agent Scaling**: Each agent scales independently based on its load
- **Zero Coordination Overhead**: Agents don't need to know about each other to deploy
- **Deployment Flexibility**: Laptop → Docker → Kubernetes with same codebase, zero rewrites
- **Storage Tiers**: Local (SQLite/BoltDB) for dev, PostgreSQL for production/cloud
- **Failure Isolation**: Agent failures don't cascade; control plane handles routing around issues

**CLI Commands:**
- `af init`: Create new agent (Python or Go)
- `af server`: Start control plane
- `af run`: Run agent locally
- `af dev`: Development mode with hot reload

**Key APIs:**
- `app.ai()`: LLM calls with structured output (Pydantic schemas)
- `app.memory`: State management (get/set/on_change)
- `app.call()`: Cross-agent communication
- `app.note()`: Observable execution notes

## Common Topics & Questions

**Getting Started:**
- Installation and setup (af init, af server)
- Creating first agent (Python vs Go choice)
- Understanding reasoners vs skills
- Basic agent structure and configuration

**Agent Development:**
- Registering reasoners and skills
- Using app.ai() for LLM integration
- Structured output with Pydantic/Go structs
- Router pattern for organizing code
- Agent notes and observability

**Multi-Agent Coordination:**
- Cross-agent communication patterns
- Shared memory and state management
- Memory scopes (when to use which)
- Event-driven workflows with memory.on_change

**Production Deployment:**
- Local development (embedded SQLite/BoltDB)
- Docker deployment
- Kubernetes deployment
- Environment variables and configuration

**Identity & Security:**
- DID generation and management
- Verifiable Credentials for audit trails
- Cryptographic proof of execution

**Advanced Features:**
- Async execution for long-running tasks
- Webhook integration
- Custom memory providers
- Performance optimization
- Testing strategies

## Documentation Structure

The documentation is organized by:
- **Getting Started**: Quick start, installation, first agent
- **Core Concepts**: Reasoners, skills, memory, identity, cross-agent communication
- **Guides**: Deployment, testing, multi-agent patterns, examples
- **API Reference**: Python SDK, Go SDK, CLI commands, REST APIs
- **Examples**: Customer support, research assistant, terminal assistant

## Search Term Relationships

When users ask about:
- "Identity" or "authentication" or "security" → Look for: DIDs, Verifiable Credentials, cryptographic identity, audit trails, W3C standards
- "State" or "data sharing" → Look for: memory, scopes, cross-agent memory
- "Setup" or "getting started" → Look for: installation, af init, quick start
- "Deployment" or "production" → Look for: Docker, Kubernetes, local development, scaling
- "Agent communication" → Look for: app.call, cross-agent, workflows
- "Long-running tasks" → Look for: async execution, webhooks
- "Functions" or "endpoints" → Look for: reasoners, skills, API endpoints
- "Differences" or "comparison" or "vs" → Look for: infrastructure vs framework, control plane vs embedded library, multi-team vs single app, production features
- "Scale" or "scalability" → Look for: stateless control plane, independent scaling, billions of requests, horizontal scaling
- "Architecture" → Look for: distributed architecture, control plane, agent nodes, microservices, stateless design
"""

app = Agent(
    node_id="documentation-chatbot",
    agentfield_server=f"{os.getenv('AGENTFIELD_SERVER')}",
    ai_config=AIConfig(
        model=os.getenv("AI_MODEL", "openrouter/openai/gpt-4o-mini"),
    ),
)


# ========================= Ingestion Skill (Unchanged) =========================


@app.reasoner()
async def ingest_folder(
    folder_path: str,
    namespace: str = "documentation",
    glob_pattern: str = "**/*",
    chunk_size: int = 1200,
    chunk_overlap: int = 250,
) -> IngestReport:
    """
    Chunk + embed every supported file inside ``folder_path``.

    Uses two-tier storage:
    1. Store full document text ONCE in regular memory
    2. Store chunk vectors with reference to document
    """

    root = Path(folder_path).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    files = sorted(p for p in root.glob(glob_pattern) if p.is_file())
    supported_files = [p for p in files if is_supported_file(p)]
    skipped = [p.as_posix() for p in files if not is_supported_file(p)]

    if not supported_files:
        return IngestReport(
            namespace=namespace, file_count=0, chunk_count=0, skipped_files=skipped
        )

    global_memory = app.memory.global_scope

    total_chunks = 0
    for file_path in supported_files:
        relative_path = file_path.relative_to(root).as_posix()
        try:
            full_text = read_text(file_path)
        except Exception as exc:  # pragma: no cover - defensive
            skipped.append(f"{relative_path} (error: {exc})")
            continue

        # TIER 1: Store full document ONCE
        document_key = f"{namespace}:doc:{relative_path}"
        await global_memory.set(
            key=document_key,
            data={
                "full_text": full_text,
                "relative_path": relative_path,
                "namespace": namespace,
                "file_size": len(full_text),
            },
        )

        # Create chunks
        doc_chunks = chunk_markdown_text(
            full_text,
            relative_path=relative_path,
            namespace=namespace,
            chunk_size=chunk_size,
            overlap=chunk_overlap,
        )
        if not doc_chunks:
            continue

        # TIER 2: Store chunk vectors with document reference
        embeddings = embed_texts([chunk.text for chunk in doc_chunks])
        for idx, (chunk, embedding) in enumerate(zip(doc_chunks, embeddings)):
            vector_key = f"{namespace}|{chunk.chunk_id}"
            metadata = {
                "text": chunk.text,
                "namespace": namespace,
                "relative_path": chunk.relative_path,
                "section": chunk.section,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                # NEW: Reference to full document (not the text itself!)
                "document_key": document_key,
                "chunk_index": idx,
                "total_chunks": len(doc_chunks),
            }
            await global_memory.set_vector(
                key=vector_key, embedding=embedding, metadata=metadata
            )
            total_chunks += 1

    log_info(
        f"Ingested {total_chunks} chunks from {len(supported_files)} files into namespace '{namespace}'"
    )

    return IngestReport(
        namespace=namespace,
        file_count=len(supported_files),
        chunk_count=total_chunks,
        skipped_files=skipped,
    )


# ========================= Helper Functions =========================


def _alpha_key(index: int) -> str:
    """Convert index to alphabetic key (0->A, 1->B, ..., 26->AA)."""
    if index < 0:
        raise ValueError("Index must be non-negative")

    letters: List[str] = []
    current = index
    while True:
        current, remainder = divmod(current, 26)
        letters.append(chr(ord("A") + remainder))
        if current == 0:
            break
        current -= 1
    return "".join(reversed(letters))


def _filter_hits(
    hits: Sequence[Dict],
    *,
    namespace: str,
    min_score: float,
) -> List[Dict]:
    """Filter vector search hits by namespace and minimum score."""
    filtered: List[Dict] = []
    for hit in hits:
        metadata = hit.get("metadata", {})
        if metadata.get("namespace") != namespace:
            continue
        if hit.get("score", 0.0) < min_score:
            continue
        filtered.append(hit)
    return filtered


def _deduplicate_results(results: List[RetrievalResult]) -> List[RetrievalResult]:
    """Deduplicate by source, keeping highest score per unique chunk."""
    by_source: Dict[str, RetrievalResult] = {}

    for result in results:
        if (
            result.source not in by_source
            or result.score > by_source[result.source].score
        ):
            by_source[result.source] = result

    # Sort by score descending, limit to top 15
    deduplicated = sorted(by_source.values(), key=lambda x: x.score, reverse=True)
    return deduplicated[:15]


def _build_citations(results: Sequence[RetrievalResult]) -> List[Citation]:
    """Convert retrieval results to citation objects with alphabetic keys."""
    citations: List[Citation] = []

    for idx, result in enumerate(results):
        # Parse source format: "file.md:10-20"
        parts = result.source.split(":")
        relative_path = parts[0]
        line_range = parts[1] if len(parts) > 1 else "0-0"
        line_parts = line_range.split("-")
        start_line = int(line_parts[0]) if line_parts else 0
        end_line = int(line_parts[1]) if len(line_parts) > 1 else start_line

        key = _alpha_key(idx)
        citation = Citation(
            key=key,
            relative_path=relative_path,
            start_line=start_line,
            end_line=end_line,
            section=None,  # Could extract from metadata if needed
            preview=result.text[:200],
            score=result.score,
        )
        citations.append(citation)

    return citations


def _format_context_for_synthesis(results: Sequence[RetrievalResult]) -> str:
    """Format retrieval results as numbered context for the synthesizer."""
    if not results:
        return "(no context available)"

    blocks: List[str] = []
    for idx, result in enumerate(results):
        key = _alpha_key(idx)
        blocks.append(f"[{key}] {result.source}\n{result.text}")

    return "\n\n".join(blocks)


def _calculate_document_score(chunks: List[RetrievalResult]) -> float:
    """
    Calculate relevance score for a document based on its matching chunks.

    Score = (chunk_frequency * 0.4) + (avg_similarity * 0.4) + (max_similarity * 0.2)

    This rewards:
    - Documents with multiple matching chunks (comprehensive coverage)
    - High average relevance across chunks
    - At least one highly relevant section
    """
    if not chunks:
        return 0.0

    chunk_count = len(chunks)
    avg_score = sum(c.score for c in chunks) / chunk_count
    max_score = max(c.score for c in chunks)

    # Normalize chunk count (diminishing returns after 3 chunks)
    normalized_count = min(chunk_count / 3.0, 1.0)

    return (normalized_count * 0.4) + (avg_score * 0.4) + (max_score * 0.2)


async def _aggregate_chunks_to_documents(
    chunks: List[RetrievalResult], top_n: int = 5
) -> List[DocumentContext]:
    """
    Group chunks by document, fetch full documents, and rank by relevance.

    Returns top N most relevant full documents.
    """
    from collections import defaultdict

    global_memory = app.memory.global_scope

    # Group chunks by document_key
    by_document: Dict[str, List[RetrievalResult]] = defaultdict(list)
    for chunk in chunks:
        doc_key = chunk.metadata.get("document_key")
        if doc_key:
            by_document[doc_key].append(chunk)

    if not by_document:
        log_info("[aggregate_chunks_to_documents] No document keys found in chunks")
        return []

    log_info(
        f"[aggregate_chunks_to_documents] Found {len(by_document)} unique documents"
    )

    # Fetch full documents and build DocumentContext objects
    document_contexts: List[DocumentContext] = []
    for doc_key, doc_chunks in by_document.items():
        # Fetch full document
        doc_data = await global_memory.get(key=doc_key)
        if not doc_data:
            log_info(f"[aggregate_chunks_to_documents] Document not found: {doc_key}")
            continue

        # Calculate relevance score
        relevance_score = _calculate_document_score(doc_chunks)

        # Extract matched sections
        matched_sections = [
            chunk.metadata.get("section")
            for chunk in doc_chunks
            if chunk.metadata.get("section")
        ]
        # Remove duplicates while preserving order
        seen = set()
        unique_sections = []
        for section in matched_sections:
            if section not in seen:
                seen.add(section)
                unique_sections.append(section)

        document_contexts.append(
            DocumentContext(
                document_key=doc_key,
                full_text=doc_data.get("full_text", ""),
                relative_path=doc_data.get("relative_path", "unknown"),
                matching_chunks=len(doc_chunks),
                relevance_score=relevance_score,
                matched_sections=unique_sections,
            )
        )

    # Sort by relevance score and return top N
    ranked_documents = sorted(
        document_contexts, key=lambda x: x.relevance_score, reverse=True
    )[:top_n]

    log_info(
        f"[aggregate_chunks_to_documents] Returning top {len(ranked_documents)} documents "
        f"(scores: {[f'{d.relevance_score:.3f}' for d in ranked_documents]})"
    )

    return ranked_documents


def _format_documents_for_synthesis(documents: Sequence[DocumentContext]) -> str:
    """Format full documents with minimal metadata for better AI comprehension."""
    if not documents:
        return "(no documents available)"

    blocks: List[str] = []
    for idx, doc in enumerate(documents):
        key = _alpha_key(idx)
        # Simple, clean format - just document ID and content
        header = f"=== DOCUMENT [{key}]: {doc.relative_path} ==="
        blocks.append(f"{header}\n\n{doc.full_text}\n")

    return "\n".join(blocks)


def _build_citations_from_documents(
    documents: Sequence[DocumentContext],
) -> List[Citation]:
    """Convert document contexts to citation objects."""
    citations: List[Citation] = []

    for idx, doc in enumerate(documents):
        key = _alpha_key(idx)
        citation = Citation(
            key=key,
            relative_path=doc.relative_path,
            start_line=0,  # Full document, no specific line
            end_line=0,
            section=", ".join(doc.matched_sections) if doc.matched_sections else None,
            preview=doc.full_text[:200],
            score=doc.relevance_score,
        )
        citations.append(citation)

    return citations


# ========================= Agent 1: Query Planner =========================


@app.reasoner()
async def plan_queries(question: str) -> QueryPlan:
    """Generate 3-5 diverse search queries from the user's question."""

    return await app.ai(
        system=(
            "You are a query planning expert for documentation search. "
            "Your job is to generate 3-5 DIVERSE search queries that maximize retrieval coverage.\n\n"
            "## PRODUCT CONTEXT\n"
            f"{PRODUCT_CONTEXT}\n\n"
            "Use this context to understand product-specific terminology and generate better search queries. "
            "For example, if a user asks about 'identity', recognize they likely mean DIDs/VCs. "
            "If they ask about 'functions', they might mean reasoners or skills.\n\n"
            "## DIVERSITY STRATEGIES\n"
            "1. Use different terminology and synonyms (including product-specific terms)\n"
            "2. Cover different aspects (setup, usage, troubleshooting, configuration)\n"
            "3. Range from broad concepts to specific terms\n"
            "4. Include related concepts using the 'Search Term Relationships' above\n"
            "5. Avoid redundancy - each query should target unique angles\n\n"
            "## QUERY TYPES\n"
            "- How-to queries: 'how to install X', 'how to create X'\n"
            "- Concept queries: 'X architecture', 'what is X'\n"
            "- Troubleshooting: 'X error', 'X not working'\n"
            "- Configuration: 'X settings', 'configure X'\n"
            "- API/Reference: 'X API', 'X methods'\n"
            "- Comparison: 'X vs Y', 'when to use X'"
        ),
        user=(
            f"Question: {question}\n\n"
            "Generate 3-5 diverse search queries that cover different angles of this question. "
            "Use your knowledge of the product (AgentField) to include relevant technical terms. "
            "Also specify the strategy: 'broad' (general exploration), 'specific' (targeted search), "
            "or 'mixed' (combination of both)."
        ),
        schema=QueryPlan,
    )


# ========================= Agent 2: Parallel Retrievers =========================


async def _retrieve_for_query(
    query: str,
    namespace: str,
    top_k: int,
    min_score: float,
) -> List[RetrievalResult]:
    """Single retrieval operation for one query."""

    global_memory = app.memory.global_scope

    # Embed the query
    embedding = embed_query(query)

    # Search vector store
    raw_hits = await global_memory.similarity_search(
        query_embedding=embedding, top_k=top_k * 2  # Get more to account for filtering
    )

    # Filter by namespace and score
    filtered_hits = _filter_hits(raw_hits, namespace=namespace, min_score=min_score)

    # Convert to RetrievalResult objects
    results: List[RetrievalResult] = []
    for hit in filtered_hits[:top_k]:
        metadata = hit.get("metadata", {})
        text = metadata.get("text", "").strip()
        if not text:
            continue

        relative_path = metadata.get("relative_path", "unknown")
        start_line = int(metadata.get("start_line", 0))
        end_line = int(metadata.get("end_line", 0))
        source = f"{relative_path}:{start_line}-{end_line}"

        results.append(
            RetrievalResult(
                text=text,
                source=source,
                score=float(hit.get("score", 0.0)),
                metadata=metadata,  # Include full metadata for document aggregation
            )
        )

    return results


@app.reasoner()
async def parallel_retrieve(
    queries: List[str],
    namespace: str = "documentation",
    top_k: int = 6,
    min_score: float = 0.35,
) -> List[RetrievalResult]:
    """Execute parallel retrieval for all queries and deduplicate results."""

    log_info(f"[parallel_retrieve] Running {len(queries)} queries in parallel")

    # Execute all retrievals in parallel
    tasks = [
        _retrieve_for_query(query, namespace, top_k, min_score) for query in queries
    ]
    all_results_lists = await asyncio.gather(*tasks)

    # Flatten results
    all_results: List[RetrievalResult] = []
    for results in all_results_lists:
        all_results.extend(results)

    log_info(
        f"[parallel_retrieve] Retrieved {len(all_results)} total chunks before deduplication"
    )

    # Deduplicate and rank
    deduplicated = _deduplicate_results(all_results)

    log_info(f"[parallel_retrieve] Returning {len(deduplicated)} unique chunks")

    return deduplicated


# ========================= Agent 3: Self-Aware Synthesizer =========================


@app.reasoner()
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

    # Format context for the AI
    context_text = _format_context_for_synthesis(results)

    # Build citations
    citations = _build_citations(results)

    # Create a mapping of keys to sources for the prompt
    key_map = "\n".join(
        [
            f"[{c.key}] = {c.relative_path}:{c.start_line}-{c.end_line}"
            for c in citations
        ]
    )

    system_prompt = (
        "You are a knowledgeable documentation assistant helping users understand and use this product effectively. "
        "Your goal is to provide accurate, helpful answers that empower users to accomplish their tasks.\n\n"
        "## PRODUCT CONTEXT\n\n"
        f"{PRODUCT_CONTEXT}\n\n"
        "Use this context to understand the product's architecture, terminology, and common use cases. "
        "This helps you provide more accurate answers and explain technical concepts correctly.\n\n"
        "## Core Principles\n\n"
        "**Accuracy & Trust:**\n"
        "- Base every statement on the provided documentation\n"
        "- Cite sources using inline references like [A] or [B][C]\n"
        "- If information isn't in the docs, clearly state: 'The documentation doesn't cover this yet'\n"
        "- Never invent API names, commands, configuration values, or examples\n\n"
        "**Clarity & Usefulness:**\n"
        "- Start with a direct answer to the user's question\n"
        "- Provide specific, actionable information (actual commands, file paths, step-by-step instructions)\n"
        "- Use code blocks for commands, configuration, and code examples\n"
        "- Structure complex answers with headings, bullets, or numbered steps\n"
        "- Adapt your detail level to the question's complexity\n\n"
        "**Tone & Style:**\n"
        "- Be professional yet approachable—like a helpful colleague\n"
        "- Use clear, concise language without unnecessary jargon\n"
        "- When technical terms are needed, briefly explain them\n"
        "- Be encouraging and supportive, especially for setup/troubleshooting questions\n\n"
        "## Answer Format\n\n"
        "**Structure your response as:**\n"
        "1. **Direct answer** - Address the question immediately\n"
        "2. **Key details** - Provide specific information, commands, or steps\n"
        "3. **Context** (if helpful) - Add relevant background or related information\n"
        "4. **Next steps** (if applicable) - Guide users on what to do next\n\n"
        "**Formatting guidelines:**\n"
        "- Use GitHub-flavored Markdown\n"
        "- Format code with backticks: `inline code` or ```language blocks```\n"
        "- Use bullets for lists, numbers for sequential steps\n"
        "- Keep paragraphs focused (2-4 sentences each)\n"
        "- Add inline citations [A][B] after each factual claim\n\n"
        "## Self-Assessment\n\n"
        "After generating your answer, honestly evaluate its completeness:\n\n"
        "**Set `confidence='high'` and `needs_more=False` when:**\n"
        "- You found specific, detailed information that fully answers the question\n"
        "- All key aspects of the question are addressed with concrete details\n"
        "- The user can take action based on your answer\n\n"
        "**Set `confidence='partial'` and `needs_more=True` when:**\n"
        "- You found some relevant information but it's incomplete\n"
        "- Key details are missing (e.g., has steps 1-2 but not step 3)\n"
        "- Specify exactly what's missing in `missing_topics` (e.g., ['configuration options', 'error handling'])\n\n"
        "**Set `confidence='insufficient'` and `needs_more=True` when:**\n"
        "- After thoroughly reading all documentation, the requested information isn't present\n"
        "- The question asks about features/topics not covered in the docs\n"
        "- Specify what information would be needed in `missing_topics`\n\n"
        f"{'**Refinement Mode:** This is a second retrieval attempt. If you have useful information—even if not complete—provide it and set `needs_more=False` to avoid retrieval loops.' if is_refinement else ''}"
    )

    user_prompt = (
        f"Question: {question}\n\n"
        f"Citation Key Map:\n{key_map}\n\n"
        f"Context Chunks:\n{context_text}\n\n"
        "Generate a concise markdown answer with inline citations. "
        "Then self-assess: can you fully answer this question with the provided context? "
        "Set confidence, needs_more, and missing_topics accordingly."
    )

    # Get structured response
    response = await app.ai(
        system=system_prompt,
        user=user_prompt,
        schema=DocAnswer,
    )

    # Ensure citations are included
    if isinstance(response, DocAnswer):
        if not response.citations:
            response.citations = citations
        return response

    # Fallback if response is dict
    response_dict = response if isinstance(response, dict) else response.model_dump()
    response_dict["citations"] = citations
    return DocAnswer.model_validate(response_dict)


# ========================= Main Orchestrator =========================


@app.reasoner()
async def qa_answer(
    question: str,
    namespace: str = "documentation",
    top_k: int = 6,
    min_score: float = 0.35,
) -> DocAnswer:
    """
    Main QA orchestrator with parallel retrieval and optional refinement.

    Flow:
    1. Plan diverse queries
    2. Parallel retrieval
    3. Synthesize with self-assessment
    4. Optional refinement if needs_more=True (max 1 iteration)
    """

    log_info(f"[qa_answer] Processing question: {question}")

    # Step 1: Plan diverse queries
    plan = await plan_queries(question)
    log_info(
        f"[qa_answer] Generated {len(plan.queries)} queries with strategy: {plan.strategy}"
    )

    # Step 2: Parallel retrieval
    results = await parallel_retrieve(
        queries=plan.queries,
        namespace=namespace,
        top_k=top_k,
        min_score=min_score,
    )

    # Step 3: Synthesize answer
    answer = await synthesize_answer(question, results, is_refinement=False)

    log_info(
        f"[qa_answer] First synthesis: confidence={answer.confidence}, "
        f"needs_more={answer.needs_more}, citations={len(answer.citations)}"
    )

    # Step 4: Optional refinement (max 1 iteration)
    if answer.needs_more and answer.missing_topics:
        log_info(f"[qa_answer] Refinement needed for: {answer.missing_topics}")

        # Generate targeted queries for missing topics
        refinement_queries = []
        for topic in answer.missing_topics[:3]:  # Limit to 3 topics
            refinement_queries.append(f"{question} {topic}")
            refinement_queries.append(topic)

        # Retrieve more context
        additional_results = await parallel_retrieve(
            queries=refinement_queries,
            namespace=namespace,
            top_k=top_k,
            min_score=min_score,
        )

        # Merge with previous results and deduplicate
        all_results = results + additional_results
        merged_results = _deduplicate_results(all_results)

        log_info(
            f"[qa_answer] Refinement retrieved {len(additional_results)} new chunks, "
            f"merged to {len(merged_results)} total"
        )

        # Synthesize again with refinement flag
        answer = await synthesize_answer(question, merged_results, is_refinement=True)

        log_info(
            f"[qa_answer] Refined synthesis: confidence={answer.confidence}, "
            f"needs_more={answer.needs_more}, citations={len(answer.citations)}"
        )

    return answer


# ========================= Document-Aware QA (NEW) =========================


@app.reasoner()
async def qa_answer_with_documents(
    question: str,
    namespace: str = "documentation",
    top_k: int = 6,
    min_score: float = 0.35,
    top_documents: int = 5,
) -> DocAnswer:
    """
    Document-aware QA orchestrator that retrieves full documents instead of chunks.

    Flow:
    1. Plan diverse queries
    2. Parallel chunk retrieval
    3. Aggregate chunks to full documents
    4. Synthesize answer using full document context
    5. Optional refinement if needs_more=True (max 1 iteration)
    """

    log_info(f"[qa_answer_with_documents] Processing question: {question}")

    # Step 1: Plan diverse queries
    plan = await plan_queries(question)
    log_info(
        f"[qa_answer_with_documents] Generated {len(plan.queries)} queries with strategy: {plan.strategy}"
    )

    # Step 2: Parallel chunk retrieval
    chunk_results = await parallel_retrieve(
        queries=plan.queries,
        namespace=namespace,
        top_k=top_k,
        min_score=min_score,
    )

    # Step 3: Aggregate chunks to full documents
    documents = await _aggregate_chunks_to_documents(chunk_results, top_n=top_documents)

    if not documents:
        return DocAnswer(
            answer="I could not find any relevant documentation to answer this question.",
            citations=[],
            confidence="insufficient",
            needs_more=False,
            missing_topics=["No documentation found for this topic"],
        )

    # Step 4: Synthesize answer using full documents
    context_text = _format_documents_for_synthesis(documents)
    citations = _build_citations_from_documents(documents)

    key_map = "\n".join([f"[{c.key}] = {c.relative_path}" for c in citations])

    system_prompt = (
        "You are a knowledgeable documentation assistant helping users understand and use this product effectively. "
        "Your goal is to provide accurate, helpful answers by thoroughly reading and comprehending the full documentation pages provided.\n\n"
        "## PRODUCT CONTEXT\n\n"
        f"{PRODUCT_CONTEXT}\n\n"
        "Use this context to understand the product's architecture, terminology, and common use cases. "
        "This helps you provide more accurate answers and explain technical concepts correctly. "
        "For example, when users ask about 'identity', you know they're asking about DIDs and VCs. "
        "When they ask about 'functions', you understand they might mean reasoners or skills.\n\n"
        "## Core Principles\n\n"
        "**Accuracy & Trust:**\n"
        "- Base every statement on the provided documentation pages\n"
        "- Cite sources using inline references like [A] or [B][C]\n"
        "- If information isn't in the docs, clearly state: 'The documentation doesn't cover this yet'\n"
        "- Never invent API names, commands, configuration values, or examples\n\n"
        "**Clarity & Usefulness:**\n"
        "- Start with a direct answer to the user's question\n"
        "- Extract and present SPECIFIC details from the documentation: actual commands, file paths, configuration values, step-by-step instructions\n"
        "- Use code blocks for commands, configuration, and code examples\n"
        "- Structure complex answers with headings, bullets, or numbered steps\n"
        "- Be concrete and actionable—give users what they need to accomplish their task\n\n"
        "**Tone & Style:**\n"
        "- Be professional yet approachable—like a helpful colleague\n"
        "- Use clear, concise language without unnecessary jargon\n"
        "- When technical terms are needed, briefly explain them\n"
        "- Be encouraging and supportive, especially for setup/troubleshooting questions\n\n"
        "## Reading Instructions\n\n"
        "**How to use the documentation:**\n"
        "1. Read the full documentation pages carefully and thoroughly\n"
        "2. Find the specific information that directly answers the user's question\n"
        "3. Extract and present the actual details, steps, commands, or explanations\n"
        "4. Quote or paraphrase directly from the documentation—be specific\n"
        "5. If the answer requires multiple steps or details, extract ALL of them\n\n"
        "**Important:** Don't just say 'the documentation mentions X'—tell users exactly what it says. "
        "Don't be vague or generic—extract specific information. You are reading the documentation FOR the user.\n\n"
        "## Answer Format\n\n"
        "**Structure your response as:**\n"
        "1. **Direct answer** - Address the question immediately with specific details\n"
        "2. **Key details** - Provide actual commands, file paths, configuration values, or step-by-step instructions\n"
        "3. **Context** (if helpful) - Add relevant background or related information\n"
        "4. **Next steps** (if applicable) - Guide users on what to do next\n\n"
        "**Formatting guidelines:**\n"
        "- Use GitHub-flavored Markdown\n"
        "- Format code with backticks: `inline code` or ```language blocks```\n"
        "- Use bullets for lists, numbers for sequential steps\n"
        "- Keep paragraphs focused (2-4 sentences each)\n"
        "- Add inline citations [A][B] after each factual claim\n\n"
        "## Examples\n\n"
        "**Question:** 'How do I get started?'\n"
        "**Good Answer:**\n"
        "To get started with AgentField:\n\n"
        "1. Install the CLI: `npm install -g agentfield` [A]\n"
        "2. Initialize a new project: `af init my-project` [A]\n"
        "3. Configure your agent in the generated `agent.yaml` file [A]\n\n"
        "The initialization creates a basic project structure with example agents you can customize [A].\n\n"
        "**Question:** 'How is IAM treated?'\n"
        "**Good Answer:**\n"
        "AgentField uses Decentralized Identifiers (DIDs) for identity management [A]. Each agent receives a unique, "
        "cryptographically verifiable DID when registered [A]. You can configure IAM policies in the control plane "
        "settings under `config/agentfield.yaml` in the `security` section [B].\n\n"
        "## Self-Assessment\n\n"
        "After generating your answer, honestly evaluate its completeness:\n\n"
        "**Set `confidence='high'` and `needs_more=False` when:**\n"
        "- You found specific, detailed information that fully answers the question\n"
        "- All key aspects are addressed with concrete details from the documentation\n"
        "- The user can take action based on your answer\n"
        "- Note: If the answer requires combining info from multiple paragraphs or sections, that's still a complete answer\n\n"
        "**Set `confidence='partial'` and `needs_more=True` when:**\n"
        "- You found some relevant information but it's incomplete\n"
        "- Key details are missing (e.g., has steps 1-2 but not step 3)\n"
        "- Specify exactly what's missing in `missing_topics` (e.g., ['configuration options', 'error handling'])\n\n"
        "**Set `confidence='insufficient'` and `needs_more=True` when:**\n"
        "- After thoroughly reading all documentation pages, the requested information isn't present\n"
        "- The question asks about features/topics not covered in the docs\n"
        "- Specify what information would be needed in `missing_topics`"
    )

    user_prompt = (
        f"Question: {question}\n\n"
        f"Citation Key Map:\n{key_map}\n\n"
        f"Full Documentation Pages:\n{context_text}\n\n"
        "Generate a concise markdown answer with inline citations. "
        "Then self-assess: can you fully answer this question with the provided documents? "
        "Set confidence, needs_more, and missing_topics accordingly."
    )

    response = await app.ai(
        system=system_prompt,
        user=user_prompt,
        schema=DocAnswer,
    )

    # Ensure citations are included
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

    # Step 5: Optional refinement (max 1 iteration)
    if answer.needs_more and answer.missing_topics:
        log_info(
            f"[qa_answer_with_documents] Refinement needed for: {answer.missing_topics}"
        )

        # Generate targeted queries for missing topics
        refinement_queries = []
        for topic in answer.missing_topics[:3]:  # Limit to 3 topics
            refinement_queries.append(f"{question} {topic}")
            refinement_queries.append(topic)

        # Retrieve more chunks
        additional_chunks = await parallel_retrieve(
            queries=refinement_queries,
            namespace=namespace,
            top_k=top_k,
            min_score=min_score,
        )

        # Merge and aggregate to documents
        all_chunks = chunk_results + additional_chunks
        merged_documents = await _aggregate_chunks_to_documents(
            all_chunks, top_n=top_documents
        )

        log_info(
            f"[qa_answer_with_documents] Refinement found {len(merged_documents)} total documents"
        )

        # Synthesize again with more lenient prompt
        context_text = _format_documents_for_synthesis(merged_documents)
        citations = _build_citations_from_documents(merged_documents)
        key_map = "\n".join([f"[{c.key}] = {c.relative_path}" for c in citations])

        system_prompt_refined = (
            system_prompt
            + "\n\nREFINEMENT MODE: This is a second attempt. Be more lenient - if you have ANY useful info, set needs_more=False."
        )

        user_prompt_refined = (
            f"Question: {question}\n\n"
            f"Citation Key Map:\n{key_map}\n\n"
            f"Full Documentation Pages:\n{context_text}\n\n"
            "Generate a concise markdown answer with inline citations. "
            "Then self-assess: can you fully answer this question with the provided documents? "
            "Set confidence, needs_more, and missing_topics accordingly."
        )

        response = await app.ai(
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


# ========================= Bootstrapping =========================


def _warmup_embeddings() -> None:
    """Warm up the embedding model on startup."""
    try:
        embed_texts(["doc-chatbot warmup"])
        log_info("FastEmbed model warmed up for documentation chatbot")
    except Exception as exc:  # pragma: no cover - best-effort
        log_info(f"FastEmbed warmup failed: {exc}")


if __name__ == "__main__":
    _warmup_embeddings()

    print("📚 Simplified Documentation Chatbot Agent")
    print("🧠 Node ID: documentation-chatbot")
    print(f"🌐 Control Plane: {app.agentfield_server}")
    print("\n🎯 Architecture: 3-Agent Parallel System + Document-Level Retrieval")
    print("  1. Query Planner → Generates diverse search queries")
    print("  2. Parallel Retrievers → Concurrent vector search")
    print("  3. Self-Aware Synthesizer → Answer + confidence assessment")
    print("\n✨ Features:")
    print("  - Parallel retrieval for 3x speed improvement")
    print("  - Self-aware synthesis (no separate review)")
    print("  - Max 1 refinement iteration (prevents loops)")
    print("  - Document-level context (full pages vs isolated chunks)")
    print("  - Smart document ranking (frequency + relevance scoring)")

    port_env = os.getenv("PORT")
    if port_env is None:
        app.run(auto_port=True, host="::")
    else:
        app.run(port=int(port_env), host="::")
