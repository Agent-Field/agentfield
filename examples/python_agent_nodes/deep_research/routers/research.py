"""Research execution router with Tavily integration."""

from __future__ import annotations

import os
from typing import List

from agentfield import AgentRouter

from schemas import Citation, ResearchFindings, SearchQueries, TaskResult


research_router = AgentRouter(prefix="research")


@research_router.reasoner()
async def generate_search_queries(
    task_description: str,
    research_question: str,
    dependency_context: str = "",
) -> SearchQueries:
    """Generate focused search queries for a research task, optionally with dependency context."""
    context_section = ""
    if dependency_context:
        context_section = (
            "\n## DEPENDENCY CONTEXT\n"
            "You have answers from child tasks. Use this context to enhance your search queries:\n"
            f"{dependency_context}\n\n"
            "Your queries should:\n"
            "- Build on the provided context\n"
            "- Fill gaps or find additional information needed\n"
            "- Use context to make queries more specific and targeted\n"
        )

    response = await research_router.ai(
        system=(
            "You are a search query expert generating queries for a research task.\n\n"
            "## CONTEXT\n"
            "This task will be answered via web search.\n"
            "Generate queries optimized for finding specific answers.\n\n"
            f"{context_section}"
            "## YOUR TASK\n"
            "Generate 2-4 focused search queries optimized for web search.\n\n"
            "Each query should be:\n"
            "- Specific and targeted to answer the task question\n"
            "- Use relevant keywords and terminology\n"
            "- Cover different angles/aspects of the task\n"
            "- Designed to find specific answers, not general information\n\n"
            "Return ONLY a JSON object with a 'queries' array of search strings."
        ),
        user=(
            f"Research Question: {research_question}\n"
            f"Task (needs web search): {task_description}\n\n"
            f"{f'Context from dependencies: {dependency_context}' if dependency_context else ''}\n\n"
            f"Generate 2-4 search queries that will find specific answers to this question."
        ),
        schema=SearchQueries,
    )
    return response


@research_router.reasoner()
async def execute_search(queries: List[str]) -> dict:
    """Execute Tavily search for given queries."""
    try:
        from tavily import TavilyClient
    except ImportError:
        raise ImportError("tavily-python not installed. Run: pip install tavily-python")

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY environment variable not set")

    client = TavilyClient(api_key=api_key)

    # Execute searches in parallel
    all_results = []
    for query in queries:
        try:
            result = client.search(
                query=query,
                search_depth="advanced",
                max_results=5,
                include_answer=False,
                include_raw_content=True,
            )
            all_results.append(result)
        except Exception as e:
            # Continue with other queries if one fails
            all_results.append({"error": str(e), "query": query})

    # Combine results
    combined = {
        "results": [],
        "queries": queries,
    }

    for result in all_results:
        if "error" not in result and "results" in result:
            combined["results"].extend(result["results"])

    return combined


@research_router.reasoner()
async def synthesize_findings(
    task_description: str,
    research_question: str,
    search_results: dict,
) -> ResearchFindings:
    """Synthesize search results into structured findings with citations."""
    # Format search results for the prompt
    results_text = ""
    citations_data = []

    for idx, result in enumerate(search_results.get("results", [])[:10]):
        title = result.get("title", "Untitled")
        url = result.get("url", "")
        content = result.get("content", "")
        raw_content = result.get("raw_content", "")

        excerpt = raw_content[:500] if raw_content else content[:300]
        results_text += f"\n[{idx + 1}] {title}\nURL: {url}\nContent: {excerpt}\n"

        citations_data.append({"url": url, "title": title, "excerpt": excerpt})

    response = await research_router.ai(
        system=(
            "You are a research synthesis expert. Synthesize search results into "
            "structured findings with numbered points.\n\n"
            "Format findings as:\n"
            "1. First finding [1] (reference by number)\n"
            "2. Second finding [2]\n"
            "etc.\n\n"
            "Extract citations: for each numbered reference, provide URL, title, and excerpt.\n\n"
            "Assess confidence: 'high' if comprehensive, 'medium' if partial, 'low' if insufficient."
        ),
        user=(
            f"Research Question: {research_question}\n"
            f"Task: {task_description}\n\n"
            f"Search Results:{results_text}\n\n"
            f"Synthesize findings with numbered points and citations. "
            f"Reference sources by number [1], [2], etc."
        ),
        schema=ResearchFindings,
    )

    # Map citations from search results
    citation_map = {idx + 1: Citation(**c) for idx, c in enumerate(citations_data)}

    # Update citations in response based on references in findings
    final_citations = []
    for i in range(1, len(citations_data) + 1):
        if f"[{i}]" in response.findings and i in citation_map:
            final_citations.append(citation_map[i])

    response.citations = final_citations
    return response


@research_router.reasoner()
async def execute_research_task(
    task_id: str,
    task_description: str,
    research_question: str,
) -> TaskResult:
    """Orchestrate research execution: queries → search → synthesize."""
    # Generate search queries
    queries_response = await generate_search_queries(
        task_description, research_question
    )

    # Execute search
    search_results = await execute_search(queries_response.queries)

    # Synthesize findings
    findings_response = await synthesize_findings(
        task_description, research_question, search_results
    )

    # Convert to TaskResult
    sources = [f"{c.title} ({c.url})" for c in findings_response.citations]

    return TaskResult(
        task_id=task_id,
        description=task_description,
        findings=findings_response.findings,
        sources=sources,
        confidence=findings_response.confidence,
    )


@research_router.reasoner()
async def synthesize_from_dependencies(
    task_id: str,
    task_description: str,
    research_question: str,
    dependency_findings: List[TaskResult],
) -> TaskResult:
    """
    Synthesize answer for a parent task from its children's findings.

    This is for tasks that have dependencies - they synthesize from
    child task answers rather than searching the web.
    """
    # Format dependency findings for the prompt
    children_text = "\n\n".join(
        f"Child Task {idx + 1} ({dep.task_id}): {dep.description}\n"
        f"Answer: {dep.findings}\n"
        f"Sources: {', '.join(dep.sources) if dep.sources else 'N/A'}"
        for idx, dep in enumerate(dependency_findings)
    )

    response = await research_router.ai(
        system=(
            "You are answering a PARENT TASK by synthesizing answers from child tasks.\n\n"
            "## CONTEXT\n"
            "This is a parent task - it depends on child tasks that have already been answered.\n"
            "Your job is to combine the child answers to answer the parent question.\n\n"
            "## IMPORTANT\n"
            "- Do NOT search the web - use ONLY the provided child findings\n"
            "- Synthesize and combine the child answers\n"
            "- Create a coherent answer to the parent question\n"
            "- Reference which child tasks contributed to each point\n\n"
            "## OUTPUT FORMAT\n"
            "Provide a structured answer with numbered points.\n"
            "Reference child tasks like: 'Based on Child Task 1, ...'\n"
            "Assess confidence based on completeness of child answers."
        ),
        user=(
            f"Research Question: {research_question}\n\n"
            f"Parent Task: {task_description}\n\n"
            f"Child Task Answers:\n{children_text}\n\n"
            f"Synthesize an answer to the parent task by combining the child task findings. "
            f"Do not search - use only the provided child answers."
        ),
        schema=ResearchFindings,
    )

    # Collect sources from all dependencies
    all_sources = []
    for dep in dependency_findings:
        all_sources.extend(dep.sources)

    # Add references to child tasks
    child_refs = [f"Child task {dep.task_id}" for dep in dependency_findings]
    all_sources.extend(child_refs)

    return TaskResult(
        task_id=task_id,
        description=task_description,
        findings=response.findings,
        sources=list(set(all_sources)),  # Deduplicate
        confidence=response.confidence,
    )


@research_router.reasoner()
async def execute_research_task_with_context(
    task_id: str,
    task_description: str,
    research_question: str,
    dependency_findings: List[TaskResult],
) -> TaskResult:
    """
    Execute research task with dependency context for enhanced search.

    This is for parent tasks that need additional web search beyond
    what their children provide. Uses dependency context to enhance queries.
    """
    # Format dependency context for query generation
    dependency_context = "\n\n".join(
        f"Child Task {idx + 1} ({dep.task_id}): {dep.description}\n"
        f"Answer: {dep.findings}"
        for idx, dep in enumerate(dependency_findings)
    )

    # Generate enhanced search queries with context
    queries_response = await generate_search_queries(
        task_description=task_description,
        research_question=research_question,
        dependency_context=dependency_context,
    )

    # Execute search
    search_results = await execute_search(queries_response.queries)

    # Synthesize findings (can reference both search results and dependency context)
    results_text = ""
    citations_data = []

    for idx, result in enumerate(search_results.get("results", [])[:10]):
        title = result.get("title", "Untitled")
        url = result.get("url", "")
        content = result.get("content", "")
        raw_content = result.get("raw_content", "")

        excerpt = raw_content[:500] if raw_content else content[:300]
        results_text += f"\n[{idx + 1}] {title}\nURL: {url}\nContent: {excerpt}\n"

        citations_data.append({"url": url, "title": title, "excerpt": excerpt})

    # Synthesize with both search results and dependency context
    response = await research_router.ai(
        system=(
            "You are a research synthesis expert. Synthesize findings from web search "
            "AND dependency context into structured findings.\n\n"
            "## CONTEXT\n"
            "You have:\n"
            "1. Web search results (new information)\n"
            "2. Dependency context (answers from child tasks)\n\n"
            "## YOUR TASK\n"
            "Combine both sources to answer the task question comprehensively.\n"
            "Format findings as:\n"
            "1. First finding [1] (reference by number)\n"
            "2. Second finding [2]\n"
            "etc.\n\n"
            "Extract citations: for each numbered reference, provide URL, title, and excerpt.\n\n"
            "Assess confidence: 'high' if comprehensive, 'medium' if partial, 'low' if insufficient."
        ),
        user=(
            f"Research Question: {research_question}\n"
            f"Task: {task_description}\n\n"
            f"Dependency Context (from child tasks):\n{dependency_context}\n\n"
            f"Web Search Results:{results_text}\n\n"
            f"Synthesize findings combining both sources. "
            f"Reference web sources by number [1], [2], etc."
        ),
        schema=ResearchFindings,
    )

    # Map citations from search results
    citation_map = {idx + 1: Citation(**c) for idx, c in enumerate(citations_data)}

    # Update citations in response based on references in findings
    final_citations = []
    for i in range(1, len(citations_data) + 1):
        if f"[{i}]" in response.findings and i in citation_map:
            final_citations.append(citation_map[i])

    response.citations = final_citations

    # Collect sources from dependencies and search
    all_sources = []
    for dep in dependency_findings:
        all_sources.extend(dep.sources)
    all_sources.extend([f"{c.title} ({c.url})" for c in final_citations])

    return TaskResult(
        task_id=task_id,
        description=task_description,
        findings=response.findings,
        sources=list(set(all_sources)),  # Deduplicate
        confidence=response.confidence,
    )
