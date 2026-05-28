"""Query planning router for documentation chatbot."""

from agentfield import AgentRouter

from product_context import PRODUCT_CONTEXT
from schemas import QueryPlan

# Create router with prefix "query"
# This will transform function names: plan_queries -> query_plan_queries
query_router = AgentRouter(prefix="query")


@query_router.reasoner()
async def plan_queries(question: str) -> QueryPlan:
    """Generate 3-5 diverse search queries from the user's question."""

    return await query_router.ai(
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
