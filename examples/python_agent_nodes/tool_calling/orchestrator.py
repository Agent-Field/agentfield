"""
Orchestrator Agent - Demonstrates the tool-calling pipeline.

This agent uses `app.ai(tools=...)` to automatically:
1. Discover available tools from the control plane
2. Present them to the LLM as callable functions
3. Let the LLM decide which tools to call
4. Dispatch tool calls via `app.call()` through the control plane
5. Feed results back to the LLM
6. Repeat until the LLM produces a final answer

Requires:
- Control plane running at localhost:8080
- Worker agent registered (run worker.py first)

Examples show:
- tools="discover" (simple auto-discovery)
- tools=ToolCallConfig (with filtering, progressive discovery, guardrails)
"""

import asyncio
import json
import os

from agentfield import Agent, AIConfig, ToolCallConfig

app = Agent(
    node_id="orchestrator",
    agentfield_server=os.getenv("AGENTFIELD_URL", "http://localhost:8080"),
    ai_config=AIConfig(
        model=os.getenv("MODEL", "openrouter/openai/gpt-4o-mini"),
        temperature=0.2,
    ),
)


# ============= EXAMPLE 1: Simple discover-all =============


@app.reasoner(tags=["demo"])
async def ask_with_tools(question: str) -> dict:
    """
    Answer a question using auto-discovered tools.

    The simplest form: tools="discover" auto-discovers ALL available tools
    from the control plane and lets the LLM use them.
    """
    result = await app.ai(
        system="You are a helpful assistant. Use the available tools to answer the user's question accurately.",
        user=question,
        tools="discover",
    )
    return {"answer": str(result)}


# ============= EXAMPLE 2: Filtered discovery with tags =============


@app.reasoner(tags=["demo"])
async def weather_report(cities: str) -> dict:
    """
    Get weather for specific cities using tag-filtered discovery.

    Only discovers tools tagged with "weather".
    """
    result = await app.ai(
        system="You are a weather reporter. Get the weather for each city and provide a brief report.",
        user=f"What's the weather like in: {cities}?",
        tools=ToolCallConfig(tags=["weather"]),
    )
    return {"report": str(result)}


# ============= EXAMPLE 3: Progressive/lazy discovery =============


@app.reasoner(tags=["demo"])
async def smart_query(question: str) -> dict:
    """
    Answer using progressive discovery (lazy schema hydration).

    First pass sends only tool names/descriptions to the LLM.
    When the LLM selects tools, their full schemas are hydrated.
    This reduces context window pressure for large capability catalogs.
    """
    result = await app.ai(
        system="You are a helpful assistant with access to tools. Use them when needed.",
        user=question,
        tools=ToolCallConfig(
            schema_hydration="lazy",
            max_candidate_tools=30,
            max_hydrated_tools=8,
        ),
    )
    return {"answer": str(result)}


# ============= EXAMPLE 4: With guardrails =============


@app.reasoner(tags=["demo"])
async def guarded_query(question: str) -> dict:
    """
    Answer with strict guardrails on tool usage.

    Limits both the number of LLM turns and total tool calls
    to prevent runaway execution.
    """
    result = await app.ai(
        system="You are a helpful assistant. Be efficient with tool usage.",
        user=question,
        tools="discover",
        max_turns=3,
        max_tool_calls=5,
    )
    return {"answer": str(result)}


# ============= CLI TESTING =============


async def run_examples():
    """Run examples directly for manual testing (requires control plane + worker)."""
    # Mark as connected so cross-agent calls route through the control plane
    app.agentfield_connected = True

    print("\n" + "=" * 60)
    print("Tool Calling Examples - Manual Test")
    print("=" * 60)

    # Example 1: Simple discovery (filtered to our worker agent)
    print("\n--- Example 1: Simple tool discovery ---")
    try:
        result = await app.ai(
            system="You are a helpful assistant. Use the available tools to answer accurately.",
            user="What's the weather like in Tokyo and New York? Also, what is 42 * 17?",
            tools=ToolCallConfig(agent_ids=["utility-worker"]),
        )
        print(f"Answer: {result}")
    except Exception as e:
        print(f"Error: {e}")

    # Example 2: Filtered by tags
    print("\n--- Example 2: Tag-filtered discovery ---")
    try:
        result = await app.ai(
            system="You are a weather reporter.",
            user="Give me the weather for London and Paris.",
            tools=ToolCallConfig(tags=["weather"]),
        )
        print(f"Weather: {result}")
    except Exception as e:
        print(f"Error: {e}")

    # Example 3: With guardrails
    print("\n--- Example 3: Guardrailed query ---")
    try:
        result = await app.ai(
            system="You are a helpful assistant.",
            user="What's 100 + 200? And what's the weather in Sydney?",
            tools=ToolCallConfig(agent_ids=["utility-worker"]),
            max_turns=3,
            max_tool_calls=4,
        )
        print(f"Answer: {result}")
    except Exception as e:
        print(f"Error: {e}")

    # Example 4: Dict-based config (Sam's suggested API shape)
    print("\n--- Example 4: Dict-based config (Sam's API) ---")
    try:
        result = await app.ai(
            system="You are a helpful assistant.",
            user="Summarize the weather conditions across Tokyo, London, and Sydney.",
            tools={
                "agent_ids": ["utility-worker"],
                "tags": ["weather"],
                "schema_hydration": "eager",
                "max_candidate_tools": 30,
                "max_tool_calls": 12,
                "max_turns": 8,
            },
        )
        print(f"Answer: {result}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    import sys

    if "--test" in sys.argv:
        # Run examples directly (requires control plane + worker agent)
        asyncio.run(run_examples())
    else:
        print("Orchestrator Agent starting...")
        print("  Node: orchestrator")
        print("  Reasoners: ask_with_tools, weather_report, smart_query, guarded_query")
        print()
        print("  Use --test flag to run examples directly")
        app.run(port=8002)
