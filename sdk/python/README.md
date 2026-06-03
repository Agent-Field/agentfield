# AgentField Python SDK

The AgentField SDK provides a production-ready Python interface for registering agents, executing workflows, and integrating with the AgentField control plane.

## Installation

```bash
pip install agentfield
```

To work on the SDK locally:

```bash
git clone https://github.com/Agent-Field/agentfield.git
cd agentfield/sdk/python
python -m pip install -e .[dev]
```

## Quick Start

```python
from agentfield import Agent

agent = Agent(
    node_id="example-agent",
    agentfield_server="http://localhost:8080",
    dev_mode=True,
)

@agent.reasoner()
async def summarize(text: str) -> dict:
    result = await agent.ai(
        prompt=f"Summarize: {text}",
        response_model={"summary": "string", "tone": "string"},
    )
    return result

if __name__ == "__main__":
    agent.serve(port=8001)
```

## AI Tool Calling

Let LLMs automatically discover and invoke agent capabilities across your system:

```python
from agentfield import Agent, AIConfig, ToolCallConfig

app = Agent(
    node_id="orchestrator",
    agentfield_server="http://localhost:8080",
    ai_config=AIConfig(model="openai/gpt-4o-mini"),
)

@app.reasoner()
async def ask_with_tools(question: str) -> dict:
    # Auto-discover all tools and let the LLM use them
    result = await app.ai(
        system="You are a helpful assistant.",
        user=question,
        tools="discover",
    )
    return {"answer": str(result), "trace": result.trace}

# Filter by tags, limit turns, use lazy hydration
result = await app.ai(
    user="Get weather for Tokyo",
    tools=ToolCallConfig(
        tags=["weather"],
        schema_hydration="lazy",  # Reduces token usage for large catalogs
        max_turns=5,
        max_tool_calls=10,
    ),
)
```

**Key features:**
- `tools="discover"` — Auto-discover all capabilities from the control plane
- `ToolCallConfig` — Filter by tags, agent IDs, health status
- **Lazy hydration** — Send only tool names/descriptions first, hydrate schemas on demand
- **Guardrails** — `max_turns` and `max_tool_calls` prevent runaway loops
- **Observability** — `result.trace` tracks every tool call with latency

See `examples/python_agent_nodes/tool_calling/` for a complete orchestrator + worker example.

## Human-in-the-Loop Approvals

The Python SDK provides a first-class waiting state for pausing agent execution mid-reasoner and waiting for human approval:

```python
from agentfield import Agent, ApprovalResult

app = Agent(node_id="reviewer", agentfield_server="http://localhost:8080")

@app.reasoner()
async def deploy(environment: str) -> dict:
    plan = await app.ai(f"Create deployment plan for {environment}")

    # Pause execution and wait for human approval
    result: ApprovalResult = await app.pause(
        approval_request_id="req-abc123",
        expires_in_hours=24,
        timeout=3600,
    )

    if result.approved:
        return {"status": "deploying", "plan": str(plan)}
    elif result.changes_requested:
        return {"status": "revising", "feedback": result.feedback}
    else:
        return {"status": result.decision}
```

**Two API levels:**

- **High-level:** `app.pause()` blocks the reasoner until approval resolves, with automatic webhook registration
- **Low-level:** `client.request_approval()`, `client.get_approval_status()`, `client.wait_for_approval()` for fine-grained control

See `examples/python_agent_nodes/waiting_state/` for a complete working example.

See `docs/DEVELOPMENT.md` for instructions on wiring agents to the control plane.

## Testing

```bash
./scripts/run_pytest.sh
```

To run coverage locally:

```bash
./scripts/run_pytest.sh --cov=agentfield --cov-report=term-missing
```

The wrapper sets a private `PYTEST_DEBUG_TEMPROOT` automatically so local runs
and CI do not rely on pytest's predictable default temp directory layout.

## License

Distributed under the Apache 2.0 License. See the project root `LICENSE` for details.
