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
pytest
```

To run coverage locally:

```bash
pytest --cov=agentfield --cov-report=term-missing
```

## License

Distributed under the Apache 2.0 License. See the project root `LICENSE` for details.
