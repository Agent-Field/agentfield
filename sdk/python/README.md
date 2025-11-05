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
