# Haxen Python SDK

The Haxen SDK provides a production-ready Python interface for registering agents, executing workflows, and integrating with the Haxen control plane.

## Installation

```bash
pip install haxen-sdk
```

To work on the SDK locally:

```bash
git clone https://github.com/your-org/haxen.git
cd haxen/sdk/python
python -m pip install -e .[dev]
```

## Quick Start

```python
from haxen_sdk import Agent

agent = Agent(
    node_id="example-agent",
    haxen_server="http://localhost:8080",
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
pytest --cov=haxen_sdk --cov-report=term-missing
```

## License

Distributed under the Apache 2.0 License. See the project root `LICENSE` for details.
