"""
Healthy Agent - Completes quickly, used as a control/baseline.
"""

import os
from agentfield import Agent, AIConfig

app = Agent(
    node_id="test-healthy",
    agentfield_server=os.getenv("AGENTFIELD_URL", "http://localhost:8080"),
    ai_config=AIConfig(
        model=os.getenv("SMALL_MODEL", "openai/gpt-4o-mini"), temperature=0.0
    ),
)


@app.reasoner()
async def echo(message: str = "hello") -> dict:
    """Instantly returns the input - used to verify basic execution works."""
    return {"echoed": message, "agent": "test-healthy", "status": "ok"}


@app.reasoner()
async def add(a: int = 0, b: int = 0) -> dict:
    """Simple math - deterministic, instant."""
    return {"result": a + b, "agent": "test-healthy"}


if __name__ == "__main__":
    app.run(auto_port=True)
