"""
Flaky Agent - Fails intermittently.
Used to test error handling, retry logic, and execution failure visibility.
"""

import os
import random
import time

from agentfield import Agent, AIConfig

app = Agent(
    node_id="test-flaky",
    agentfield_server=os.getenv("AGENTFIELD_URL", "http://localhost:8080"),
    ai_config=AIConfig(
        model=os.getenv("SMALL_MODEL", "openai/gpt-4o-mini"), temperature=0.0
    ),


)


@app.reasoner()
async def maybe_fail(fail_rate: float = 0.5) -> dict:
    """
    Fails with the given probability (0.0 = never, 1.0 = always).
    Useful for testing retry behavior.
    """
    if random.random() < fail_rate:
        raise RuntimeError(f"Simulated failure (fail_rate={fail_rate})")
    return {"agent": "test-flaky", "status": "ok", "fail_rate": fail_rate}


@app.reasoner()
async def timeout_then_fail(delay_seconds: int = 60) -> dict:
    """
    Hangs for a long time then raises an error.
    Simulates an agent stuck on a dead LLM backend.
    """
    time.sleep(delay_seconds)
    raise RuntimeError("Simulated timeout + failure after long hang")


@app.reasoner()
async def crash(message: str = "boom") -> dict:
    """Always raises an exception. Simulates an agent crash."""
    raise RuntimeError(f"Agent crashed: {message}")


if __name__ == "__main__":
    app.run(auto_port=True)
