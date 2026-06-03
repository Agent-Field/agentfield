"""
Slow Agent - Simulates long-running tasks.
Used to test concurrency limits and stale execution detection.
"""

import asyncio
import os
import time

from agentfield import Agent, AIConfig

app = Agent(
    node_id="test-slow",
    agentfield_server=os.getenv("AGENTFIELD_URL", "http://localhost:8080"),
    ai_config=AIConfig(
        model=os.getenv("SMALL_MODEL", "openai/gpt-4o-mini"), temperature=0.0
    ),
)


@app.reasoner()
async def slow_task(duration_seconds: int = 30) -> dict:
    """
    Blocks for the specified duration.
    Simulates an agent waiting on an LLM response that takes forever.
    """
    start = time.time()
    await asyncio.sleep(duration_seconds)
    elapsed = time.time() - start
    return {
        "agent": "test-slow",
        "requested_duration": duration_seconds,
        "actual_duration": round(elapsed, 2),
        "status": "completed",
    }


if __name__ == "__main__":
    app.run(auto_port=True)
