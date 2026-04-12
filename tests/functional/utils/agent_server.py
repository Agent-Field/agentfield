"""
Utilities for running test agents inside functional tests.

The `run_agent_server` async context manager handles spinning up a FastAPI
server via uvicorn, registering the agent with the control plane, and then
cleanly shutting everything down when the test completes.
"""

from __future__ import annotations

import asyncio
import os
import socket
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator

import uvicorn
from agentfield import Agent

AGENT_BIND_HOST = os.environ.get("TEST_AGENT_BIND_HOST", "127.0.0.1")
AGENT_CALLBACK_HOST = os.environ.get("TEST_AGENT_CALLBACK_HOST", "127.0.0.1")


@dataclass
class RunningAgent:
    """Metadata about a running agent server."""

    agent: Agent
    port: int
    base_url: str


@asynccontextmanager
async def run_agent_server(
    agent: Agent,
    *,
    bind_host: str = AGENT_BIND_HOST,
    callback_host: str = AGENT_CALLBACK_HOST,
    startup_delay: float = 2.0,
    registration_delay: float = 2.0,
) -> AsyncIterator[RunningAgent]:
    """
    Start the given agent in a background uvicorn server for the duration of a test.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((bind_host, 0))
        port = s.getsockname()[1]

    agent.base_url = f"http://{callback_host}:{port}"

    config = uvicorn.Config(
        app=agent,
        host=bind_host,
        port=port,
        log_level="error",
        access_log=False,
        ws="websockets-sansio",
    )
    server = uvicorn.Server(config)
    loop = asyncio.new_event_loop()

    def run_server():
        asyncio.set_event_loop(loop)
        loop.run_until_complete(server.serve())

    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()

    await asyncio.sleep(startup_delay)

    try:
        await agent.agentfield_handler.register_with_agentfield_server(port)
        agent.agentfield_server = None

        # Registration runs on the pytest event loop, but reasoners execute on the
        # uvicorn event loop inside a background thread. Reset the AgentField client
        # so async HTTP clients are re-created within the uvicorn loop to avoid
        # "bound to a different event loop" errors when performing memory operations.
        try:
            await agent.client.aclose()
        except AttributeError:
            pass

        await asyncio.sleep(registration_delay)

        yield RunningAgent(agent=agent, port=port, base_url=agent.base_url)
    finally:
        server.should_exit = True
        if loop.is_running():
            loop.call_soon_threadsafe(lambda: None)
        thread.join(timeout=10)
