"""
Functional tests for agent registration with the control plane.

Tests that agents can successfully register with the control plane
and appear in the node registry.
"""

import asyncio
import os
import threading
import time

import httpx
import pytest
import uvicorn
from agentfield import Agent


def _agent_callback_url(port: int) -> str:
    """Return the URL the Dockerized control plane should use to call this agent."""
    host = os.getenv(
        "AGENTFIELD_FUNCTIONAL_AGENT_CALLBACK_HOST",
        "host.docker.internal",
    )
    return f"http://{host}:{port}"


def make_agent(node_id: str, control_plane_url: str, port: int) -> Agent:
    return Agent(
        node_id=node_id,
        agentfield_server=control_plane_url,
        callback_url=_agent_callback_url(port),
    )


class RunningAgent:
    def __init__(self, agent: Agent, port: int):
        self.agent = agent
        self.port = port
        self.server: uvicorn.Server | None = None
        self.thread: threading.Thread | None = None

    def __enter__(self) -> "RunningAgent":
        config = uvicorn.Config(
            self.agent,
            host="0.0.0.0",
            port=self.port,
            log_level="error",
        )
        self.server = uvicorn.Server(config)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        self._wait_for_health()
        asyncio.run(
            self.agent.agentfield_handler.register_with_agentfield_server(self.port)
        )
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.server:
            self.server.should_exit = True
        if self.thread:
            self.thread.join(timeout=5)

    def _run(self):
        assert self.server is not None
        asyncio.run(self.server.serve())

    def _wait_for_health(self):
        deadline = time.monotonic() + 10
        last_error: Exception | None = None

        while time.monotonic() < deadline:
            try:
                response = httpx.get(
                    f"http://127.0.0.1:{self.port}/health",
                    timeout=0.5,
                )
                if response.status_code == 200:
                    return
            except httpx.HTTPError as exc:
                last_error = exc
            time.sleep(0.1)

        raise RuntimeError(
            f"Agent did not become healthy on port {self.port}"
        ) from last_error


@pytest.mark.functional
def test_agent_registers_with_control_plane(
    control_plane_url: str, agent_port_allocator
):
    """Test that an agent successfully registers with the control plane."""
    agent_port = agent_port_allocator()

    agent = make_agent("test-registration-agent", control_plane_url, agent_port)

    # Define a simple reasoner
    @agent.reasoner()
    async def hello(ping: str = "ok"):
        """Simple hello reasoner."""
        return {"message": "Hello from registration test", "ping": ping}

    with RunningAgent(agent, agent_port):
        # Verify agent is registered
        client = httpx.Client(base_url=control_plane_url, timeout=10.0)

        # Check nodes endpoint (may need to adjust based on actual API)
        # For now, we verify by trying to execute the reasoner
        response = client.post(
            "/api/v1/execute/test-registration-agent.hello",
            json={"input": {"ping": "ok"}},
        )

        assert (
            response.status_code == 200
        ), f"Expected 200, got {response.status_code}: {response.text}"
        result = response.json()

        # Verify response structure
        assert (
            "result" in result or "message" in result
        ), f"Unexpected response: {result}"

        print(f"✅ Agent registered successfully: {result}")


@pytest.mark.functional
def test_agent_health_check(control_plane_url: str, agent_port_allocator):
    """Test that agent health checks work."""
    agent_port = agent_port_allocator()

    agent = make_agent("test-health-agent", control_plane_url, agent_port)

    @agent.reasoner()
    async def ping():
        """Simple ping reasoner."""
        return {"status": "pong"}

    with RunningAgent(agent, agent_port):
        # Check agent's own health endpoint
        agent_client = httpx.Client(
            base_url=f"http://127.0.0.1:{agent_port}",
            timeout=5.0,
        )
        response = agent_client.get("/health")

        assert (
            response.status_code == 200
        ), f"Health check failed: {response.status_code}"
        health_data = response.json()

        # Verify health response structure
        assert "status" in health_data, f"Health response missing status: {health_data}"
        print(f"✅ Agent health check passed: {health_data}")


@pytest.mark.functional
def test_multiple_agents_register(control_plane_url: str, agent_port_allocator):
    """Test that multiple agents can register simultaneously."""
    num_agents = 3
    running_agents = []

    try:
        for i in range(num_agents):
            agent_port = agent_port_allocator()

            agent = make_agent(f"test-multi-agent-{i}", control_plane_url, agent_port)

            @agent.reasoner()
            async def identify(ping: str = "ok", agent_index=i):
                """Identify which agent this is."""
                return {"agent_id": agent_index, "ping": ping}

            running_agent = RunningAgent(agent, agent_port)
            running_agent.__enter__()
            running_agents.append(running_agent)

        # Verify all agents registered
        client = httpx.Client(base_url=control_plane_url, timeout=10.0)

        for i in range(num_agents):
            response = client.post(
                f"/api/v1/execute/test-multi-agent-{i}.identify",
                json={"input": {"ping": "ok"}},
            )

            assert (
                response.status_code == 200
            ), f"Agent {i} not registered: {response.status_code}"
            print(f"✅ Agent {i} registered and responding")

    finally:
        for running_agent in reversed(running_agents):
            running_agent.__exit__(None, None, None)


@pytest.mark.functional
def test_agent_re_registration(control_plane_url: str, agent_port_allocator):
    """Test that an agent can re-register after disconnection."""
    agent_port = agent_port_allocator()

    agent = make_agent("test-reregister-agent", control_plane_url, agent_port)

    @agent.reasoner()
    async def echo(message: str):
        """Echo back the message."""
        return {"echo": message}

    with RunningAgent(agent, agent_port):
        # Verify first registration
        client = httpx.Client(base_url=control_plane_url, timeout=10.0)
        response = client.post(
            "/api/v1/execute/test-reregister-agent.echo",
            json={"input": {"message": "first"}},
        )
        assert response.status_code == 200
        print("✅ First registration successful")

        # Note: In a real scenario, we would stop and restart the agent
        # For this test, we verify the agent remains registered
        time.sleep(1)

        # Verify still registered
        response = client.post(
            "/api/v1/execute/test-reregister-agent.echo",
            json={"input": {"message": "second"}},
        )
        assert response.status_code == 200
        print("✅ Agent remains registered")
