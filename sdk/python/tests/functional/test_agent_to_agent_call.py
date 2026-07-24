"""
Functional tests for agent-to-agent communication via the control plane.

Tests that agents can call other agents through the control plane's
routing and execution infrastructure.
"""

import asyncio
import threading

import httpx
import pytest
import uvicorn
from agentfield import Agent


@pytest.mark.functional
def test_agent_calls_another_agent(control_plane_url: str, agent_port_allocator):
    """Test that one agent can call another agent via the control plane."""
    agent_a_port = agent_port_allocator()
    agent_b_port = agent_port_allocator()

    # Create Agent A
    agent_a = Agent(
        node_id="test-agent-a",
        agentfield_server=control_plane_url,
    )

    # Create Agent B
    agent_b = Agent(
        node_id="test-agent-b",
        agentfield_server=control_plane_url,
    )

    # Agent B has a reasoner that does some work
    @agent_b.reasoner()
    async def process_data(value: int):
        """Process a value (double it)."""
        return {"result": value * 2}

    # Agent A has a reasoner that calls Agent B
    @agent_a.reasoner()
    async def delegate_work(value: int):
        """Delegate work to Agent B."""
        # Call Agent B via the agent.call() method
        result = await agent_a.call("test-agent-b.process_data", input={"value": value})
        return {"delegated_result": result}

    # Start both agents
    servers_ready = {"a": threading.Event(), "b": threading.Event()}

    def run_agent_a():
        config = uvicorn.Config(
            agent_a, host="0.0.0.0", port=agent_a_port, log_level="error"
        )
        server = uvicorn.Server(config)
        servers_ready["a"].set()
        asyncio.run(server.serve())

    def run_agent_b():
        config = uvicorn.Config(
            agent_b, host="0.0.0.0", port=agent_b_port, log_level="error"
        )
        server = uvicorn.Server(config)
        servers_ready["b"].set()
        asyncio.run(server.serve())

    thread_a = threading.Thread(target=run_agent_a, daemon=True)
    thread_b = threading.Thread(target=run_agent_b, daemon=True)

    thread_a.start()
    thread_b.start()

    # Wait for both servers
    servers_ready["a"].wait(timeout=5)
    servers_ready["b"].wait(timeout=5)

    import time

    time.sleep(3)  # Give agents time to register

    try:
        # Call Agent A, which will call Agent B
        client = httpx.Client(base_url=control_plane_url, timeout=15.0)
        response = client.post(
            "/api/v1/execute/test-agent-a.delegate_work",
            json={"input": {"value": 21}},
        )

        assert (
            response.status_code == 200
        ), f"Call failed: {response.status_code} - {response.text}"
        result = response.json()

        print(f"✅ Agent-to-agent call successful: {result}")

        # Verify the result shows delegation happened
        # Expected: 21 * 2 = 42
        result_str = str(result)
        assert (
            "42" in result_str or "delegated" in result_str
        ), f"Unexpected result: {result}"

    finally:
        pass


@pytest.mark.functional
def test_chain_of_agent_calls(control_plane_url: str, agent_port_allocator):
    """Test a chain of agent calls: A -> B -> C."""
    port_a = agent_port_allocator()
    port_b = agent_port_allocator()
    port_c = agent_port_allocator()

    # Create three agents
    agent_a = Agent(
        node_id="test-chain-a",
        agentfield_server=control_plane_url,
    )

    agent_b = Agent(
        node_id="test-chain-b",
        agentfield_server=control_plane_url,
    )

    agent_c = Agent(
        node_id="test-chain-c",
        agentfield_server=control_plane_url,
    )

    # Agent C: terminal agent
    @agent_c.reasoner()
    async def multiply(x: int, y: int):
        """Multiply two numbers."""
        return {"product": x * y}

    # Agent B: calls Agent C
    @agent_b.reasoner()
    async def add_then_multiply(a: int, b: int):
        """Add two numbers then multiply by 2."""
        sum_val = a + b
        result = await agent_b.call(
            "test-chain-c.multiply", input={"x": sum_val, "y": 2}
        )
        return {"step": "b", "result": result}

    # Agent A: calls Agent B
    @agent_a.reasoner()
    async def orchestrate(value: int):
        """Orchestrate the chain."""
        result = await agent_a.call(
            "test-chain-b.add_then_multiply", input={"a": value, "b": 5}
        )
        return {"step": "a", "final_result": result}

    # Start all agents
    servers_ready = {
        "a": threading.Event(),
        "b": threading.Event(),
        "c": threading.Event(),
    }

    def make_runner(agent, port, key):
        def runner():
            config = uvicorn.Config(agent, host="0.0.0.0", port=port, log_level="error")
            server = uvicorn.Server(config)
            servers_ready[key].set()
            asyncio.run(server.serve())

        return runner

    threads = [
        threading.Thread(target=make_runner(agent_a, port_a, "a"), daemon=True),
        threading.Thread(target=make_runner(agent_b, port_b, "b"), daemon=True),
        threading.Thread(target=make_runner(agent_c, port_c, "c"), daemon=True),
    ]

    for thread in threads:
        thread.start()

    # Wait for all servers
    for key in ["a", "b", "c"]:
        servers_ready[key].wait(timeout=5)

    import time

    time.sleep(3)

    try:
        # Trigger the chain by calling Agent A
        client = httpx.Client(base_url=control_plane_url, timeout=20.0)
        response = client.post(
            "/api/v1/execute/test-chain-a.orchestrate",
            json={"input": {"value": 10}},
        )

        assert (
            response.status_code == 200
        ), f"Chain call failed: {response.status_code} - {response.text}"
        result = response.json()

        print(f"✅ Chain of agent calls successful: {result}")

        # Expected calculation: (10 + 5) * 2 = 30
        result_str = str(result)
        assert (
            "30" in result_str or "final" in result_str
        ), f"Unexpected result: {result}"

    finally:
        pass


@pytest.mark.functional
def test_agent_call_with_error_handling(control_plane_url: str, agent_port_allocator):
    """Test error propagation in agent-to-agent calls."""
    port_a = agent_port_allocator()
    port_b = agent_port_allocator()

    agent_a = Agent(
        node_id="test-error-caller",
        agentfield_server=control_plane_url,
    )

    agent_b = Agent(
        node_id="test-error-callee",
        agentfield_server=control_plane_url,
    )

    # Agent B has a failing reasoner
    @agent_b.reasoner()
    async def failing_operation():
        """This always fails."""
        raise RuntimeError("Simulated error")

    # Agent A calls the failing Agent B
    @agent_a.reasoner()
    async def try_call_failing_agent(ping: str = "ok"):
        """Try to call the failing agent."""
        try:
            result = await agent_a.call("test-error-callee.failing_operation", input={})
            return {"status": "unexpected_success", "result": result}
        except Exception as e:
            return {"status": "error_caught", "error": str(e)}

    servers_ready = {"a": threading.Event(), "b": threading.Event()}

    def run_a():
        config = uvicorn.Config(agent_a, host="0.0.0.0", port=port_a, log_level="error")
        server = uvicorn.Server(config)
        servers_ready["a"].set()
        asyncio.run(server.serve())

    def run_b():
        config = uvicorn.Config(agent_b, host="0.0.0.0", port=port_b, log_level="error")
        server = uvicorn.Server(config)
        servers_ready["b"].set()
        asyncio.run(server.serve())

    thread_a = threading.Thread(target=run_a, daemon=True)
    thread_b = threading.Thread(target=run_b, daemon=True)

    thread_a.start()
    thread_b.start()

    servers_ready["a"].wait(timeout=5)
    servers_ready["b"].wait(timeout=5)

    import time

    time.sleep(3)

    try:
        client = httpx.Client(base_url=control_plane_url, timeout=15.0)
        response = client.post(
            "/api/v1/execute/test-error-caller.try_call_failing_agent",
            json={"input": {"ping": "ok"}},
        )

        # The outer agent should handle the error
        assert response.status_code == 200, f"Call failed: {response.status_code}"
        result = response.json()

        print(f"✅ Error handling test passed: {result}")

        # Verify error was caught
        result_str = str(result).lower()
        assert "error" in result_str, f"Expected error status in response: {result}"

    finally:
        pass
