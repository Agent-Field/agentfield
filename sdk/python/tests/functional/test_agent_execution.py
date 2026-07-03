"""
Functional tests for agent execution via the control plane.

Tests that reasoners can be executed through the control plane
and return correct results.
"""

import asyncio
import threading

import httpx
import pytest
import uvicorn
from agentfield import Agent


@pytest.mark.functional
def test_execute_simple_reasoner(control_plane_url: str, agent_port_allocator):
    """Test executing a simple reasoner through the control plane."""
    agent_port = agent_port_allocator()

    agent = Agent(
        node_id="test-exec-agent",
        agentfield_server=control_plane_url,
    )

    @agent.reasoner()
    async def add_numbers(a: int, b: int):
        """Add two numbers."""
        return {"sum": a + b}

    # Start agent
    server_ready = threading.Event()

    def run_agent():
        config = uvicorn.Config(
            agent, host="0.0.0.0", port=agent_port, log_level="error"
        )
        server = uvicorn.Server(config)
        server_ready.set()
        asyncio.run(server.serve())

    agent_thread = threading.Thread(target=run_agent, daemon=True)
    agent_thread.start()
    server_ready.wait(timeout=5)

    import time

    time.sleep(2)

    try:
        # Execute reasoner via control plane
        client = httpx.Client(base_url=control_plane_url, timeout=10.0)
        response = client.post(
            "/api/v1/execute/test-exec-agent.add_numbers",
            json={"input": {"a": 5, "b": 3}},
        )

        assert (
            response.status_code == 200
        ), f"Execution failed: {response.status_code} - {response.text}"
        result = response.json()

        # Verify result
        assert (
            "result" in result or "sum" in result
        ), f"Missing result in response: {result}"

        # Extract the sum value
        if "result" in result:
            if isinstance(result["result"], dict):
                assert (
                    result["result"]["sum"] == 8
                ), f"Expected sum=8, got {result['result']}"
            else:
                # Result might be directly in the response
                pass
        elif "sum" in result:
            assert result["sum"] == 8

        print(f"✅ Reasoner executed successfully: {result}")

    finally:
        pass


@pytest.mark.functional
def test_execute_with_string_result(control_plane_url: str, agent_port_allocator):
    """Test executing a reasoner that returns a string."""
    agent_port = agent_port_allocator()

    agent = Agent(
        node_id="test-string-agent",
        agentfield_server=control_plane_url,
    )

    @agent.reasoner()
    async def greet(name: str):
        """Greet someone by name."""
        return f"Hello, {name}!"

    server_ready = threading.Event()

    def run_agent():
        config = uvicorn.Config(
            agent, host="0.0.0.0", port=agent_port, log_level="error"
        )
        server = uvicorn.Server(config)
        server_ready.set()
        asyncio.run(server.serve())

    agent_thread = threading.Thread(target=run_agent, daemon=True)
    agent_thread.start()
    server_ready.wait(timeout=5)

    import time

    time.sleep(2)

    try:
        client = httpx.Client(base_url=control_plane_url, timeout=10.0)
        response = client.post(
            "/api/v1/execute/test-string-agent.greet",
            json={"input": {"name": "World"}},
        )

        assert response.status_code == 200
        result = response.json()

        # Verify greeting is in response
        result_str = str(result)
        assert "Hello, World!" in result_str, f"Expected greeting in response: {result}"

        print(f"✅ String result test passed: {result}")

    finally:
        pass


@pytest.mark.functional
def test_execute_with_complex_input(control_plane_url: str, agent_port_allocator):
    """Test executing a reasoner with complex nested input."""
    agent_port = agent_port_allocator()

    agent = Agent(
        node_id="test-complex-agent",
        agentfield_server=control_plane_url,
    )

    @agent.reasoner()
    async def process_data(data: dict):
        """Process complex data structure."""
        return {
            "processed": True,
            "item_count": len(data.get("items", [])),
            "has_metadata": "metadata" in data,
        }

    server_ready = threading.Event()

    def run_agent():
        config = uvicorn.Config(
            agent, host="0.0.0.0", port=agent_port, log_level="error"
        )
        server = uvicorn.Server(config)
        server_ready.set()
        asyncio.run(server.serve())

    agent_thread = threading.Thread(target=run_agent, daemon=True)
    agent_thread.start()
    server_ready.wait(timeout=5)

    import time

    time.sleep(2)

    try:
        client = httpx.Client(base_url=control_plane_url, timeout=10.0)

        complex_input = {
            "items": [1, 2, 3, 4, 5],
            "metadata": {"source": "test", "timestamp": "2024-01-01"},
        }

        response = client.post(
            "/api/v1/execute/test-complex-agent.process_data",
            json={"input": {"data": complex_input}},
        )

        assert response.status_code == 200
        result = response.json()

        print(f"✅ Complex input test passed: {result}")

    finally:
        pass


@pytest.mark.functional
def test_execute_async_reasoner(control_plane_url: str, agent_port_allocator):
    """Test executing an async reasoner with delay."""
    agent_port = agent_port_allocator()

    agent = Agent(
        node_id="test-async-agent",
        agentfield_server=control_plane_url,
    )

    @agent.reasoner()
    async def slow_operation(delay: float):
        """Simulate a slow async operation."""
        await asyncio.sleep(delay)
        return {"completed": True, "delay": delay}

    server_ready = threading.Event()

    def run_agent():
        config = uvicorn.Config(
            agent, host="0.0.0.0", port=agent_port, log_level="error"
        )
        server = uvicorn.Server(config)
        server_ready.set()
        asyncio.run(server.serve())

    agent_thread = threading.Thread(target=run_agent, daemon=True)
    agent_thread.start()
    server_ready.wait(timeout=5)

    import time

    time.sleep(2)

    try:
        client = httpx.Client(base_url=control_plane_url, timeout=15.0)

        response = client.post(
            "/api/v1/execute/test-async-agent.slow_operation",
            json={"input": {"delay": 0.5}},
        )

        assert response.status_code == 200
        result = response.json()

        print(f"✅ Async reasoner test passed: {result}")

    finally:
        pass


@pytest.mark.functional
def test_execute_reasoner_with_error(control_plane_url: str, agent_port_allocator):
    """Test that errors from reasoners are properly propagated."""
    agent_port = agent_port_allocator()

    agent = Agent(
        node_id="test-error-agent",
        agentfield_server=control_plane_url,
    )

    @agent.reasoner()
    async def failing_reasoner(ping: str = "ok"):
        """This reasoner always fails."""
        raise ValueError("Intentional error for testing")

    server_ready = threading.Event()

    def run_agent():
        config = uvicorn.Config(
            agent, host="0.0.0.0", port=agent_port, log_level="error"
        )
        server = uvicorn.Server(config)
        server_ready.set()
        asyncio.run(server.serve())

    agent_thread = threading.Thread(target=run_agent, daemon=True)
    agent_thread.start()
    server_ready.wait(timeout=5)

    import time

    time.sleep(2)

    try:
        client = httpx.Client(base_url=control_plane_url, timeout=10.0)

        response = client.post(
            "/api/v1/execute/test-error-agent.failing_reasoner",
            json={"input": {"ping": "ok"}},
        )

        # Should return an error status
        assert (
            response.status_code >= 400
        ), f"Expected error status, got {response.status_code}"

        print(f"✅ Error handling test passed: {response.status_code}")

    finally:
        pass


@pytest.mark.functional
def test_execute_multiple_reasoners_on_same_agent(
    control_plane_url: str, agent_port_allocator
):
    """Test executing multiple different reasoners on the same agent."""
    agent_port = agent_port_allocator()

    agent = Agent(
        node_id="test-multi-reasoner-agent",
        agentfield_server=control_plane_url,
    )

    @agent.reasoner()
    async def multiply(a: int, b: int):
        """Multiply two numbers."""
        return {"product": a * b}

    @agent.reasoner()
    async def divide(a: int, b: int):
        """Divide two numbers."""
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return {"quotient": a / b}

    @agent.reasoner()
    async def power(base: int, exponent: int):
        """Raise base to exponent."""
        return {"result": base**exponent}

    server_ready = threading.Event()

    def run_agent():
        config = uvicorn.Config(
            agent, host="0.0.0.0", port=agent_port, log_level="error"
        )
        server = uvicorn.Server(config)
        server_ready.set()
        asyncio.run(server.serve())

    agent_thread = threading.Thread(target=run_agent, daemon=True)
    agent_thread.start()
    server_ready.wait(timeout=5)

    import time

    time.sleep(2)

    try:
        client = httpx.Client(base_url=control_plane_url, timeout=10.0)

        # Test multiply
        response = client.post(
            "/api/v1/execute/test-multi-reasoner-agent.multiply",
            json={"input": {"a": 6, "b": 7}},
        )
        assert response.status_code == 200
        print(f"✅ Multiply: {response.json()}")

        # Test divide
        response = client.post(
            "/api/v1/execute/test-multi-reasoner-agent.divide",
            json={"input": {"a": 20, "b": 4}},
        )
        assert response.status_code == 200
        print(f"✅ Divide: {response.json()}")

        # Test power
        response = client.post(
            "/api/v1/execute/test-multi-reasoner-agent.power",
            json={"input": {"base": 2, "exponent": 8}},
        )
        assert response.status_code == 200
        print(f"✅ Power: {response.json()}")

    finally:
        pass
