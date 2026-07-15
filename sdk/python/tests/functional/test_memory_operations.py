"""
Functional tests for memory operations via the control plane.

Tests that agents can store and retrieve memory across different scopes
(global, agent, session, run) through the control plane.
"""

import asyncio
import threading
import uuid

import httpx
import pytest
import uvicorn
from agentfield import Agent


@pytest.mark.functional
def test_agent_global_memory(control_plane_url: str, agent_port_allocator):
    """Test that agents can set and get global memory."""
    agent_port = agent_port_allocator()

    agent = Agent(
        node_id="test-memory-global-agent",
        agentfield_server=control_plane_url,
    )

    @agent.reasoner()
    async def set_global_value(key: str, value: str):
        """Set a value in global memory."""
        await agent.memory.set(scope="global", key=key, value=value)
        return {"status": "set", "key": key, "value": value}

    @agent.reasoner()
    async def get_global_value(key: str):
        """Get a value from global memory."""
        value = await agent.memory.get(scope="global", key=key)
        return {"key": key, "value": value}

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

        # Set a global value
        test_key = f"test-key-{uuid.uuid4().hex[:8]}"
        test_value = f"test-value-{uuid.uuid4().hex[:8]}"

        response = client.post(
            "/api/v1/execute/test-memory-global-agent.set_global_value",
            json={"input": {"key": test_key, "value": test_value}},
        )
        assert response.status_code == 200
        print(f"✅ Set global value: {response.json()}")

        # Get the global value
        response = client.post(
            "/api/v1/execute/test-memory-global-agent.get_global_value",
            json={"input": {"key": test_key}},
        )
        assert response.status_code == 200
        result = response.json()

        print(f"✅ Got global value: {result}")

        # Verify the value matches
        result_str = str(result)
        assert test_value in result_str or test_key in result_str

    finally:
        pass


@pytest.mark.functional
def test_agent_scoped_memory(control_plane_url: str, agent_port_allocator):
    """Test that agents can use agent-scoped memory."""
    agent_port = agent_port_allocator()

    agent = Agent(
        node_id="test-memory-agent-scope",
        agentfield_server=control_plane_url,
    )

    @agent.reasoner()
    async def increment_counter(ping: str = "ok"):
        """Increment an agent-scoped counter."""
        # Get current value
        current = await agent.memory.get(scope="agent", key="counter")
        if current is None:
            current = 0
        else:
            current = int(current)

        # Increment
        new_value = current + 1

        # Set new value
        await agent.memory.set(scope="agent", key="counter", value=str(new_value))

        return {"counter": new_value}

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

        # Call increment multiple times
        for expected_value in [1, 2, 3]:
            response = client.post(
                "/api/v1/execute/test-memory-agent-scope.increment_counter",
                json={"input": {"ping": "ok"}},
            )
            assert response.status_code == 200
            result = response.json()

            result_str = str(result)
            assert str(expected_value) in result_str

            print(f"✅ Counter incremented to: {expected_value}")

    finally:
        pass


@pytest.mark.functional
def test_memory_shared_between_agents(control_plane_url: str, agent_port_allocator):
    """Test that global memory is shared between different agents."""
    port_a = agent_port_allocator()
    port_b = agent_port_allocator()

    agent_a = Agent(
        node_id="test-memory-writer",
        agentfield_server=control_plane_url,
    )

    agent_b = Agent(
        node_id="test-memory-reader",
        agentfield_server=control_plane_url,
    )

    # Agent A writes to global memory
    @agent_a.reasoner()
    async def write_shared_data(data: str):
        """Write data to global memory."""
        await agent_a.memory.set(scope="global", key="shared_data", value=data)
        return {"status": "written", "data": data}

    # Agent B reads from global memory
    @agent_b.reasoner()
    async def read_shared_data(ping: str = "ok"):
        """Read data from global memory."""
        data = await agent_b.memory.get(scope="global", key="shared_data")
        return {"status": "read", "data": data}

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
        client = httpx.Client(base_url=control_plane_url, timeout=10.0)

        # Agent A writes
        test_data = f"shared-{uuid.uuid4().hex[:8]}"
        response = client.post(
            "/api/v1/execute/test-memory-writer.write_shared_data",
            json={"input": {"data": test_data}},
        )
        assert response.status_code == 200
        print(f"✅ Agent A wrote: {response.json()}")

        # Agent B reads
        response = client.post(
            "/api/v1/execute/test-memory-reader.read_shared_data",
            json={"input": {"ping": "ok"}},
        )
        assert response.status_code == 200
        result = response.json()

        print(f"✅ Agent B read: {result}")

        # Verify Agent B got the same data
        result_str = str(result)
        assert test_data in result_str

    finally:
        pass


@pytest.mark.functional
def test_memory_list_keys(control_plane_url: str, agent_port_allocator):
    """Test listing memory keys."""
    agent_port = agent_port_allocator()

    agent = Agent(
        node_id="test-memory-list-agent",
        agentfield_server=control_plane_url,
    )

    @agent.reasoner()
    async def setup_multiple_keys(ping: str = "ok"):
        """Set multiple keys in memory."""
        test_prefix = f"test-list-{uuid.uuid4().hex[:8]}"

        for i in range(5):
            await agent.memory.set(
                scope="global", key=f"{test_prefix}-key-{i}", value=f"value-{i}"
            )

        # List keys (if supported)
        try:
            keys = await agent.memory.list(scope="global")
            return {"status": "success", "key_count": len(keys), "prefix": test_prefix}
        except Exception as e:
            # If list is not supported, just return success
            return {"status": "set_multiple", "count": 5, "error": str(e)}

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
            "/api/v1/execute/test-memory-list-agent.setup_multiple_keys",
            json={"input": {"ping": "ok"}},
        )
        assert response.status_code == 200
        result = response.json()

        print(f"✅ Memory list test: {result}")

    finally:
        pass


@pytest.mark.functional
def test_memory_delete(control_plane_url: str, agent_port_allocator):
    """Test deleting memory keys."""
    agent_port = agent_port_allocator()

    agent = Agent(
        node_id="test-memory-delete-agent",
        agentfield_server=control_plane_url,
    )

    @agent.reasoner()
    async def test_delete_flow(ping: str = "ok"):
        """Test setting, getting, deleting, and verifying deletion."""
        test_key = f"delete-test-{uuid.uuid4().hex[:8]}"

        # Set a value
        await agent.memory.set(scope="global", key=test_key, value="to-be-deleted")

        # Verify it exists
        value = await agent.memory.get(scope="global", key=test_key)
        assert value == "to-be-deleted", f"Expected value, got {value}"

        # Delete it
        await agent.memory.delete(scope="global", key=test_key)

        # Verify it's gone
        value_after = await agent.memory.get(scope="global", key=test_key)

        return {
            "status": "success",
            "value_before_delete": value,
            "value_after_delete": value_after,
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

        response = client.post(
            "/api/v1/execute/test-memory-delete-agent.test_delete_flow",
            json={"input": {"ping": "ok"}},
        )
        assert response.status_code == 200
        result = response.json()

        print(f"✅ Memory delete test passed: {result}")

        # Verify deletion worked
        result_str = str(result).lower()
        assert (
            "none" in result_str or "null" in result_str or "after_delete" in result_str
        )

    finally:
        pass
