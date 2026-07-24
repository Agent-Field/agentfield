import asyncio
import pytest
import httpx
import time
from agentfield.agent import Agent


# Mock the client to avoid network calls
@pytest.fixture
def mock_client(monkeypatch):
    class MockClient:
        def __init__(self, *args, **kwargs):
            pass

        async def _async_request(self, *args, **kwargs):
            class MockResponse:
                status_code = 200

                def json(self):
                    return {}

            return MockResponse()

        async def notify_graceful_shutdown_sync(self, *args, **kwargs):
            return True

        async def post_execution_logs(self, execution_id, entries):
            pass

        async def aclose(self):
            pass

    monkeypatch.setattr("agentfield.agent.AgentFieldClient", MockClient)
    monkeypatch.setattr("agentfield.client.AgentFieldClient", MockClient)

    # Also mock AgentUtils.is_port_available to avoid binding issues
    monkeypatch.setattr(
        "agentfield.agent_utils.AgentUtils.is_port_available", lambda p: True
    )


@pytest.fixture
def resilient_agent(mock_client):
    agent = Agent(
        node_id="resilient-agent",
        agentfield_server="http://mock-control-plane",
        auto_register=False,
        dev_mode=True,
        async_config=None,  # Use defaults
    )
    return agent


@pytest.mark.asyncio
async def test_concurrent_execution_resilience(resilient_agent):
    """Test that the agent can handle multiple concurrent requests without blocking."""

    @resilient_agent.reasoner()
    async def slow_echo(value: int) -> dict:
        await asyncio.sleep(0.1)
        return {"value": value}

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=resilient_agent), base_url="http://test"
    ) as client:
        # Fire 50 requests concurrently
        tasks = []
        num_requests = 50
        for i in range(num_requests):
            tasks.append(client.post("/reasoners/slow_echo", json={"value": i}))

        start_time = time.time()
        responses = await asyncio.gather(*tasks)
        end_time = time.time()

        # Check results
        for i, response in enumerate(responses):
            assert response.status_code == 200, f"Request {i} failed: {response.text}"
            data = response.json()
            assert data["value"] == i

        # Ensure it didn't take 50 * 0.1 = 5 seconds (sequential)
        # It should be closer to 0.1s + overhead.
        # Giving it a generous 2.0s buffer for test env overhead.
        duration = end_time - start_time
        print(f"Duration for {num_requests} requests: {duration}s")
        assert duration < 2.0, f"Concurrency check failed, took {duration}s"


@pytest.mark.asyncio
async def test_error_containment(resilient_agent):
    """Test that a crashing reasoner doesn't bring down the agent."""

    @resilient_agent.reasoner()
    async def crasher():
        raise ValueError("Boom!")

    @resilient_agent.reasoner()
    async def ping():
        return {"status": "ok"}

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=resilient_agent, raise_app_exceptions=False),
        base_url="http://test",
    ) as client:
        # 1. Call the crashing endpoint
        # FastAPI usually catches exceptions and returns 500
        response = await client.post("/reasoners/crasher", json={})
        assert response.status_code == 500

        # 2. Ensure the agent is still responsive
        resp2 = await client.post("/reasoners/ping", json={})
        assert resp2.status_code == 200
        assert resp2.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_input_validation_resilience(resilient_agent):
    """Test that malformed inputs are handled gracefully."""

    @resilient_agent.reasoner()
    async def typed_input(x: int, y: str) -> dict:
        return {"result": f"{y}-{x}"}

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=resilient_agent), base_url="http://test"
    ) as client:
        # 1. Missing fields
        resp = await client.post("/reasoners/typed_input", json={"x": 1})
        assert resp.status_code == 422  # Validation error

        # 2. Wrong types
        resp = await client.post(
            "/reasoners/typed_input", json={"x": "not-int", "y": "ok"}
        )
        assert resp.status_code == 422

        # 3. Malformed JSON
        resp = await client.post(
            "/reasoners/typed_input",
            content="{ bad json }",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 400 or resp.status_code == 422
