"""Tests for reasoner path normalization.

When a positional argument is passed to @agent.reasoner(), it sets the endpoint
path, not the name. The control plane always forwards requests to
/reasoners/{reasoner_id}, so the SDK must normalize custom paths to include the
/reasoners/ prefix. Otherwise the agent returns 404 on forwarded requests.
"""

import httpx
import pytest

from agentfield.router import AgentRouter

from tests.helpers import create_test_agent


@pytest.mark.asyncio
async def test_positional_path_normalized_to_reasoners_prefix(monkeypatch):
    """@agent.reasoner("call_b") should register endpoint at /reasoners/call_b."""
    agent, _ = create_test_agent(monkeypatch)
    agent.async_config.enable_async_execution = False
    agent.agentfield_server = None

    @agent.reasoner("call_b")
    async def call_b(input: str) -> dict:
        return {"echo": input}

    # Verify the reasoner is registered with the correct ID
    assert any(r["id"] == "call_b" for r in agent.reasoners)

    # The key assertion: endpoint must be reachable at /reasoners/call_b
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=agent), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/reasoners/call_b",
            json={"input": "hello"},
            headers={"x-workflow-id": "wf-1", "x-execution-id": "exec-1"},
        )

    assert resp.status_code == 200
    assert resp.json() == {"echo": "hello"}


@pytest.mark.asyncio
async def test_positional_path_with_leading_slash(monkeypatch):
    """@agent.reasoner("/my_func") should normalize to /reasoners/my_func."""
    agent, _ = create_test_agent(monkeypatch)
    agent.async_config.enable_async_execution = False
    agent.agentfield_server = None

    @agent.reasoner("/my_func")
    async def my_func(value: int) -> dict:
        return {"value": value}

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=agent), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/reasoners/my_func",
            json={"value": 42},
            headers={"x-workflow-id": "wf-2", "x-execution-id": "exec-2"},
        )

    assert resp.status_code == 200
    assert resp.json() == {"value": 42}


@pytest.mark.asyncio
async def test_positional_path_already_prefixed(monkeypatch):
    """@agent.reasoner("/reasoners/foo") should remain unchanged."""
    agent, _ = create_test_agent(monkeypatch)
    agent.async_config.enable_async_execution = False
    agent.agentfield_server = None

    @agent.reasoner("/reasoners/foo")
    async def foo(x: int) -> dict:
        return {"x": x}

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=agent), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/reasoners/foo",
            json={"x": 1},
            headers={"x-workflow-id": "wf-3", "x-execution-id": "exec-3"},
        )

    assert resp.status_code == 200
    assert resp.json() == {"x": 1}


@pytest.mark.asyncio
async def test_dynamic_positional_path(monkeypatch):
    """@agent.reasoner(f"handler-{i}") should normalize to /reasoners/handler-{i}."""
    agent, _ = create_test_agent(monkeypatch)
    agent.async_config.enable_async_execution = False
    agent.agentfield_server = None

    for i in range(3):
        idx = i

        @agent.reasoner(f"handler-{i}")
        async def handler(input_data: dict, _idx=idx) -> dict:
            return {"id": _idx}

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=agent), base_url="http://test"
    ) as client:
        for i in range(3):
            resp = await client.post(
                f"/reasoners/handler-{i}",
                json={"input_data": {}},
                headers={"x-workflow-id": "wf-dyn", "x-execution-id": f"exec-dyn-{i}"},
            )
            assert resp.status_code == 200, f"handler-{i} not reachable at /reasoners/handler-{i}"


@pytest.mark.asyncio
async def test_default_path_unchanged(monkeypatch):
    """@agent.reasoner() should still register at /reasoners/{func_name}."""
    agent, _ = create_test_agent(monkeypatch)
    agent.async_config.enable_async_execution = False
    agent.agentfield_server = None

    @agent.reasoner()
    async def compute(value: int) -> dict:
        return {"result": value * 2}

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=agent), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/reasoners/compute",
            json={"value": 5},
            headers={"x-workflow-id": "wf-def", "x-execution-id": "exec-def"},
        )

    assert resp.status_code == 200
    assert resp.json() == {"result": 10}


@pytest.mark.asyncio
async def test_bare_decorator_unchanged(monkeypatch):
    """@agent.reasoner (no parens) should still register at /reasoners/{func_name}."""
    agent, _ = create_test_agent(monkeypatch)
    agent.async_config.enable_async_execution = False
    agent.agentfield_server = None

    @agent.reasoner
    async def ping(msg: str) -> dict:
        return {"pong": msg}

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=agent), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/reasoners/ping",
            json={"msg": "hi"},
            headers={"x-workflow-id": "wf-bare", "x-execution-id": "exec-bare"},
        )

    assert resp.status_code == 200
    assert resp.json() == {"pong": "hi"}


@pytest.mark.asyncio
async def test_router_positional_path_normalized(monkeypatch):
    """Router reasoner with positional path should also normalize."""
    agent, _ = create_test_agent(monkeypatch)
    agent.async_config.enable_async_execution = False
    agent.agentfield_server = None

    router = AgentRouter()

    @router.reasoner("process")
    async def process(data: str) -> dict:
        return {"processed": data}

    agent.include_router(router)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=agent), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/reasoners/process",
            json={"data": "test"},
            headers={"x-workflow-id": "wf-router", "x-execution-id": "exec-router"},
        )

    assert resp.status_code == 200
    assert resp.json() == {"processed": "test"}
