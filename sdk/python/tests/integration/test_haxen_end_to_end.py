import asyncio
from typing import Any, Dict

import httpx
import pytest

from haxen_sdk.agent import Agent
from haxen_sdk.types import AgentStatus


async def _wait_for_node(
    client: httpx.AsyncClient, node_id: str, attempts: int = 40
) -> Dict[str, Any]:
    for _ in range(attempts):
        response = await client.get(f"/api/v1/nodes/{node_id}")
        if response.status_code == 200:
            payload = response.json()
            if payload.get("id") == node_id:
                return payload
        await asyncio.sleep(0.5)
    raise AssertionError(f"Node {node_id} did not appear in Haxen registry")


async def _wait_for_status(
    client: httpx.AsyncClient,
    node_id: str,
    expected: str,
    attempts: int = 40,
) -> Dict[str, Any]:
    for _ in range(attempts):
        response = await client.get(f"/api/v1/nodes/{node_id}/status")
        if response.status_code == 200:
            data = response.json().get("status", {})
            lifecycle = data.get("lifecycle_status")
            if lifecycle == expected:
                return data
        await asyncio.sleep(0.5)
    raise AssertionError(f"Status for {node_id} never reached '{expected}'")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_agent_registration_and_status_propagation(haxen_server, run_agent):
    agent = Agent(
        node_id="integration-agent-status",
        haxen_server=haxen_server.base_url,
        dev_mode=True,
        callback_url="http://127.0.0.1",
    )

    @agent.reasoner()
    async def ping() -> Dict[str, bool]:
        return {"ok": True}

    runtime = run_agent(agent)

    await agent.haxen_handler.register_with_haxen_server(runtime.port)
    assert agent.haxen_connected is True

    async with httpx.AsyncClient(base_url=haxen_server.base_url, timeout=5.0) as client:
        node = await _wait_for_node(client, agent.node_id)
        assert any(r["id"] == "ping" for r in node.get("reasoners", []))

        agent._current_status = AgentStatus.READY
        await agent.haxen_handler.send_enhanced_heartbeat()

        status = await _wait_for_status(client, agent.node_id, expected="ready")
        assert status.get("state") == "active"
        assert status.get("health_score", 0) >= 60


@pytest.mark.integration
@pytest.mark.asyncio
async def test_reasoner_execution_roundtrip(haxen_server, run_agent):
    agent = Agent(
        node_id="integration-agent-reasoner",
        haxen_server=haxen_server.base_url,
        dev_mode=True,
        callback_url="http://127.0.0.1",
    )

    @agent.reasoner()
    async def double(value: int) -> Dict[str, int]:
        return {"value": value * 2}

    runtime = run_agent(agent)

    await agent.haxen_handler.register_with_haxen_server(runtime.port)
    agent._current_status = AgentStatus.READY
    await agent.haxen_handler.send_enhanced_heartbeat()

    async with httpx.AsyncClient(base_url=haxen_server.base_url, timeout=5.0) as client:
        await _wait_for_node(client, agent.node_id)
        await _wait_for_status(client, agent.node_id, expected="ready")

        response = await client.post(
            f"/api/v1/reasoners/{agent.node_id}.double",
            json={"input": {"value": 7}},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["node_id"] == agent.node_id
    assert payload["result"]["value"] == 14
    assert payload["duration_ms"] >= 0
    assert "X-Workflow-ID" in response.headers
    assert "X-Execution-ID" in response.headers
