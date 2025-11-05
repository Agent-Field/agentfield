import asyncio

import httpx
import pytest
from fastapi import APIRouter

from agentfield.router import AgentRouter

from tests.helpers import create_test_agent


@pytest.mark.asyncio
async def test_agent_reasoner_routing_and_workflow(monkeypatch):
    agent, agentfield_client = create_test_agent(
        monkeypatch, callback_url="https://callback.example.com"
    )

    @agent.reasoner()
    async def double(value: int) -> dict:
        memory = agent.memory
        memory_present = memory is not None
        fetched = await memory.get("last", default="missing") if memory else "missing"
        return {
            "value": value * 2,
            "memory_present": memory_present,
            "memory_value": fetched,
        }

    @agent.skill()
    def annotate(text: str) -> str:
        return f"annotated:{text}"

    router = APIRouter()

    @router.get("/status")
    async def status():
        return {"node": agent.node_id}

    agent.include_router(router, prefix="/ops")

    await agent.agentfield_handler.register_with_agentfield_server(port=9100)
    assert agentfield_client.register_calls
    registration = agentfield_client.register_calls[-1]
    assert registration["base_url"] == "https://callback.example.com:9100"
    assert registration["reasoners"][0]["id"] == "double"
    assert registration["skills"][0]["id"] == "annotate"

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=agent), base_url="http://test"
    ) as client:
        reasoner_resp = await client.post(
            "/reasoners/double",
            json={"value": 3},
            headers={"x-workflow-id": "wf-123", "x-execution-id": "exec-root"},
        )
        router_resp = await client.get("/ops/status")

    assert reasoner_resp.status_code == 200
    data = reasoner_resp.json()
    assert data["value"] == 6
    assert data["memory_present"] is True
    assert data["memory_value"] == "missing"

    assert router_resp.status_code == 200
    assert router_resp.json() == {"node": agent.node_id}

    await asyncio.sleep(0)
    events = getattr(agent, "_captured_workflow_events", [])
    assert ("start", "exec-root", "double", None) in events
    assert any(evt[0] == "complete" and evt[2] == "double" for evt in events)


@pytest.mark.asyncio
async def test_agent_reasoner_custom_name(monkeypatch):
    agent, _ = create_test_agent(monkeypatch)

    @agent.reasoner(name="reports_generate")
    async def generate_report(report_id: str) -> dict:
        return {"report_id": report_id}

    assert any(r["id"] == "reports_generate" for r in agent.reasoners)
    assert "reports_generate" in agent._reasoner_return_types
    assert hasattr(agent, "reports_generate")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=agent), base_url="http://test"
    ) as client:
        response = await client.post(
            "/reasoners/generate_report",
            json={"report_id": "r-123"},
            headers={
                "x-workflow-id": "wf-custom",
                "x-execution-id": "exec-custom",
            },
        )

    assert response.status_code == 200
    assert response.json() == {"report_id": "r-123"}


@pytest.mark.asyncio
async def test_agent_router_prefix_registration(monkeypatch):
    agent, _ = create_test_agent(monkeypatch)

    quickstart = AgentRouter(prefix="demo")

    @quickstart.reasoner()
    async def hello(name: str) -> dict:
        return {"message": f"hello {name}"}

    @quickstart.skill()
    def repeat(text: str) -> dict:
        return {"echo": text}

    agent.include_router(quickstart)

    assert any(r["id"] == "demo_hello" for r in agent.reasoners)
    assert any(s["id"] == "demo_repeat" for s in agent.skills)
    assert "demo_hello" in agent._reasoner_return_types
    assert hasattr(agent, "demo_hello")
    assert hasattr(agent, "demo_repeat")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=agent), base_url="http://test"
    ) as client:
        reasoner_resp = await client.post(
            "/reasoners/demo/hello",
            json={"name": "Agent"},
            headers={"x-workflow-id": "wf-router", "x-execution-id": "exec-router"},
        )

        skill_resp = await client.post(
            "/skills/demo/repeat",
            json={"text": "ping"},
            headers={"x-workflow-id": "wf-router", "x-execution-id": "exec-router"},
        )

    assert reasoner_resp.status_code == 200
    assert reasoner_resp.json() == {"message": "hello Agent"}

    assert skill_resp.status_code == 200
    assert skill_resp.json() == {"echo": "ping"}


@pytest.mark.asyncio
async def test_agent_router_prefix_sanitization(monkeypatch):
    agent, _ = create_test_agent(monkeypatch)

    router = AgentRouter(prefix="/Users/Profile-v1/")

    @router.reasoner()
    async def fetch_order(order_id: int) -> dict:
        return {"order_id": order_id}

    agent.include_router(router)

    assert any(r["id"] == "users_profile_v1_fetch_order" for r in agent.reasoners)
    assert hasattr(agent, "users_profile_v1_fetch_order")


@pytest.mark.asyncio
async def test_callback_url_precedence_and_env(monkeypatch):
    monkeypatch.setenv("AGENT_CALLBACK_URL", "https://env.example.com")

    explicit_agent, explicit_client = create_test_agent(
        monkeypatch, callback_url="https://explicit.example.com"
    )
    await explicit_agent.agentfield_handler.register_with_agentfield_server(port=9200)
    assert explicit_agent.base_url == "https://explicit.example.com:9200"
    assert (
        explicit_client.register_calls[-1]["base_url"]
        == "https://explicit.example.com:9200"
    )

    env_agent, env_client = create_test_agent(monkeypatch)
    await env_agent.agentfield_handler.register_with_agentfield_server(port=9300)
    assert env_agent.base_url == "https://env.example.com:9300"
    assert env_client.register_calls[-1]["base_url"] == "https://env.example.com:9300"
