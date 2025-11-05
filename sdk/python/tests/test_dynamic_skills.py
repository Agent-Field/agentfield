from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI

from haxen_sdk.dynamic_skills import DynamicMCPSkillManager


class StubMCPClient:
    def __init__(self, tools, *, result=None):
        self._tools = tools
        self._result = result or {"echo": "ok"}
        self.calls = []

    async def health_check(self):
        return True

    async def list_tools(self):
        return self._tools

    async def call_tool(self, name, args):
        self.calls.append((name, args))
        return self._result


@pytest.mark.asyncio
async def test_dynamic_skill_registration(monkeypatch):
    app = FastAPI()
    app.node_id = "agent"
    app.reasoners = []
    app.skills = []
    app.dev_mode = False
    app.mcp_client_registry = SimpleNamespace(
        clients={
            "server": StubMCPClient(
                tools=[
                    {
                        "name": "Echo",
                        "description": "Echo tool",
                        "inputSchema": {
                            "properties": {"text": {"type": "string"}},
                            "required": ["text"],
                        },
                    }
                ]
            )
        },
        get_client=lambda alias: None,
    )

    def get_client(alias):
        return app.mcp_client_registry.clients[alias]

    app.mcp_client_registry.get_client = get_client

    manager = DynamicMCPSkillManager(app)
    await manager.discover_and_register_all_skills()

    assert "server_Echo" in manager.registered_skills
    assert any(skill["id"] == "server_Echo" for skill in app.skills)

    client_stub = app.mcp_client_registry.get_client("server")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/skills/server_Echo",
            json={"text": "hello"},
            headers={"x-workflow-id": "wf", "x-execution-id": "exec"},
        )

    assert resp.status_code == 200
    assert client_stub.calls[0][0].lower() == "echo"
    assert client_stub.calls[0][1] == {"text": "hello"}


@pytest.mark.asyncio
async def test_mcp_registry_absent_succeeds_quickly():
    app = FastAPI()
    app.node_id = "agent"
    app.reasoners = []
    app.skills = []
    app.dev_mode = False
    app.mcp_client_registry = None

    manager = DynamicMCPSkillManager(app)
    await manager.discover_and_register_all_skills()
    assert manager.registered_skills == {}
