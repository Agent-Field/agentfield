import asyncio
import httpx
import pytest
import sys
from types import SimpleNamespace
from fastapi import FastAPI

from haxen_sdk.agent_server import AgentServer


def make_agent_app():
    app = FastAPI()
    app.node_id = "agent-1"
    app.version = "1.0.0"
    app.reasoners = [{"id": "reasoner_a"}]
    app.skills = [{"id": "skill_b"}]
    app.client = SimpleNamespace(notify_graceful_shutdown_sync=lambda node_id: True)
    app.mcp_manager = type(
        "MCPManager",
        (),
        {
            "get_all_status": lambda self: {
                "test": {
                    "status": "running",
                    "port": 1234,
                    "process": type("Proc", (), {"pid": 42})(),
                }
            }
        },
    )()
    app.dev_mode = False
    app.haxen_server = "http://haxen"
    return app


@pytest.mark.asyncio
async def test_setup_haxen_routes_health_endpoint():
    app = make_agent_app()
    server = AgentServer(app)
    server.setup_haxen_routes()

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.get("/health")

    assert resp.status_code == 200
    data = resp.json()
    assert data["node_id"] == "agent-1"
    assert data["mcp_servers"]["running"] == 1

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp_reasoners = await client.get("/reasoners")
        resp_skills = await client.get("/skills")

    assert resp_reasoners.json()["reasoners"] == app.reasoners
    assert resp_skills.json()["skills"] == app.skills


@pytest.mark.asyncio
async def test_shutdown_endpoint_triggers_flags():
    app = make_agent_app()
    app.dev_mode = True
    server = AgentServer(app)
    server.setup_haxen_routes()

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/shutdown",
            json={"graceful": True, "timeout_seconds": 5},
            headers={"content-type": "application/json"},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["graceful"] is True
    assert app._shutdown_requested is True


@pytest.mark.asyncio
async def test_status_endpoint_reports_psutil(monkeypatch):
    app = make_agent_app()
    server = AgentServer(app)

    class DummyProcess:
        def memory_info(self):
            return SimpleNamespace(rss=50 * 1024 * 1024)

        def cpu_percent(self):
            return 12.5

        def num_threads(self):
            return 4

    dummy_psutil = SimpleNamespace(Process=lambda: DummyProcess())
    monkeypatch.setitem(sys.modules, "psutil", dummy_psutil)

    server.setup_haxen_routes()

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.get("/status")

    data = resp.json()
    assert data["status"] == "running"
    assert data["resources"]["memory_mb"] == 50.0
    assert data["resources"]["threads"] == 4


@pytest.mark.asyncio
async def test_shutdown_immediate_path(monkeypatch):
    app = make_agent_app()
    server = AgentServer(app)
    server.setup_haxen_routes()

    triggered = {}

    async def fake_immediate(self):
        triggered["called"] = True

    monkeypatch.setattr(AgentServer, "_immediate_shutdown", fake_immediate)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post("/shutdown", json={"graceful": False})

    assert resp.status_code == 200
    await asyncio.sleep(0)
    assert triggered.get("called") is True
    assert app._shutdown_requested is True


@pytest.mark.asyncio
async def test_mcp_start_stop_routes(monkeypatch):
    app = make_agent_app()

    class StubMCPManager:
        async def start_server_by_alias(self, alias):
            self.last_start = alias
            return True

        def stop_server(self, alias):
            self.last_stop = alias
            return True

        async def restart_server(self, alias):
            self.last_restart = alias
            return True

        def get_server_status(self, alias):
            return {"status": "running"}

        def get_all_status(self):
            return {}

    manager = StubMCPManager()
    app.mcp_manager = manager
    server = AgentServer(app)
    server.setup_haxen_routes()

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        start = await client.post("/mcp/foo/start")
        stop = await client.post("/mcp/foo/stop")
        restart = await client.post("/mcp/foo/restart")

    assert start.json()["success"] is True
    assert stop.json()["success"] is True
    assert restart.json()["success"] is True
    assert manager.last_start == "foo"
    assert manager.last_stop == "foo"
    assert manager.last_restart == "foo"
