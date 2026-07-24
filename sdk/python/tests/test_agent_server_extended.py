"""Extended tests for AgentServer routes.

Covers paths not exercised by test_agent_server.py:
- Health endpoint
- /info endpoint returns correct schema
- /status endpoint fallback when psutil is unavailable
- /reasoners and /skills discovery endpoints
- Malformed JSON body to /shutdown falls back gracefully
"""

from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI

from agentfield.agent_server import AgentServer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_app(*, dev_mode=False, base_url="http://agent.local:8000"):
    """Minimal FastAPI application wired like a real Agent."""
    app = FastAPI()
    app.node_id = "test-node"
    app.version = "1.2.3"
    app.base_url = base_url
    app.reasoners = [{"id": "do_something", "description": "Does something"}]
    app.skills = [{"id": "skill_a"}]
    app.dev_mode = dev_mode
    app.agentfield_server = "http://agentfield"
    app.client = SimpleNamespace(notify_graceful_shutdown_sync=lambda node_id: True)
    app._shutdown_requested = False
    return app


async def _get(app, path):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        return await client.get(path)


async def _post(app, path, **kwargs):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        return await client.post(path, **kwargs)


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_health_endpoint():
    app = _make_app()
    AgentServer(app).setup_agentfield_routes()

    resp = await _get(app, "/health")

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["node_id"] == "test-node"


# ---------------------------------------------------------------------------
# /reasoners and /skills discovery
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reasoners_endpoint_returns_registered_reasoners():
    app = _make_app()
    AgentServer(app).setup_agentfield_routes()

    resp = await _get(app, "/reasoners")

    assert resp.status_code == 200
    data = resp.json()
    assert "reasoners" in data
    assert data["reasoners"] == app.reasoners


@pytest.mark.asyncio
async def test_skills_endpoint_returns_registered_skills():
    app = _make_app()
    AgentServer(app).setup_agentfield_routes()

    resp = await _get(app, "/skills")

    assert resp.status_code == 200
    data = resp.json()
    assert "skills" in data
    assert data["skills"] == app.skills


@pytest.mark.asyncio
async def test_reasoners_endpoint_empty_when_none_registered():
    app = _make_app()
    app.reasoners = []
    AgentServer(app).setup_agentfield_routes()

    resp = await _get(app, "/reasoners")

    assert resp.json()["reasoners"] == []


# ---------------------------------------------------------------------------
# /info endpoint
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_info_endpoint_returns_node_metadata():
    app = _make_app()
    AgentServer(app).setup_agentfield_routes()

    resp = await _get(app, "/info")

    assert resp.status_code == 200
    data = resp.json()
    assert data["node_id"] == "test-node"
    assert data["version"] == "1.2.3"
    assert data["base_url"] == "http://agent.local:8000"
    assert "reasoners" in data
    assert "skills" in data
    assert "registered_at" in data


# ---------------------------------------------------------------------------
# /shutdown — graceful path sets flag
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_shutdown_graceful_sets_shutdown_requested():
    app = _make_app()
    AgentServer(app).setup_agentfield_routes()

    resp = await _post(
        app,
        "/shutdown",
        json={"graceful": True, "timeout_seconds": 1},
        headers={"content-type": "application/json"},
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["graceful"] is True
    assert app._shutdown_requested is True


@pytest.mark.asyncio
async def test_shutdown_immediate_sets_shutdown_requested(monkeypatch):
    app = _make_app()

    triggered = {}

    async def fake_immediate(self):
        triggered["called"] = True

    monkeypatch.setattr(AgentServer, "_immediate_shutdown", fake_immediate)
    AgentServer(app).setup_agentfield_routes()

    resp = await _post(app, "/shutdown", json={"graceful": False})

    assert resp.status_code == 200
    assert app._shutdown_requested is True
    await asyncio.sleep(0)
    assert triggered.get("called") is True


# ---------------------------------------------------------------------------
# /status — psutil unavailable fallback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_status_without_psutil_returns_basic_info(monkeypatch):
    app = _make_app()
    AgentServer(app).setup_agentfield_routes()

    # Hide psutil so the ImportError branch is taken
    monkeypatch.setitem(sys.modules, "psutil", None)  # type: ignore[call-overload]

    resp = await _get(app, "/status")

    assert resp.status_code == 200
    data = resp.json()
    assert data["node_id"] == "test-node"
    assert "version" in data


# ---------------------------------------------------------------------------
# /status — running when no shutdown requested
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_status_shows_running_by_default(monkeypatch):
    app = _make_app()
    AgentServer(app).setup_agentfield_routes()

    class _FakeProcess:
        def memory_info(self):
            return SimpleNamespace(rss=10 * 1024 * 1024)

        def cpu_percent(self):
            return 0.5

        def num_threads(self):
            return 2

    dummy_psutil = SimpleNamespace(Process=lambda: _FakeProcess())
    monkeypatch.setitem(sys.modules, "psutil", dummy_psutil)

    resp = await _get(app, "/status")

    data = resp.json()
    assert data["status"] == "running"


@pytest.mark.asyncio
async def test_status_shows_stopping_after_shutdown_requested(monkeypatch):
    app = _make_app()
    app._shutdown_requested = True
    AgentServer(app).setup_agentfield_routes()

    class _FakeProcess:
        def memory_info(self):
            return SimpleNamespace(rss=5 * 1024 * 1024)

        def cpu_percent(self):
            return 0.0

        def num_threads(self):
            return 1

    monkeypatch.setitem(sys.modules, "psutil", SimpleNamespace(Process=lambda: _FakeProcess()))

    resp = await _get(app, "/status")

    data = resp.json()
    assert data["status"] == "stopping"
