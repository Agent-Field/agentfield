"""
Tests for agentfield.agent_server — AgentServer route registration and utility methods.
"""
from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import FastAPI

from agentfield.agent_server import AgentServer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_agent_app(**overrides):
    """Build a minimal FastAPI app that acts as a stand-in for Agent."""
    app = FastAPI()
    app.node_id = overrides.get("node_id", "agent-1")
    app.version = overrides.get("version", "1.0.0")
    app.reasoners = overrides.get("reasoners", [{"id": "reasoner_a"}])
    app.skills = overrides.get("skills", [{"id": "skill_b"}])
    app.client = overrides.get(
        "client",
        SimpleNamespace(notify_graceful_shutdown_sync=lambda node_id: True),
    )
    app.mcp_manager = overrides.get(
        "mcp_manager",
        type(
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
        )(),
    )
    app.mcp_client_registry = overrides.get("mcp_client_registry", None)
    app.dev_mode = overrides.get("dev_mode", False)
    app.agentfield_server = overrides.get("agentfield_server", "http://agentfield")
    app.base_url = overrides.get("base_url", "http://localhost:8001")
    app._pause_manager = overrides.get(
        "_pause_manager",
        SimpleNamespace(
            resolve=AsyncMock(return_value=True),
            resolve_by_execution_id=AsyncMock(return_value=False),
        ),
    )
    return app


def _setup_server(app):
    """Create AgentServer and register routes, patching install_stdio_tee."""
    server = AgentServer(app)
    with patch("agentfield.node_logs.install_stdio_tee"):
        server.setup_agentfield_routes()
    return server


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
# Route registration and health endpoint
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_health_endpoint_basic():
    app = make_agent_app()
    _setup_server(app)
    resp = await _get(app, "/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["node_id"] == "agent-1"
    assert data["status"] == "healthy"
    assert "timestamp" in data


@pytest.mark.asyncio
async def test_health_endpoint_with_mcp_servers():
    app = make_agent_app()
    _setup_server(app)
    resp = await _get(app, "/health")
    data = resp.json()
    assert data["mcp_servers"]["running"] == 1
    assert data["mcp_servers"]["total"] == 1
    assert data["mcp_servers"]["failed"] == 0
    assert data["mcp_servers"]["servers"]["test"]["pid"] == 42


@pytest.mark.asyncio
async def test_health_endpoint_no_mcp_manager():
    app = make_agent_app(mcp_manager=None)
    _setup_server(app)
    resp = await _get(app, "/health")
    data = resp.json()
    assert data["status"] == "healthy"
    assert "mcp_servers" not in data


@pytest.mark.asyncio
async def test_health_degraded_when_mcp_failed():
    mgr = type(
        "M",
        (),
        {
            "get_all_status": lambda self: {
                "a": {"status": "running", "port": 1, "process": None},
                "b": {"status": "failed", "port": 2, "process": None},
            }
        },
    )()
    app = make_agent_app(mcp_manager=mgr)
    _setup_server(app)
    resp = await _get(app, "/health")
    data = resp.json()
    assert data["status"] == "degraded"


@pytest.mark.asyncio
async def test_health_mcp_error_handling():
    """MCP manager raising an exception should not crash health endpoint."""

    class BadMgr:
        def get_all_status(self):
            raise RuntimeError("boom")

    app = make_agent_app(mcp_manager=BadMgr(), dev_mode=True)
    _setup_server(app)
    resp = await _get(app, "/health")
    data = resp.json()
    assert data["mcp_servers"]["error"] == "Failed to get MCP status"


# ---------------------------------------------------------------------------
# Reasoners / Skills listing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_reasoners():
    app = make_agent_app(reasoners=[{"id": "r1"}, {"id": "r2"}])
    _setup_server(app)
    resp = await _get(app, "/reasoners")
    assert resp.json()["reasoners"] == [{"id": "r1"}, {"id": "r2"}]


@pytest.mark.asyncio
async def test_list_skills():
    app = make_agent_app(skills=[{"id": "s1"}])
    _setup_server(app)
    resp = await _get(app, "/skills")
    assert resp.json()["skills"] == [{"id": "s1"}]


# ---------------------------------------------------------------------------
# Info endpoint
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_info_endpoint():
    app = make_agent_app()
    _setup_server(app)
    resp = await _get(app, "/info")
    data = resp.json()
    assert data["node_id"] == "agent-1"
    assert data["version"] == "1.0.0"
    assert "registered_at" in data


# ---------------------------------------------------------------------------
# Shutdown endpoint
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_shutdown_graceful():
    app = make_agent_app(dev_mode=True)
    _setup_server(app)
    # Patch _graceful_shutdown to avoid os._exit
    with patch.object(AgentServer, "_graceful_shutdown", new_callable=AsyncMock):
        resp = await _post(
            app,
            "/shutdown",
            json={"graceful": True, "timeout_seconds": 5},
            headers={"content-type": "application/json"},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["graceful"] is True
    assert data["status"] == "shutting_down"
    assert app._shutdown_requested is True


@pytest.mark.asyncio
async def test_shutdown_immediate():
    app = make_agent_app()
    _setup_server(app)
    triggered = {}

    async def fake_immediate(self):
        triggered["called"] = True

    with patch.object(AgentServer, "_immediate_shutdown", fake_immediate):
        resp = await _post(app, "/shutdown", json={"graceful": False})

    assert resp.status_code == 200
    assert resp.json()["graceful"] is False
    await asyncio.sleep(0)
    assert triggered.get("called") is True
    assert app._shutdown_requested is True


@pytest.mark.asyncio
async def test_shutdown_notification_failure():
    """Shutdown endpoint should not crash if notification fails."""
    app = make_agent_app(dev_mode=True)
    app.client = SimpleNamespace(
        notify_graceful_shutdown_sync=MagicMock(side_effect=RuntimeError("oops"))
    )
    _setup_server(app)
    with patch.object(AgentServer, "_graceful_shutdown", new_callable=AsyncMock):
        resp = await _post(
            app,
            "/shutdown",
            json={"graceful": True},
            headers={"content-type": "application/json"},
        )
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Status endpoint
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_status_endpoint_with_psutil(monkeypatch):
    app = make_agent_app()
    _setup_server(app)

    class DummyProcess:
        def memory_info(self):
            return SimpleNamespace(rss=50 * 1024 * 1024)

        def cpu_percent(self):
            return 12.5

        def num_threads(self):
            return 4

    dummy_psutil = SimpleNamespace(Process=lambda: DummyProcess())
    monkeypatch.setitem(sys.modules, "psutil", dummy_psutil)

    resp = await _get(app, "/status")
    data = resp.json()
    assert data["status"] == "running"
    assert data["resources"]["memory_mb"] == 50.0
    assert data["resources"]["threads"] == 4


@pytest.mark.asyncio
async def test_status_endpoint_without_psutil(monkeypatch):
    """When psutil is not installed, fallback info is returned."""
    app = make_agent_app()
    _setup_server(app)

    # Force ImportError for psutil
    monkeypatch.setitem(sys.modules, "psutil", None)

    resp = await _get(app, "/status")
    data = resp.json()
    assert data["status"] == "running"
    assert "Limited status info" in data.get("message", "")


@pytest.mark.asyncio
async def test_status_endpoint_shutdown_requested():
    app = make_agent_app()
    app._shutdown_requested = True
    _setup_server(app)

    class DummyProcess:
        def memory_info(self):
            return SimpleNamespace(rss=10 * 1024 * 1024)

        def cpu_percent(self):
            return 0.0

        def num_threads(self):
            return 1

    with patch.dict(sys.modules, {"psutil": SimpleNamespace(Process=lambda: DummyProcess())}):
        resp = await _get(app, "/status")

    assert resp.json()["status"] == "stopping"


# ---------------------------------------------------------------------------
# MCP status / health endpoints
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mcp_status_no_manager():
    app = make_agent_app(mcp_manager=None)
    _setup_server(app)
    resp = await _get(app, "/mcp/status")
    data = resp.json()
    assert data["total"] == 0
    assert "error" in data


@pytest.mark.asyncio
async def test_mcp_health_no_manager():
    app = make_agent_app(mcp_manager=None)
    _setup_server(app)
    resp = await _get(app, "/health/mcp")
    data = resp.json()
    assert data["summary"]["total_servers"] == 0
    assert data["servers"] == []


# ---------------------------------------------------------------------------
# MCP start / stop / restart
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mcp_start_stop_restart():
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
    _setup_server(app)

    start = await _post(app, "/mcp/foo/start")
    stop = await _post(app, "/mcp/foo/stop")
    restart = await _post(app, "/mcp/foo/restart")

    assert start.json()["success"] is True
    assert stop.json()["success"] is True
    assert restart.json()["success"] is True
    assert manager.last_start == "foo"
    assert manager.last_stop == "foo"
    assert manager.last_restart == "foo"


@pytest.mark.asyncio
async def test_mcp_start_no_manager():
    app = make_agent_app(mcp_manager=None)
    _setup_server(app)
    resp = await _post(app, "/mcp/test/start")
    assert resp.json()["success"] is False


@pytest.mark.asyncio
async def test_mcp_stop_no_manager():
    app = make_agent_app(mcp_manager=None)
    _setup_server(app)
    resp = await _post(app, "/mcp/test/stop")
    assert resp.json()["success"] is False


@pytest.mark.asyncio
async def test_mcp_restart_no_manager():
    app = make_agent_app(mcp_manager=None)
    _setup_server(app)
    resp = await _post(app, "/mcp/test/restart")
    assert resp.json()["success"] is False


@pytest.mark.asyncio
async def test_mcp_start_failure():
    class FailMgr:
        async def start_server_by_alias(self, alias):
            return False

        def get_all_status(self):
            return {}

    app = make_agent_app(mcp_manager=FailMgr())
    _setup_server(app)
    resp = await _post(app, "/mcp/bar/start")
    data = resp.json()
    assert data["success"] is False
    assert "Failed" in data["error"]


@pytest.mark.asyncio
async def test_mcp_start_exception():
    class ExcMgr:
        async def start_server_by_alias(self, alias):
            raise RuntimeError("boom")

        def get_all_status(self):
            return {}

    app = make_agent_app(mcp_manager=ExcMgr())
    _setup_server(app)
    resp = await _post(app, "/mcp/bar/start")
    data = resp.json()
    assert data["success"] is False
    assert "boom" in data["error"]


@pytest.mark.asyncio
async def test_mcp_stop_failure():
    class FailMgr:
        def stop_server(self, alias):
            return False

        def get_all_status(self):
            return {}

    app = make_agent_app(mcp_manager=FailMgr())
    _setup_server(app)
    resp = await _post(app, "/mcp/bar/stop")
    assert resp.json()["success"] is False


@pytest.mark.asyncio
async def test_mcp_stop_exception():
    class ExcMgr:
        def stop_server(self, alias):
            raise RuntimeError("fail")

        def get_all_status(self):
            return {}

    app = make_agent_app(mcp_manager=ExcMgr())
    _setup_server(app)
    resp = await _post(app, "/mcp/bar/stop")
    assert resp.json()["success"] is False
    assert "fail" in resp.json()["error"]


@pytest.mark.asyncio
async def test_mcp_restart_failure():
    class FailMgr:
        async def restart_server(self, alias):
            return False

        def get_all_status(self):
            return {}

    app = make_agent_app(mcp_manager=FailMgr())
    _setup_server(app)
    resp = await _post(app, "/mcp/bar/restart")
    assert resp.json()["success"] is False


@pytest.mark.asyncio
async def test_mcp_restart_exception():
    class ExcMgr:
        async def restart_server(self, alias):
            raise RuntimeError("err")

        def get_all_status(self):
            return {}

    app = make_agent_app(mcp_manager=ExcMgr())
    _setup_server(app)
    resp = await _post(app, "/mcp/bar/restart")
    assert "err" in resp.json()["error"]


# ---------------------------------------------------------------------------
# MCP server tools endpoint
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mcp_server_tools_no_registry():
    app = make_agent_app(mcp_client_registry=None)
    _setup_server(app)
    resp = await _get(app, "/mcp/servers/test/tools")
    data = resp.json()
    assert data["tools"] == []
    assert "error" in data


@pytest.mark.asyncio
async def test_mcp_server_tools_client_not_found():
    registry = SimpleNamespace(get_client=lambda alias: None)
    app = make_agent_app(mcp_client_registry=registry)
    _setup_server(app)
    resp = await _get(app, "/mcp/servers/missing/tools")
    data = resp.json()
    assert data["tools"] == []
    assert "not found" in data["error"]


# ---------------------------------------------------------------------------
# Approval webhook
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_approval_webhook_valid():
    app = make_agent_app(dev_mode=True)
    _setup_server(app)
    resp = await _post(
        app,
        "/webhooks/approval",
        json={
            "execution_id": "exec-1",
            "decision": "approved",
            "feedback": "looks good",
            "approval_request_id": "ar-1",
        },
    )
    data = resp.json()
    assert data["status"] == "received"
    assert data["resolved"] is True


@pytest.mark.asyncio
async def test_approval_webhook_missing_fields():
    app = make_agent_app()
    _setup_server(app)
    resp = await _post(
        app, "/webhooks/approval", json={"execution_id": "", "decision": ""}
    )
    data = resp.json()
    assert data["status"] == 400


@pytest.mark.asyncio
async def test_approval_webhook_with_string_response():
    app = make_agent_app()
    _setup_server(app)
    resp = await _post(
        app,
        "/webhooks/approval",
        json={
            "execution_id": "e1",
            "decision": "approved",
            "response": '{"key": "val"}',
        },
    )
    assert resp.json()["status"] == "received"


@pytest.mark.asyncio
async def test_approval_webhook_with_dict_response():
    app = make_agent_app()
    _setup_server(app)
    resp = await _post(
        app,
        "/webhooks/approval",
        json={
            "execution_id": "e1",
            "decision": "rejected",
            "response": {"key": "val"},
        },
    )
    assert resp.json()["status"] == "received"


@pytest.mark.asyncio
async def test_approval_webhook_unparseable_response():
    app = make_agent_app()
    _setup_server(app)
    resp = await _post(
        app,
        "/webhooks/approval",
        json={
            "execution_id": "e1",
            "decision": "approved",
            "response": "not json at all",
        },
    )
    assert resp.json()["status"] == "received"


@pytest.mark.asyncio
async def test_approval_webhook_resolve_by_execution_id_fallback():
    """When approval_request_id is missing, resolves by execution_id."""
    app = make_agent_app()
    app._pause_manager = SimpleNamespace(
        resolve=AsyncMock(return_value=False),
        resolve_by_execution_id=AsyncMock(return_value=True),
    )
    _setup_server(app)
    resp = await _post(
        app,
        "/webhooks/approval",
        json={"execution_id": "e1", "decision": "approved"},
    )
    data = resp.json()
    assert data["resolved"] is True
    app._pause_manager.resolve_by_execution_id.assert_awaited_once_with(
        "e1", pytest.importorskip("unittest.mock").ANY
    )


# ---------------------------------------------------------------------------
# Utility methods (no HTTP needed)
# ---------------------------------------------------------------------------


class TestFormatUptime:
    def _server(self):
        app = make_agent_app()
        return AgentServer(app)

    def test_seconds_only(self):
        assert self._server()._format_uptime(45) == "45s"

    def test_minutes_and_seconds(self):
        assert self._server()._format_uptime(125) == "2m 5s"

    def test_hours_minutes_seconds(self):
        assert self._server()._format_uptime(3661) == "1h 1m 1s"

    def test_zero_seconds(self):
        assert self._server()._format_uptime(0) == "0s"

    def test_exact_hour(self):
        assert self._server()._format_uptime(3600) == "1h"

    def test_exact_minute(self):
        assert self._server()._format_uptime(60) == "1m"


class TestValidateSSLConfig:
    def _server(self, dev_mode=False):
        app = make_agent_app(dev_mode=dev_mode)
        return AgentServer(app)

    def test_both_none(self):
        assert self._server()._validate_ssl_config(None, None) is False

    def test_key_none(self):
        assert self._server()._validate_ssl_config(None, "/some/cert") is False

    def test_cert_none(self):
        assert self._server()._validate_ssl_config("/some/key", None) is False

    def test_nonexistent_files(self, tmp_path):
        s = self._server(dev_mode=True)
        assert s._validate_ssl_config(str(tmp_path / "nope.key"), str(tmp_path / "nope.crt")) is False

    def test_valid_files(self, tmp_path):
        key = tmp_path / "server.key"
        cert = tmp_path / "server.crt"
        key.write_text("key")
        cert.write_text("cert")
        assert self._server()._validate_ssl_config(str(key), str(cert)) is True


class TestGetOptimalWorkers:
    def _server(self, dev_mode=False):
        app = make_agent_app(dev_mode=dev_mode)
        return AgentServer(app)

    def test_explicit_workers(self):
        assert self._server()._get_optimal_workers(4) == 4

    def test_env_var(self, monkeypatch):
        monkeypatch.setenv("UVICORN_WORKERS", "6")
        assert self._server()._get_optimal_workers() == 6

    def test_env_var_non_numeric(self, monkeypatch):
        monkeypatch.setenv("UVICORN_WORKERS", "abc")
        # Falls through to CPU auto-detect
        result = self._server()._get_optimal_workers()
        assert result is None or isinstance(result, int)

    def test_auto_detect(self, monkeypatch):
        monkeypatch.delenv("UVICORN_WORKERS", raising=False)
        result = self._server()._get_optimal_workers()
        assert result is None or isinstance(result, int)


class TestCheckPerformanceDependencies:
    def _server(self):
        app = make_agent_app()
        return AgentServer(app)

    def test_returns_dict(self):
        deps = self._server()._check_performance_dependencies()
        assert "uvloop" in deps
        assert "psutil" in deps
        assert "orjson" in deps
        assert all(isinstance(v, bool) for v in deps.values())


# ---------------------------------------------------------------------------
# Logs endpoint
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_logs_endpoint_disabled():
    app = make_agent_app()
    _setup_server(app)
    with patch("agentfield.node_logs.logs_enabled", return_value=False):
        resp = await _get(app, "/agentfield/v1/logs")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_logs_endpoint_unauthorized():
    app = make_agent_app()
    _setup_server(app)
    with patch("agentfield.node_logs.logs_enabled", return_value=True), \
         patch("agentfield.node_logs.verify_internal_bearer", return_value=False):
        resp = await _get(app, "/agentfield/v1/logs")
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_logs_endpoint_tail_too_large(monkeypatch):
    app = make_agent_app()
    _setup_server(app)
    monkeypatch.setenv("AGENTFIELD_LOG_MAX_TAIL_LINES", "100")
    with patch("agentfield.node_logs.logs_enabled", return_value=True), \
         patch("agentfield.node_logs.verify_internal_bearer", return_value=True):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/agentfield/v1/logs?tail_lines=999")
    assert resp.status_code == 413


@pytest.mark.asyncio
async def test_logs_endpoint_success():
    app = make_agent_app()
    _setup_server(app)

    async def fake_iter(tail, since, follow):
        yield '{"line": 1}\n'

    with patch("agentfield.node_logs.logs_enabled", return_value=True), \
         patch("agentfield.node_logs.verify_internal_bearer", return_value=True), \
         patch("agentfield.node_logs.iter_tail_ndjson", side_effect=fake_iter):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get(
                "/agentfield/v1/logs?tail_lines=10",
                headers={"Authorization": "Bearer tok"},
            )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/x-ndjson")
