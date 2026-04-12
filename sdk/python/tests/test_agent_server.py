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


# ---------------------------------------------------------------------------
# /debug/tasks endpoint
#
# This endpoint exists specifically to diagnose silent litellm deadlocks like
# the one in PR #384: when py-spy shows all asyncio worker threads idle and
# the agent is unresponsive, hitting /debug/tasks reveals which coroutines
# are awaiting which Future. Without this, finding the hang requires
# attaching py-spy to a production container.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_debug_tasks_endpoint_returns_running_task_stacks():
    """The endpoint must enumerate all live asyncio.Tasks and include the
    current task itself in the response."""
    app = make_agent_app()
    _setup_server(app)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.get("/debug/tasks")

    assert resp.status_code == 200
    body = resp.json()
    assert "count" in body
    assert "tasks" in body
    assert isinstance(body["tasks"], list)
    assert body["count"] == len(body["tasks"])
    # The HTTP request handler itself runs as an asyncio task → must be present.
    assert body["count"] >= 1
    joined = "\n".join(body["tasks"])
    assert "Task" in joined  # each entry starts with "=== Task ..."


@pytest.mark.asyncio
async def test_debug_tasks_endpoint_captures_pending_coroutines():
    """If a coroutine is suspended on an awaitable that never resolves, the
    debug dump must surface it. This is the production scenario the endpoint
    is designed for: a hung litellm.acompletion call sitting on a Future."""
    app = make_agent_app()
    _setup_server(app)

    pending_event = asyncio.Event()  # never set → simulates a hung HTTP read

    async def hung_coroutine():
        await pending_event.wait()

    hung_task = asyncio.create_task(hung_coroutine(), name="simulated-hung-llm-call")
    try:
        # Yield to let the task get scheduled.
        await asyncio.sleep(0)

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/debug/tasks")

        assert resp.status_code == 200
        body = resp.json()
        joined = "\n".join(body["tasks"])
        assert "simulated-hung-llm-call" in joined, (
            "/debug/tasks must surface tasks suspended on Futures so we can "
            "diagnose deadlocks in production. Found tasks: "
            + joined[:500]
        )
    finally:
        pending_event.set()
        hung_task.cancel()
        try:
            await hung_task
        except (asyncio.CancelledError, BaseException):
            pass


@pytest.mark.asyncio
async def test_debug_tasks_endpoint_survives_cancelled_and_done_tasks():
    """The endpoint must remain responsive even when the task list contains
    tasks in pathological states (cancelled, done with exception, etc.).
    A naive implementation that calls `task.get_stack()` on a finished task
    will not crash but will return an empty stack — verify the JSON is still
    well-formed."""
    app = make_agent_app()
    _setup_server(app)

    async def quick_done():
        return "done"

    async def cancelled_one():
        await asyncio.sleep(60)

    done_task = asyncio.create_task(quick_done(), name="already-done-task")
    cancelled_task = asyncio.create_task(cancelled_one(), name="will-be-cancelled-task")
    await asyncio.sleep(0)  # let scheduling happen
    cancelled_task.cancel()
    try:
        await cancelled_task
    except asyncio.CancelledError:
        pass
    await done_task  # let it finish

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.get("/debug/tasks")

    assert resp.status_code == 200
    body = resp.json()
    # Must return well-formed JSON regardless of task states.
    assert isinstance(body["count"], int)
    assert isinstance(body["tasks"], list)
    # The endpoint should not crash even though the cancelled and done tasks
    # may already have been removed from the live set by the time the request
    # handler runs.


@pytest.mark.asyncio
async def test_debug_tasks_endpoint_reports_done_and_cancelled_state():
    """Pin down the schema: each task entry must include `done=` and
    `cancelled=` markers so operators can quickly distinguish "stuck on
    await" from "finished but not yet collected" when diagnosing a hang."""
    app = make_agent_app()
    _setup_server(app)

    pending_event = asyncio.Event()

    async def stuck():
        await pending_event.wait()

    stuck_task = asyncio.create_task(stuck(), name="stuck-on-await-task")
    try:
        await asyncio.sleep(0)

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/debug/tasks")

        body = resp.json()
        joined = "\n".join(body["tasks"])
        # Find the entry for our stuck task and assert it carries the
        # done/cancelled markers we documented in the endpoint.
        assert "stuck-on-await-task" in joined
        assert "done=False" in joined
        assert "cancelled=False" in joined
    finally:
        pending_event.set()
        stuck_task.cancel()
        try:
            await stuck_task
        except (asyncio.CancelledError, BaseException):
            pass
