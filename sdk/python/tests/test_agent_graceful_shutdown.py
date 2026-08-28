import asyncio
import os
import signal
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from fastapi import FastAPI

from agentfield.agent import Agent
from agentfield.agent_field_handler import AgentFieldHandler
from agentfield.agent_server import AgentServer, parse_shutdown_timeout
from agentfield.client import AgentFieldClient
from agentfield.types import AgentStatus
from tests.helpers import DummyAgentFieldClient, StubAgent


class ExitCalled(Exception):
    pass


def make_shutdown_agent():
    return StubAgent(
        client=DummyAgentFieldClient(),
        dev_mode=True,
    )


@pytest.mark.asyncio
async def test_agent_stop_is_idempotent():
    agent = Agent(
        node_id="shutdown-agent",
        agentfield_server="http://agentfield",
        auto_register=False,
        enable_mcp=False,
        enable_did=False,
    )

    heartbeat_stop = Mock()
    notify_shutdown = AsyncMock(return_value=True)
    stop_connection_manager = AsyncMock()
    close_memory_event_client = AsyncMock()

    agent.agentfield_handler = SimpleNamespace(stop_heartbeat=heartbeat_stop)
    agent.agentfield_connected = True
    agent.client = SimpleNamespace(notify_graceful_shutdown=notify_shutdown)
    agent.connection_manager = SimpleNamespace(stop=stop_connection_manager)
    agent.memory_event_client = SimpleNamespace(close=close_memory_event_client)
    agent._cleanup_async_resources = AsyncMock()
    agent._set_as_current()

    assert Agent.get_current() is agent

    await agent.stop()
    await agent.stop()

    assert agent._shutdown_requested is True
    assert agent._current_status == AgentStatus.OFFLINE
    assert Agent.get_current() is None
    heartbeat_stop.assert_called_once()
    notify_shutdown.assert_awaited_once_with(agent.node_id)
    stop_connection_manager.assert_awaited_once()
    close_memory_event_client.assert_awaited_once()
    agent._cleanup_async_resources.assert_awaited_once()


@pytest.mark.asyncio
async def test_agent_stop_skips_shutdown_notification_when_not_connected():
    agent = Agent(
        node_id="shutdown-agent-disconnected",
        agentfield_server="http://agentfield",
        auto_register=False,
        enable_mcp=False,
        enable_did=False,
    )

    notify_shutdown = AsyncMock(return_value=True)
    agent.agentfield_connected = False
    agent.client = SimpleNamespace(notify_graceful_shutdown=notify_shutdown)
    agent._cleanup_async_resources = AsyncMock()

    await agent.stop()

    notify_shutdown.assert_not_awaited()
    agent._cleanup_async_resources.assert_awaited_once()


@pytest.mark.parametrize(
    ("value", "expected"),
    [("45", 45.0), ("45s", 45.0), ("2m", 120.0)],
)
def test_shutdown_timeout_parses_seconds_and_minutes(value, expected):
    assert parse_shutdown_timeout(value) == expected


def test_shutdown_timeout_invalid_uses_default_and_warns(monkeypatch):
    warning = Mock()
    monkeypatch.setattr("agentfield.agent_server.log_warn", warning)

    assert parse_shutdown_timeout("eventually") == 30.0
    warning.assert_called_once()


def test_legacy_fast_lifecycle_signal_handler_marks_shutdown_and_notifies(monkeypatch):
    agent = make_shutdown_agent()
    handler = AgentFieldHandler(agent)
    registered = {}
    kill_calls = []

    def fake_signal(signum, callback):
        registered[signum] = callback

    monkeypatch.setattr("agentfield.agent_field_handler.signal.signal", fake_signal)
    monkeypatch.setattr(
        "agentfield.agent_field_handler.os.kill",
        lambda pid, signum: kill_calls.append((pid, signum)),
    )

    handler.setup_fast_lifecycle_signal_handlers()
    registered[signal.SIGTERM](signal.SIGTERM, None)

    assert agent._shutdown_requested is True
    assert agent._current_status == AgentStatus.OFFLINE
    assert agent.client.shutdown_calls == [agent.node_id]
    assert kill_calls == [(os.getpid(), signal.SIGTERM)]


def test_fast_lifecycle_signal_handler_tolerates_notification_failure(monkeypatch):
    agent = make_shutdown_agent()

    def fail_notify(node_id):
        raise RuntimeError("shutdown notify failed")

    agent.client.notify_graceful_shutdown_sync = fail_notify
    handler = AgentFieldHandler(agent)
    registered = {}
    kill_calls = []

    def fake_signal(signum, callback):
        registered[signum] = callback

    monkeypatch.setattr("agentfield.agent_field_handler.signal.signal", fake_signal)
    monkeypatch.setattr(
        "agentfield.agent_field_handler.os.kill",
        lambda pid, signum: kill_calls.append((pid, signum)),
    )

    handler.setup_fast_lifecycle_signal_handlers()
    registered[signal.SIGTERM](signal.SIGTERM, None)

    assert agent._shutdown_requested is True
    assert agent._current_status == AgentStatus.OFFLINE
    assert kill_calls == [(os.getpid(), signal.SIGTERM)]


def test_legacy_signal_handler_skips_duplicate_shutdown_work(monkeypatch):
    agent = make_shutdown_agent()
    agent._shutdown_requested = True
    agent._current_status = AgentStatus.READY
    handler = AgentFieldHandler(agent)
    registered = {}
    kill_calls = []

    monkeypatch.setattr(
        "agentfield.agent_field_handler.signal.signal",
        lambda signum, callback: registered.setdefault(signum, callback),
    )
    monkeypatch.setattr(
        "agentfield.agent_field_handler.os.kill",
        lambda pid, signum: kill_calls.append((pid, signum)),
    )

    handler.setup_fast_lifecycle_signal_handlers()
    registered[signal.SIGTERM](signal.SIGTERM, None)

    assert agent._current_status == AgentStatus.READY
    assert agent.client.shutdown_calls == []
    assert kill_calls == [(os.getpid(), signal.SIGTERM)]


@pytest.mark.asyncio
async def test_cleanup_async_resources_releases_manager_and_client():
    agent = Agent(
        node_id="cleanup-agent",
        agentfield_server="http://agentfield",
        auto_register=False,
        enable_mcp=False,
        enable_did=False,
    )

    manager = SimpleNamespace(stop=AsyncMock(), closed=False)
    client = SimpleNamespace(aclose=AsyncMock())
    agent._async_execution_manager = manager
    agent.client = client

    await agent._cleanup_async_resources()

    manager.stop.assert_awaited_once()
    client.aclose.assert_awaited_once()
    assert agent._async_execution_manager is None


@pytest.mark.asyncio
async def test_graceful_shutdown_cancels_in_flight_tasks_within_deadline(monkeypatch):
    agent = make_shutdown_agent()
    agent.mcp_handler = SimpleNamespace(_cleanup_mcp_servers=lambda: None)
    agent.agentfield_handler = SimpleNamespace(stop_heartbeat=lambda: None)
    server = AgentServer(agent)

    started = asyncio.Event()

    async def long_running():
        started.set()
        await asyncio.sleep(60)

    tasks = [asyncio.create_task(long_running()) for _ in range(5)]
    server._in_flight_tasks.update(tasks)
    await started.wait()

    monkeypatch.setattr(
        "agentfield.agent_server.clear_current_agent", lambda: None, raising=False
    )
    monkeypatch.setattr(
        "agentfield.agent_server.asyncio.sleep", AsyncMock(return_value=None)
    )
    monkeypatch.setattr(
        "agentfield.agent_server.os._exit",
        lambda code: (_ for _ in ()).throw(ExitCalled(code)),
    )

    with pytest.raises(ExitCalled):
        await server._graceful_shutdown(timeout_seconds=0)

    assert all(task.done() for task in tasks)


@pytest.mark.asyncio
async def test_reasoner_background_task_completes_and_callback_precedes_cleanup():
    agent = make_shutdown_agent()
    agent._background_tasks = set()
    server = AgentServer(agent)
    events = []

    async def dispatched_reasoner():
        await asyncio.sleep(0)
        events.append("callback:succeeded")

    task = asyncio.create_task(dispatched_reasoner())
    agent._background_tasks.add(task)
    task.add_done_callback(agent._background_tasks.discard)

    await server._drain_reasoner_tasks(1)
    events.append("client:close")

    assert events == ["callback:succeeded", "client:close"]


@pytest.mark.asyncio
async def test_reasoner_background_task_cancelled_at_budget_reports_shutdown():
    agent = make_shutdown_agent()
    agent._background_tasks = set()
    agent._shutdown_cancelling = False
    server = AgentServer(agent)
    terminal = asyncio.Event()
    payload = {}

    async def dispatched_reasoner():
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            payload.update(status="cancelled", error="cancelled during shutdown")
            terminal.set()
            raise

    task = asyncio.create_task(dispatched_reasoner())
    agent._background_tasks.add(task)
    task.add_done_callback(agent._background_tasks.discard)
    await asyncio.sleep(0)

    await server._drain_reasoner_tasks(0)

    assert agent._shutdown_cancelling is True
    assert terminal.is_set()
    assert payload == {"status": "cancelled", "error": "cancelled during shutdown"}


@pytest.mark.asyncio
async def test_cancellation_resistant_reasoner_is_abandoned_after_settlement(monkeypatch):
    agent = make_shutdown_agent()
    agent._background_tasks = set()
    server = AgentServer(agent)
    release = asyncio.Event()

    async def cancellation_resistant_reasoner():
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            await release.wait()

    task = asyncio.create_task(cancellation_resistant_reasoner())
    agent._background_tasks.add(task)
    await asyncio.sleep(0)
    monkeypatch.setattr("agentfield.agent_server.SHUTDOWN_SETTLEMENT_SECONDS", 0.02)

    started = time.monotonic()
    await server._drain_reasoner_tasks(0.01)
    elapsed = time.monotonic() - started

    assert elapsed < 0.1
    assert not task.done()
    release.set()
    await task


@pytest.mark.asyncio
async def test_hanging_terminal_callback_is_abandoned_after_settlement(monkeypatch):
    agent = Agent(
        node_id="shutdown-callback",
        agentfield_server="http://control",
        auto_register=False,
        enable_mcp=False,
        enable_did=False,
    )
    agent._background_tasks = set()
    callback_started = asyncio.Event()
    release_callback = asyncio.Event()

    async def hanging_request(*args, **kwargs):
        callback_started.set()
        await release_callback.wait()

    agent.client._async_request = hanging_request
    server = AgentServer(agent)

    async def reasoner_with_terminal_callback():
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            await agent._post_execution_status(
                "/executions/exec/status", {"status": "cancelled"}, "exec"
            )

    task = asyncio.create_task(reasoner_with_terminal_callback())
    agent._background_tasks.add(task)
    await asyncio.sleep(0)
    monkeypatch.setattr("agentfield.agent_server.SHUTDOWN_SETTLEMENT_SECONDS", 0.02)

    await server._drain_reasoner_tasks(0)

    assert callback_started.is_set()
    assert not task.done()
    release_callback.set()
    await task


@pytest.mark.asyncio
async def test_repeated_http_shutdown_sends_one_notice_and_schedules_one_task():
    agent = FastAPI()
    agent.node_id = "shutdown-http"
    agent.version = "1.0.0"
    agent.reasoners = []
    agent.skills = []
    agent.dev_mode = False
    agent.agentfield_server = "http://control"
    agent.base_url = "http://agent"
    agent._shutdown_requested = False
    agent.client = SimpleNamespace(notify_graceful_shutdown_sync=Mock(return_value=True))
    server = AgentServer(agent)
    server._graceful_shutdown = AsyncMock()
    server.setup_agentfield_routes()

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=agent), base_url="http://agent"
    ) as client:
        responses = await asyncio.gather(
            client.post("/shutdown", json={"timeout_seconds": 1}),
            client.post("/shutdown", json={"timeout_seconds": 1}),
        )
    await asyncio.sleep(0)

    assert [response.status_code for response in responses] == [202, 202]
    agent.client.notify_graceful_shutdown_sync.assert_called_once_with(
        agent.node_id, reason="http", timeout_seconds=1
    )
    server._graceful_shutdown.assert_awaited_once_with(1)


@pytest.mark.asyncio
async def test_shutdown_cancellation_status_and_workflow_payloads_agree(monkeypatch):
    agent = Agent(
        node_id="shutdown-events",
        agentfield_server="http://control",
        auto_register=False,
        enable_mcp=False,
        enable_did=False,
    )
    started = asyncio.Event()
    recorded = []

    @agent.reasoner()
    async def wait_for_shutdown() -> None:
        started.set()
        await asyncio.sleep(60)

    class Response:
        status_code = 200

    async def record_request(self, method, url, **kwargs):
        recorded.append((url, kwargs.get("json")))
        return Response()

    monkeypatch.setattr(AgentFieldClient, "_async_request", record_request)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=agent), base_url="http://agent"
    ) as client:
        response = await client.post(
            "/reasoners/wait_for_shutdown",
            json={},
            headers={
                "X-Execution-ID": "exec-shutdown",
                "X-Run-ID": "run-shutdown",
                "X-Workflow-ID": "workflow-shutdown",
            },
        )

    assert response.status_code == 202
    await started.wait()
    agent._shutdown_cancelling = True
    for task in list(agent._background_tasks):
        task.cancel()
    await asyncio.gather(*list(agent._background_tasks), return_exceptions=True)

    status_payload = next(
        payload for url, payload in recorded if url.endswith("/exec-shutdown/status")
    )
    workflow_payload = next(
        payload
        for url, payload in recorded
        if url.endswith("/api/v1/workflow/executions/events")
        and payload.get("status") == "cancelled"
    )
    assert status_payload["status"] == workflow_payload["status"] == "cancelled"
    assert (
        status_payload["status_reason"]
        == workflow_payload["status_reason"]
        == "shutdown timeout exceeded"
    )


@pytest.mark.asyncio
async def test_signal_shutdown_notifies_without_delaying_uvicorn_exit():
    agent = make_shutdown_agent()
    notification_started = asyncio.Event()
    release_notification = asyncio.Event()

    async def notify(node_id, **kwargs):
        notification_started.set()
        await release_notification.wait()
        return True

    agent.client.notify_graceful_shutdown = notify
    agent.agentfield_handler = SimpleNamespace(stop_heartbeat=Mock())
    server = AgentServer(agent)
    uvicorn_server = SimpleNamespace(handle_exit=Mock())
    server._uvicorn_server = uvicorn_server

    server._begin_signal_shutdown(signal.SIGTERM)

    uvicorn_server.handle_exit.assert_called_once_with(signal.SIGTERM, None)
    await notification_started.wait()
    assert server._shutdown_notification_task is not None
    assert not server._shutdown_notification_task.done()
    release_notification.set()
    await server._shutdown_notification_task


@pytest.mark.asyncio
async def test_graceful_shutdown_force_cancels_tasks_after_timeout(monkeypatch):
    agent = make_shutdown_agent()
    agent.mcp_handler = SimpleNamespace(_cleanup_mcp_servers=lambda: None)
    agent.agentfield_handler = SimpleNamespace(stop_heartbeat=lambda: None)
    server = AgentServer(agent)

    task = asyncio.create_task(asyncio.sleep(60))
    server._in_flight_tasks.update({task})

    monkeypatch.setattr(
        "agentfield.agent_server.clear_current_agent", lambda: None, raising=False
    )
    monkeypatch.setattr(
        "agentfield.agent_server.asyncio.sleep", AsyncMock(return_value=None)
    )
    monkeypatch.setattr(
        "agentfield.agent_server.os._exit",
        lambda code: (_ for _ in ()).throw(ExitCalled(code)),
    )

    with pytest.raises(ExitCalled):
        await server._graceful_shutdown(timeout_seconds=0)

    assert task.cancelled()


@pytest.mark.asyncio
async def test_track_task_adds_and_removes_task_on_completion():
    server = AgentServer(make_shutdown_agent())
    release = asyncio.Event()

    async def worker():
        await release.wait()

    task = asyncio.create_task(worker())
    tracked = server._track_task(task)

    assert tracked is task
    assert task in server._in_flight_tasks

    release.set()
    await task
    await asyncio.sleep(0)

    assert task not in server._in_flight_tasks
