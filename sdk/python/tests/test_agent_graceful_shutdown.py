import asyncio
import os
import signal
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from agentfield.agent import Agent
from agentfield.agent_field_handler import AgentFieldHandler
from agentfield.agent_server import AgentServer
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


def test_fast_lifecycle_signal_handler_marks_shutdown_and_notifies(monkeypatch):
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
