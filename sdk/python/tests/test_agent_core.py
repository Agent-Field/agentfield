import asyncio
import sys
from types import SimpleNamespace

import pytest

from agentfield.agent import Agent
from agentfield.agent_registry import get_current_agent_instance
from agentfield.execution_context import (
    ExecutionContext,
    set_execution_context,
    reset_execution_context,
)


def make_agent_stub():
    agent = object.__new__(Agent)
    agent.node_id = "node"
    agent.agentfield_server = "http://agentfield"
    agent.dev_mode = False
    agent.async_config = SimpleNamespace(
        enable_async_execution=True, fallback_to_sync=True
    )
    agent._async_execution_manager = None
    agent._current_execution_context = None
    agent.client = SimpleNamespace(
        api_base="http://agentfield/api/v1",
        _get_auth_headers=lambda: {},
    )
    agent._background_tasks = set()
    return agent


def test_get_current_execution_context_creates_and_reuses():
    agent = make_agent_stub()
    ctx1 = agent._get_current_execution_context()
    assert isinstance(ctx1, ExecutionContext)
    assert agent._current_execution_context is ctx1

    # Thread-local context should override agent-level
    token = set_execution_context(ctx1)
    try:
        ctx2 = agent._get_current_execution_context()
        assert ctx2 is ctx1
    finally:
        reset_execution_context(token)

    # Clearing agent-level should create new context
    agent._current_execution_context = None
    ctx3 = agent._get_current_execution_context()
    assert ctx3 is not ctx1


def test_set_as_current_updates_agent_registry():
    agent = make_agent_stub()

    agent._clear_current()
    assert get_current_agent_instance() is None

    agent._set_as_current()
    assert get_current_agent_instance() is agent

    agent._clear_current()
    assert get_current_agent_instance() is None


@pytest.mark.asyncio
async def test_cleanup_async_resources(monkeypatch):
    agent = make_agent_stub()

    class DummyManager:
        def __init__(self):
            self.stopped = False

        async def stop(self):
            self.stopped = True

    class DummyNotificationDispatcher:
        def __init__(self):
            self.stopped = False

        async def shutdown(self):
            self.stopped = True

    async def dummy_async_task(sleep_delay: int):
        await asyncio.sleep(sleep_delay)

    manager = DummyManager()
    notification_dispatcher = DummyNotificationDispatcher()
    agent._async_execution_manager = manager
    agent._notification_dispatcher = notification_dispatcher

    for i in range(1, 6):
        task = asyncio.create_task(dummy_async_task(i))
        agent._background_tasks.add(task)
        task.add_done_callback(agent._background_tasks.discard)

    await agent._cleanup_async_resources()
    assert manager.stopped is True
    assert agent._async_execution_manager is None
    assert len(agent._background_tasks) == 0
    assert notification_dispatcher.stopped is True


@pytest.mark.asyncio
async def test_note_sends_async_request(monkeypatch):
    agent = make_agent_stub()

    called = {}

    class DummyTimeout:
        def __init__(self, total):
            self.total = total

    class DummySession:
        def __init__(self, timeout):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def post(self, url, json=None, headers=None):
            called["url"] = url
            called["json"] = json
            called["headers"] = headers

            class DummyResponse:
                status = 200

                async def __aenter__(self_inner):
                    return self_inner

                async def __aexit__(self_inner, exc_type, exc, tb):
                    return False

                async def text(self_inner):
                    return "ok"

            return DummyResponse()

    stub_aiohttp = SimpleNamespace(
        ClientTimeout=DummyTimeout, ClientSession=DummySession
    )
    monkeypatch.setitem(sys.modules, "aiohttp", stub_aiohttp)
    monkeypatch.setattr("agentfield.agent.aiohttp", stub_aiohttp)

    context = SimpleNamespace(to_headers=lambda: {"X-Workflow-ID": "wf"})
    monkeypatch.setattr(agent, "_get_current_execution_context", lambda: context)

    agent.note("hello", tags=["debug"])
    # fire_and_forget creates a task on the running loop; give it a tick.
    await asyncio.sleep(0.1)

    assert called["url"].startswith("http://agentfield/api/v1")
    assert called["json"]["message"] == "hello"
    assert called["json"]["tags"] == ["debug"]


def _agent_stub_for_del(cleanup_coro_factory):
    """Bare Agent instance carrying only what ``Agent.__del__`` touches."""
    agent = object.__new__(Agent)
    agent._async_execution_manager = SimpleNamespace(name="manager")
    agent._cleanup_async_resources = cleanup_coro_factory
    return agent


def test_del_runs_cleanup_to_completion_when_no_loop_is_running():
    """C1: with no running event loop (the destructor-at-exit case), cleanup
    must have finished by the time __del__ returns — handing it to a daemon
    thread would let the interpreter kill it before it does any work."""
    state = {"finished": False}

    async def cleanup():
        # Yield at least once so a half-run coroutine would be observable.
        await asyncio.sleep(0)
        state["finished"] = True

    agent = _agent_stub_for_del(cleanup)

    Agent.__del__(agent)

    assert state["finished"] is True


@pytest.mark.asyncio
async def test_del_schedules_cleanup_on_the_running_loop_without_blocking():
    """C2: with a loop already running, __del__ must not block it (and must
    not raise); the cleanup runs as soon as the loop gets control back."""
    state = {"finished": False}

    async def cleanup():
        state["finished"] = True

    agent = _agent_stub_for_del(cleanup)

    Agent.__del__(agent)

    # Still scheduled, not executed: the destructor handed off without
    # blocking the running loop.
    assert state["finished"] is False

    await asyncio.sleep(0.05)
    assert state["finished"] is True
