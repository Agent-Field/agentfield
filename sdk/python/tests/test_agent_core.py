import asyncio
import sys
from types import SimpleNamespace

import pytest

from haxen_sdk.agent import Agent
from haxen_sdk.agent_registry import get_current_agent_instance
from haxen_sdk.execution_context import (
    ExecutionContext,
    set_execution_context,
    reset_execution_context,
)


def make_agent_stub():
    agent = object.__new__(Agent)
    agent.node_id = "node"
    agent.haxen_server = "http://haxen"
    agent.dev_mode = False
    agent.async_config = SimpleNamespace(
        enable_async_execution=True, fallback_to_sync=True
    )
    agent._async_execution_manager = None
    agent._current_execution_context = None
    agent.client = SimpleNamespace(api_base="http://haxen/api/v1")
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

    manager = DummyManager()
    agent._async_execution_manager = manager
    await agent._cleanup_async_resources()
    assert manager.stopped is True
    assert agent._async_execution_manager is None


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
    monkeypatch.setattr("haxen_sdk.agent.aiohttp", stub_aiohttp)

    context = SimpleNamespace(to_headers=lambda: {"X-Workflow-ID": "wf"})
    monkeypatch.setattr(agent, "_get_current_execution_context", lambda: context)

    tasks = []

    class DummyLoop:
        def is_running(self):
            return True

        def create_task(self, coro):
            task = asyncio.ensure_future(coro)
            tasks.append(task)
            return task

    monkeypatch.setattr("asyncio.get_event_loop", lambda: DummyLoop())

    agent.note("hello", tags=["debug"])
    await asyncio.gather(*tasks)

    assert called["url"].startswith("http://haxen/api/ui/v1")
    assert called["json"]["message"] == "hello"
    assert called["json"]["tags"] == ["debug"]
