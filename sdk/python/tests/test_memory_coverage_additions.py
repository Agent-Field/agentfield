from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from agentfield.memory import GlobalMemoryClient, MemoryClient, MemoryInterface, _vector_to_list


@pytest.fixture
def execution_context():
    return SimpleNamespace(to_headers=lambda: {"X-Workflow-ID": "wf-1"})


@pytest.fixture
def memory_client(execution_context):
    return MemoryClient(
        agentfield_client=SimpleNamespace(api_base="http://agentfield.test/api/v1"),
        execution_context=execution_context,
        agent_node_id="node-7",
    )


def test_vector_to_list_normalizes_tolist_and_scalars():
    class Vector:
        def tolist(self):
            return (1, "2.5", 3)

    assert _vector_to_list(Vector()) == [1.0, 2.5, 3.0]


def test_build_headers_adds_agent_node_and_scope_override(memory_client):
    headers = memory_client._build_headers(scope="session", scope_id="sess-1")

    assert headers["X-Workflow-ID"] == "wf-1"
    assert headers["X-Agent-Node-ID"] == "node-7"
    assert headers["X-Session-ID"] == "sess-1"


@pytest.mark.asyncio
async def test_get_returns_raw_string_when_data_is_not_json(memory_client, monkeypatch):
    response = SimpleNamespace(
        status_code=200,
        raise_for_status=lambda: None,
        json=lambda: {"data": "plain-text"},
    )
    monkeypatch.setattr(memory_client, "_async_request", AsyncMock(return_value=response))

    result = await memory_client.get("greeting")

    assert result == "plain-text"


@pytest.mark.asyncio
async def test_get_wraps_transport_errors(memory_client, monkeypatch):
    monkeypatch.setattr(
        memory_client,
        "_async_request",
        AsyncMock(side_effect=RuntimeError("network broke")),
    )

    with pytest.raises(Exception, match="Failed to get memory key 'missing'"):
        await memory_client.get("missing")


def test_global_scope_on_change_without_event_client_is_noop(memory_client):
    global_client = GlobalMemoryClient(memory_client, event_client=None)

    @global_client.on_change("profile.*")
    async def listener(event):
        return event

    assert listener.__name__ == "listener"


@pytest.mark.asyncio
async def test_global_scope_on_change_registers_wrapper_metadata(memory_client):
    subscribed = {}
    event_client = Mock()

    def subscribe(patterns, callback, scope=None, scope_id=None):
        subscribed["patterns"] = patterns
        subscribed["callback"] = callback
        subscribed["scope"] = scope
        subscribed["scope_id"] = scope_id

    event_client.subscribe = subscribe
    global_client = GlobalMemoryClient(memory_client, event_client=event_client)

    @global_client.on_change(["profile.*", "prefs.*"])
    async def listener(event):
        return {"event": event}

    assert subscribed["patterns"] == ["profile.*", "prefs.*"]
    assert subscribed["scope"] == "global"
    assert subscribed["scope_id"] is None
    assert getattr(listener, "_memory_event_listener") is True
    assert getattr(listener, "_memory_event_patterns") == ["profile.*", "prefs.*"]
    assert await subscribed["callback"]("changed") == {"event": "changed"}


def test_memory_interface_scope_helpers_and_global_scope(memory_client):
    events = Mock()
    interface = MemoryInterface(memory_client, events)

    session_client = interface.session("sess-1")
    actor_client = interface.actor("actor-1")
    workflow_client = interface.workflow("wf-2")
    global_client = interface.global_scope

    assert session_client.scope == "session"
    assert session_client.scope_id == "sess-1"
    assert actor_client.scope == "actor"
    assert workflow_client.scope == "workflow"
    assert global_client.event_client is events

