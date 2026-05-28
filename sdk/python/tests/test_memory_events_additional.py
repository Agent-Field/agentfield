import asyncio
import builtins
from datetime import datetime
from types import SimpleNamespace

import pytest

import agentfield.memory_events as memory_events_module
from agentfield.memory_events import (
    MemoryEventClient,
    PatternMatcher,
    ScopedMemoryEventClient,
)
from agentfield.types import MemoryChangeEvent


@pytest.fixture
def execution_context():
    return SimpleNamespace(to_headers=lambda: {"Authorization": "token"})


def test_pattern_matcher_invalid_regex_falls_back_to_exact_match():
    assert PatternMatcher.matches_pattern("[", "[") is True
    assert PatternMatcher.matches_pattern("[", "different") is False


def test_memory_event_client_is_connected_variants(execution_context):
    client = MemoryEventClient("http://agentfield", execution_context)
    assert client._is_connected() is False

    client.websocket = SimpleNamespace(open=True)
    assert client._is_connected() is True

    client.websocket = SimpleNamespace(open=None, closed=True)
    assert client._is_connected() is False

    client.websocket = SimpleNamespace(open=None, closed=None)
    assert client._is_connected() is True


@pytest.mark.asyncio
async def test_listen_handles_connection_closed_and_generic_errors(monkeypatch, execution_context):
    client = MemoryEventClient("http://agentfield", execution_context)
    reconnect_calls = []
    async def fake_reconnect():
        reconnect_calls.append("reconnect")

    monkeypatch.setattr(client, "_handle_reconnect", fake_reconnect)

    class ClosedError(Exception):
        pass

    monkeypatch.setattr(
        memory_events_module.websockets,
        "exceptions",
        SimpleNamespace(ConnectionClosed=ClosedError),
        raising=False,
    )
    client.websocket = SimpleNamespace(recv=lambda: (_ for _ in ()).throw(ClosedError()), open=True)
    client.is_listening = True
    await client._listen()
    assert client.websocket is None
    assert reconnect_calls == ["reconnect"]

    reconnect_calls.clear()

    class ClosingSocket:
        def __init__(self):
            self.closed = False

        async def recv(self):
            raise RuntimeError("bad recv")

        async def close(self):
            self.closed = True

    socket = ClosingSocket()
    client.websocket = socket
    client.is_listening = True
    monkeypatch.setattr("agentfield.memory_events.log_error", lambda message: reconnect_calls.append(message))
    async def fake_retry():
        reconnect_calls.append("retry")

    monkeypatch.setattr(client, "_handle_reconnect", fake_retry)
    await client._listen()

    assert socket.closed is True
    assert client.websocket is None
    assert any("Error in event listener: bad recv" == entry for entry in reconnect_calls)
    assert "retry" in reconnect_calls


@pytest.mark.asyncio
async def test_handle_reconnect_logs_max_and_failed_reconnect(monkeypatch, execution_context):
    client = MemoryEventClient("http://agentfield", execution_context)
    logged = []
    monkeypatch.setattr("agentfield.memory_events.log_error", logged.append)
    monkeypatch.setattr("agentfield.memory_events.log_info", lambda message: logged.append(message))
    original_sleep = asyncio.sleep
    monkeypatch.setattr("agentfield.memory_events.asyncio.sleep", lambda delay: original_sleep(0))

    client._reconnect_attempts = client._max_reconnect_attempts
    await client._handle_reconnect()
    assert logged == [f"Max reconnection attempts reached ({client._max_reconnect_attempts})"]

    logged.clear()
    client._reconnect_attempts = 0

    async def failing_connect():
        raise RuntimeError("connect boom")

    monkeypatch.setattr(client, "connect", failing_connect)
    await client._handle_reconnect()
    assert any("Reconnecting to memory events" in entry for entry in logged)
    assert "Reconnection failed: connect boom" in logged


@pytest.mark.asyncio
async def test_subscribe_on_change_close_and_scoped_client(monkeypatch, execution_context):
    client = MemoryEventClient("http://agentfield", execution_context)
    scheduled = []

    def fake_create_task(coro):
        scheduled.append(coro)
        coro.close()
        return SimpleNamespace()

    monkeypatch.setattr("agentfield.memory_events.asyncio.create_task", fake_create_task)

    async def callback(event):
        return event.key

    subscription = client.subscribe("cart.*", callback)
    assert subscription.patterns == ["cart.*"]
    assert len(scheduled) == 1

    @client.on_change(["order.*"])
    async def wrapped(event):
        return event.scope_id

    event = MemoryChangeEvent(scope="session", scope_id="s1", key="order.id", action="set")
    assert await wrapped(event) == "s1"
    assert wrapped._memory_event_listener is True
    assert wrapped._memory_event_patterns == ["order.*"]

    socket = SimpleNamespace(closed=False)

    async def close_socket():
        socket.closed = True

    socket.close = close_socket
    client.websocket = socket
    client.subscriptions = [subscription]
    await client.close()
    assert socket.closed is True
    assert client.subscriptions == []

    scoped = ScopedMemoryEventClient(client, "agent", "a1")

    @scoped.on_change("profile.*")
    async def scoped_handler(event):
        return event.key

    assert scoped_handler._memory_event_scope == "agent"
    assert scoped_handler._memory_event_scope_id == "a1"

    recorded = {}

    async def fake_history(**kwargs):
        recorded.update(kwargs)
        return ["history"]

    monkeypatch.setattr(client, "history", fake_history)
    assert await scoped.history(patterns="profile.*", limit=2) == ["history"]
    assert recorded["scope"] == "agent"
    assert recorded["scope_id"] == "a1"


@pytest.mark.asyncio
async def test_history_handles_parse_errors_importerror_fallback_and_request_failures(monkeypatch, execution_context):
    client = MemoryEventClient("http://agentfield", execution_context, api_key="secret")
    logged = []
    monkeypatch.setattr("agentfield.memory_events.log_error", logged.append)

    class AsyncResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return [
                {"scope": "session", "scope_id": "s1", "key": "cart.total", "action": "set"},
                [],
            ]

    class AsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, url, params=None, headers=None, timeout=None):
            assert params["patterns"] == "cart.*"
            assert params["scope"] == "session"
            assert params["scope_id"] == "s1"
            assert headers["X-API-Key"] == "secret"
            return AsyncResponse()

    import httpx

    monkeypatch.setattr(httpx, "AsyncClient", AsyncClient)
    since = datetime(2024, 1, 1)
    events = await client.history(
        patterns="cart.*",
        since=since,
        limit=5,
        scope="session",
        scope_id="s1",
    )
    assert [event.key for event in events] == ["cart.total"]
    assert any("Failed to parse event" in entry for entry in logged)

    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "httpx":
            raise ImportError("no httpx")
        return original_import(name, *args, **kwargs)

    class RequestsResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return [{"scope": "agent", "scope_id": "a1", "key": "profile.name", "action": "set"}]

    requests_module = SimpleNamespace(
        get=lambda url, params=None, headers=None, timeout=None: RequestsResponse()
    )
    monkeypatch.setitem(__import__("sys").modules, "requests", requests_module)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    sync_events = await client.history(patterns=["profile.*"], limit=1)
    monkeypatch.setattr(builtins, "__import__", original_import)
    assert [event.key for event in sync_events] == ["profile.name"]

    class FailingClient:
        async def __aenter__(self):
            raise RuntimeError("boom")

        async def __aexit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(httpx, "AsyncClient", FailingClient)
    assert await client.history(limit=1) == []
    assert "Failed to get event history: boom" in logged
