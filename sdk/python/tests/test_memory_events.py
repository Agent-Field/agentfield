import asyncio
import json
from types import SimpleNamespace

import websockets

import pytest

from agentfield.memory_events import (
    PatternMatcher,
    EventSubscription,
    MemoryEventClient,
)
from agentfield.types import MemoryChangeEvent


def test_pattern_matcher_wildcards():
    assert PatternMatcher.matches_pattern("customer_*", "customer_123")
    assert PatternMatcher.matches_pattern("order.*.status", "order.45.status")
    assert not PatternMatcher.matches_pattern("user_*", "device_1")


def test_event_subscription_matches_scoped_event():
    event = MemoryChangeEvent(
        scope="session",
        scope_id="s1",
        key="cart.total",
        action="set",
        data=42,
    )
    sub = EventSubscription(["cart.*"], lambda e: None, scope="session", scope_id="s1")
    assert sub.matches_event(event) is True

    other = MemoryChangeEvent(
        scope="session", scope_id="s2", key="cart.total", action="set"
    )
    assert sub.matches_event(other) is False


def test_memory_event_client_subscription_and_unsubscribe(monkeypatch):
    ctx = SimpleNamespace(to_headers=lambda: {"Authorization": "token"})
    client = MemoryEventClient("http://agentfield", ctx)

    callback_called = asyncio.Event()

    async def callback(event):
        callback_called.set()

    client.websocket = SimpleNamespace(open=True)

    sub = client.subscribe(["order.*"], callback)
    assert sub in client.subscriptions

    sub.unsubscribe()
    client.unsubscribe_all()
    assert client.subscriptions == []


@pytest.mark.asyncio
async def test_memory_event_client_history(monkeypatch):
    ctx = SimpleNamespace(to_headers=lambda: {"Authorization": "token"})
    client = MemoryEventClient("http://agentfield", ctx)

    class DummyResponse:
        def __init__(self):
            self._json = [
                {
                    "scope": "session",
                    "scope_id": "s1",
                    "key": "cart.total",
                    "action": "set",
                    "data": 10,
                }
            ]

        def json(self):
            return self._json

        def raise_for_status(self):
            return None

    class DummyAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, url, params=None, headers=None, timeout=None):
            return DummyResponse()

    import httpx

    monkeypatch.setattr(httpx, "AsyncClient", DummyAsyncClient, raising=True)

    events = await client.history(patterns="cart.*", limit=1)
    assert len(events) == 1
    assert events[0].key == "cart.total"


@pytest.mark.asyncio
async def test_memory_event_client_connect_builds_ws_url(monkeypatch):
    ctx = SimpleNamespace(to_headers=lambda: {"Authorization": "token"})
    client = MemoryEventClient("http://agentfield", ctx)

    record = {}
    listener_called = {}

    class DummyWebSocket:
        def __init__(self):
            self.open = True

    async def fake_connect(url, **kwargs):
        record["url"] = url
        record["headers"] = kwargs.get("additional_headers") or kwargs.get("extra_headers")
        return DummyWebSocket()

    async def fake_listen(self):
        listener_called["run"] = True

    monkeypatch.setattr("agentfield.memory_events.websockets.connect", fake_connect)
    monkeypatch.setattr(MemoryEventClient, "_listen", fake_listen, raising=False)

    await client.connect(
        patterns=["cart.*", "order.*"], scope="session", scope_id="abc"
    )
    await asyncio.sleep(0)

    assert record["url"].startswith("ws://agentfield")
    assert "patterns=cart.*,order.*" in record["url"]
    assert "scope=session" in record["url"]
    assert "scope_id=abc" in record["url"]
    assert record["headers"] == {"Authorization": "token"}
    assert listener_called.get("run") is True


def test_websockets_version_detection():
    """Verify version detection picks the correct header kwarg for the installed websockets."""
    from agentfield.memory_events import _WEBSOCKETS_MAJOR, _HEADERS_KWARG

    major = int(websockets.__version__.split(".")[0])
    assert _WEBSOCKETS_MAJOR == major

    if major >= 14:
        assert _HEADERS_KWARG == "additional_headers"
    else:
        assert _HEADERS_KWARG == "extra_headers"


@pytest.mark.asyncio
async def test_connect_passes_correct_headers_kwarg(monkeypatch):
    """Verify connect() uses the version-appropriate header parameter."""
    from agentfield.memory_events import _HEADERS_KWARG

    ctx = SimpleNamespace(to_headers=lambda: {"Authorization": "token"})
    client = MemoryEventClient("http://agentfield", ctx)

    called_with = {}

    class DummyWebSocket:
        def __init__(self):
            self.open = True

    async def fake_connect(url, **kwargs):
        called_with.update(kwargs)
        return DummyWebSocket()

    monkeypatch.setattr("agentfield.memory_events.websockets.connect", fake_connect)
    monkeypatch.setattr(MemoryEventClient, "_listen", lambda self: asyncio.sleep(0))

    await client.connect()
    await asyncio.sleep(0)

    assert _HEADERS_KWARG in called_with
    assert called_with[_HEADERS_KWARG] == {"Authorization": "token"}


@pytest.mark.asyncio
async def test_connect_does_not_block_startup_on_failure(monkeypatch):
    """When connection fails, reconnect retries run in the background."""
    ctx = SimpleNamespace(to_headers=lambda: {})
    client = MemoryEventClient("http://agentfield", ctx)

    async def failing_connect(url, **kwargs):
        raise ConnectionRefusedError("server unavailable")

    reconnect_started = asyncio.Event()

    async def fake_reconnect(self):
        reconnect_started.set()

    monkeypatch.setattr("agentfield.memory_events.websockets.connect", failing_connect)
    monkeypatch.setattr(MemoryEventClient, "_handle_reconnect", fake_reconnect)

    # connect() should return immediately, not block on retries
    await client.connect()

    # Give the background task a chance to start
    await asyncio.sleep(0.05)

    assert reconnect_started.is_set(), "reconnect should have been started in background"
    assert not client.is_listening


@pytest.mark.asyncio
async def test_memory_event_client_listen_dispatches(monkeypatch):
    ctx = SimpleNamespace(to_headers=lambda: {})
    client = MemoryEventClient("http://agentfield", ctx)

    received = []

    async def callback(event):
        received.append((event.key, event.data))
        client.is_listening = False  # stop after first event

    client.subscriptions.append(EventSubscription(["order.*"], callback))

    class DummyWebSocket:
        def __init__(self, messages):
            self._messages = messages
            self.open = True

        async def recv(self):
            if self._messages:
                return self._messages.pop(0)
            while client.is_listening:
                await asyncio.sleep(0)
            raise websockets.exceptions.ConnectionClosed(1000, "closed")

    message = json.dumps(
        {
            "scope": "session",
            "scope_id": "s1",
            "key": "order.total",
            "action": "set",
            "data": 99,
        }
    )

    client.websocket = DummyWebSocket([message])
    client.is_listening = True
    monkeypatch.setattr(client, "_handle_reconnect", lambda: asyncio.sleep(0))

    tasks = []
    original_create_task = asyncio.create_task

    def capture_task(coro):
        task = original_create_task(coro)
        tasks.append(task)
        return task

    monkeypatch.setattr(asyncio, "create_task", capture_task)

    await client._listen()
    if tasks:
        await asyncio.gather(*tasks)

    assert received == [("order.total", 99)]


@pytest.mark.asyncio
async def test_memory_event_client_handle_reconnect(monkeypatch):
    ctx = SimpleNamespace(to_headers=lambda: {})
    client = MemoryEventClient("http://agentfield", ctx)
    client._max_reconnect_attempts = 2

    sleeps = []
    connects = []

    async def fake_sleep(delay):
        sleeps.append(delay)

    async def fake_connect(*args, **kwargs):
        connects.append("connect")

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(client, "connect", fake_connect)

    await client._handle_reconnect()

    assert sleeps == [1.0]
    assert connects == ["connect"]

    # Subsequent call should stop once max attempts reached
    client._reconnect_attempts = client._max_reconnect_attempts
    await client._handle_reconnect()
    assert len(connects) == 1  # no new connect call


def test_on_change_decorator_marks_wrapper():
    ctx = SimpleNamespace(to_headers=lambda: {})
    client = MemoryEventClient("http://agentfield", ctx)
    client.websocket = SimpleNamespace(open=True)

    @client.on_change("foo.*")
    async def handle(event):
        return event

    assert getattr(handle, "_memory_event_listener") is True
    assert client.subscriptions[0].patterns == ["foo.*"]
