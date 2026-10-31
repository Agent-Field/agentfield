import asyncio
import json
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

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
@pytest.mark.parametrize(
    "pattern, nonmatching_key, matching_key",
    [
        ("config.*", "configXtheme", "config.theme"),
        ("user+.*", "userr.preferences", "user+.preferences"),
        ("queue[0].*", "queue0.pending", "queue[0].pending"),
    ],
)
async def test_memory_event_client_dispatches_literal_key_patterns(
    monkeypatch, pattern, nonmatching_key, matching_key
):
    ctx = SimpleNamespace(to_headers=lambda: {})
    client = MemoryEventClient("http://agentfield", ctx)
    incoming = asyncio.Queue()
    handled = asyncio.Event()
    received = []

    class DummyWebSocket:
        open = True

        async def recv(self):
            message = await incoming.get()
            if message is None:
                return json.dumps({"scope": "global", "key": "closed", "action": "set"})
            return message

        async def close(self):
            self.open = False
            incoming.put_nowait(None)

    websocket = DummyWebSocket()

    async def fake_connect(url, **kwargs):
        return websocket

    async def callback(event):
        received.append(event.key)

    async def observe(event):
        if event.key == matching_key:
            handled.set()

    monkeypatch.setattr("agentfield.memory_events.websockets.connect", fake_connect)

    await client.connect()
    client.subscribe(pattern, callback)
    client.subscribe("*", observe)
    try:
        for key in (nonmatching_key, matching_key):
            incoming.put_nowait(
                json.dumps({"scope": "global", "key": key, "action": "set"})
            )
        await asyncio.wait_for(handled.wait(), timeout=1)
        assert received == [matching_key]
    finally:
        await client.close()
        assert not websocket.open


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
@pytest.mark.parametrize(
    ("patterns", "scope", "scope_id"),
    [
        (["cart.*", "order.*"], "session", "abc"),
        (["user+plan.*"], "actor", "alice+trial@example.com"),
        (["cart&total.*", "issue#*"], "session", "team&env=prod#1"),
        (["user preferences.*", "用户.*"], "workflow", "会话 + 1"),
    ],
)
async def test_memory_event_client_connect_builds_ws_url(
    monkeypatch, patterns, scope, scope_id
):
    ctx = SimpleNamespace(to_headers=lambda: {"Authorization": "token"})
    client = MemoryEventClient("http://agentfield", ctx)

    record = {}
    listener_called = {}

    class DummyWebSocket:
        def __init__(self):
            self.open = True

    async def fake_connect(url, **kwargs):
        record["url"] = url
        record["headers"] = kwargs.get("additional_headers") or kwargs.get(
            "extra_headers"
        )
        return DummyWebSocket()

    async def fake_listen(self):
        listener_called["run"] = True

    monkeypatch.setattr("agentfield.memory_events.websockets.connect", fake_connect)
    monkeypatch.setattr(MemoryEventClient, "_listen", fake_listen, raising=False)

    await client.connect(patterns=patterns, scope=scope, scope_id=scope_id)
    await asyncio.sleep(0)

    parsed_url = urlsplit(record["url"])
    assert parsed_url.scheme == "ws"
    assert parsed_url.netloc == "agentfield"
    assert parsed_url.path == "/api/v1/memory/events/ws"
    assert parsed_url.fragment == ""
    assert parse_qs(parsed_url.query) == {
        "patterns": [",".join(patterns)],
        "scope": [scope],
        "scope_id": [scope_id],
    }
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
async def test_close_does_not_reconnect_after_listener_receives_close(monkeypatch):
    client = MemoryEventClient(
        "http://agentfield", SimpleNamespace(to_headers=lambda: {})
    )
    client._reconnect_delay = 0
    sockets = []
    tasks = []
    create_task = asyncio.create_task

    class Socket:
        open = True

        def __init__(self):
            self.receiving = asyncio.Event()
            self.closed = asyncio.Event()

        async def recv(self):
            self.receiving.set()
            await self.closed.wait()
            raise websockets.exceptions.ConnectionClosedOK(None, None)

        async def close(self):
            self.open = False
            self.closed.set()

    async def connect(url, **kwargs):
        socket = Socket()
        sockets.append(socket)
        return socket

    def capture_task(coro, *args, **kwargs):
        task = create_task(coro, *args, **kwargs)
        tasks.append(task)
        return task

    monkeypatch.setattr("agentfield.memory_events.websockets.connect", connect)
    monkeypatch.setattr(asyncio, "create_task", capture_task)
    try:
        await client.connect()
        await asyncio.wait_for(sockets[0].receiving.wait(), timeout=1)
        listener = tasks[0]

        await client.close()
        await asyncio.wait_for(
            asyncio.gather(listener, return_exceptions=True), timeout=1
        )

        assert len(sockets) == 1
        assert sockets[0].closed.is_set()
        assert client.websocket is None
        assert not client.is_listening
    finally:
        for socket in sockets:
            await socket.close()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_close_discards_connection_that_finishes_after_shutdown(monkeypatch):
    client = MemoryEventClient(
        "http://agentfield", SimpleNamespace(to_headers=lambda: {})
    )
    connecting = asyncio.Event()
    release = asyncio.Event()
    closed = asyncio.Event()

    class Socket:
        open = True

        async def recv(self):
            await asyncio.Event().wait()

        async def close(self):
            self.open = False
            closed.set()

    async def connect(url, **kwargs):
        connecting.set()
        await release.wait()
        return Socket()

    monkeypatch.setattr("agentfield.memory_events.websockets.connect", connect)
    pending = asyncio.create_task(client.connect())
    try:
        await asyncio.wait_for(connecting.wait(), timeout=1)
        await client.close()
        release.set()
        await asyncio.wait_for(pending, timeout=1)

        assert closed.is_set()
        assert client.websocket is None
        assert not client.is_listening
    finally:
        release.set()
        await pending
        await client.close()


@pytest.mark.asyncio
async def test_close_discards_connection_queued_behind_inflight_connect(monkeypatch):
    client = MemoryEventClient(
        "http://agentfield", SimpleNamespace(to_headers=lambda: {})
    )
    connecting = asyncio.Event()
    queued = asyncio.Event()
    release = asyncio.Event()
    attempts = []

    class Socket:
        open = True

        async def recv(self):
            await asyncio.Event().wait()

        async def close(self):
            self.open = False

    async def connect(url, **kwargs):
        socket = Socket()
        attempts.append(socket)
        connecting.set()
        await release.wait()
        return socket

    async def connect_again():
        queued.set()
        await client.connect()

    monkeypatch.setattr("agentfield.memory_events.websockets.connect", connect)
    pending = [asyncio.create_task(client.connect())]
    try:
        await asyncio.wait_for(connecting.wait(), timeout=1)
        pending.append(asyncio.create_task(connect_again()))
        await asyncio.wait_for(queued.wait(), timeout=1)

        await client.close()
        release.set()
        await asyncio.wait_for(asyncio.gather(*pending), timeout=1)

        assert len(attempts) == 1
        assert not attempts[0].open
        assert client.websocket is None
        assert not client.is_listening
    finally:
        release.set()
        await asyncio.gather(*pending, return_exceptions=True)
        await client.close()


@pytest.mark.asyncio
async def test_close_cancels_retry_backoff(monkeypatch):
    client = MemoryEventClient(
        "http://agentfield", SimpleNamespace(to_headers=lambda: {})
    )
    sleeping = asyncio.Event()
    stopped = asyncio.Event()
    release = asyncio.Event()
    attempts = []

    async def connect(url, **kwargs):
        attempts.append(url)
        raise ConnectionRefusedError("offline")

    async def sleep(delay):
        sleeping.set()
        try:
            await release.wait()
        finally:
            stopped.set()

    monkeypatch.setattr("agentfield.memory_events.websockets.connect", connect)
    monkeypatch.setattr("agentfield.memory_events.asyncio.sleep", sleep)
    try:
        await client.connect()
        await asyncio.wait_for(sleeping.wait(), timeout=1)

        await client.close()

        assert stopped.is_set()
        assert len(attempts) == 1
        assert client.websocket is None
    finally:
        release.set()
        await client.close()


@pytest.mark.asyncio
async def test_cancelled_close_still_closes_transport(monkeypatch):
    client = MemoryEventClient(
        "http://agentfield", SimpleNamespace(to_headers=lambda: {})
    )
    receiving = asyncio.Event()
    stopping = asyncio.Event()
    release = asyncio.Event()
    closed = asyncio.Event()

    class Socket:
        open = True

        async def recv(self):
            receiving.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopping.set()
                await release.wait()

        async def close(self):
            self.open = False
            closed.set()

    socket = Socket()

    async def connect(url, **kwargs):
        return socket

    monkeypatch.setattr("agentfield.memory_events.websockets.connect", connect)
    await client.connect()
    await asyncio.wait_for(receiving.wait(), timeout=1)
    closing = asyncio.create_task(client.close())
    try:
        await asyncio.wait_for(stopping.wait(), timeout=1)
        closing.cancel()
        with pytest.raises(asyncio.CancelledError):
            await closing

        assert closed.is_set()
    finally:
        release.set()
        if not closing.done():
            closing.cancel()
        await asyncio.gather(closing, return_exceptions=True)
        await client.close()
        await socket.close()


@pytest.mark.asyncio
async def test_client_can_explicitly_reconnect_after_close():
    received = []
    events = asyncio.Queue()
    connections = 0

    async def handler(socket, path=None):
        nonlocal connections
        connections += 1
        await socket.send(
            json.dumps(
                {"scope": "global", "key": f"connection.{connections}", "action": "set"}
            )
        )
        await socket.wait_closed()

    async def callback(event):
        received.append(event.key)
        events.put_nowait(event.key)

    async with websockets.serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        client = MemoryEventClient(
            f"http://127.0.0.1:{port}", SimpleNamespace(to_headers=lambda: {})
        )
        try:
            for key in ("connection.1", "connection.2"):
                await client.connect()
                client.subscribe("connection.*", callback)
                assert await asyncio.wait_for(events.get(), timeout=1) == key
                await client.close()
        finally:
            await client.close()

    assert received == ["connection.1", "connection.2"]


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
        called_with["url"] = url
        return DummyWebSocket()

    monkeypatch.setattr("agentfield.memory_events.websockets.connect", fake_connect)
    monkeypatch.setattr(MemoryEventClient, "_listen", lambda self: asyncio.sleep(0))

    await client.connect()
    await asyncio.sleep(0)

    assert _HEADERS_KWARG in called_with
    assert called_with[_HEADERS_KWARG] == {"Authorization": "token"}
    assert called_with["url"] == "ws://agentfield/api/v1/memory/events/ws"


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

    assert reconnect_started.is_set(), (
        "reconnect should have been started in background"
    )
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
