import asyncio
import json
import sys
import types

import pytest
import requests

from haxen_sdk.client import HaxenClient
from haxen_sdk.types import AgentStatus, HeartbeatData


@pytest.fixture(autouse=True)
def ensure_event_loop():
    try:
        loop = asyncio.get_event_loop()
        had_loop = True
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        had_loop = False

    try:
        yield
    finally:
        if not had_loop:
            loop.close()
            asyncio.set_event_loop(None)


class DummyResponse:
    def __init__(self, payload, status_code=200, headers=None):
        self._payload = payload
        self.status_code = status_code
        self.headers = headers or {}
        try:
            self.content = json.dumps(payload).encode("utf-8")
        except Exception:
            self.content = b""
        self.headers.setdefault("Content-Length", str(len(self.content)))

    def json(self):
        return self._payload

    def raise_for_status(self):
        if not (200 <= self.status_code < 400):
            raise RuntimeError("bad status")


def install_httpx_stub(monkeypatch, *, on_request):
    class _DummyAsyncClient:
        def __init__(self, *args, **kwargs):
            self.is_closed = False

        async def request(self, method, url, **kwargs):
            return on_request(method, url, **kwargs)

        async def aclose(self):
            self.is_closed = True

    module = types.SimpleNamespace(
        AsyncClient=_DummyAsyncClient,
        Limits=lambda *args, **kwargs: None,
        Timeout=lambda *args, **kwargs: None,
    )

    import haxen_sdk.client as client_mod

    monkeypatch.setitem(sys.modules, "httpx", module)
    client_mod.httpx = module
    monkeypatch.setattr(
        client_mod, "_ensure_httpx", lambda force_reload=False: module, raising=False
    )
    return module


def test_execute_sync_injects_run_id(monkeypatch):
    captured = {}

    def fake_post(url, json, headers, timeout):
        captured["post"] = (url, headers)
        return DummyResponse(
            {
                "execution_id": "exec-1",
                "run_id": headers["X-Run-ID"],
                "status": "queued",
            },
            status_code=202,
        )

    def fake_get(url, headers=None, timeout=None):
        captured["get"] = (url, headers)
        return DummyResponse(
            {
                "execution_id": "exec-1",
                "run_id": headers["X-Run-ID"],
                "status": "succeeded",
                "result": {"ok": True},
                "duration_ms": 42,
            }
        )

    import haxen_sdk.client as client_mod

    monkeypatch.setattr(client_mod.requests, "post", fake_post)
    monkeypatch.setattr(client_mod.requests, "get", fake_get)

    client = HaxenClient(base_url="http://example.com")
    result = client.execute_sync("node.reasoner", {"payload": 1})

    assert result["status"] == "succeeded"
    post_url, post_headers = captured["post"]
    assert post_url.endswith("/api/v1/execute/async/node.reasoner")
    assert post_headers["Content-Type"] == "application/json"
    assert post_headers["X-Run-ID"].startswith("run_")
    get_url, get_headers = captured["get"]
    assert get_url.endswith("/api/v1/executions/exec-1")
    assert get_headers["X-Run-ID"] == post_headers["X-Run-ID"]


def test_execute_sync_respects_parent_header(monkeypatch):
    captured = {}

    def fake_post(url, json, headers, timeout):
        captured["post"] = headers
        return DummyResponse(
            {
                "execution_id": "exec-2",
                "run_id": headers["X-Run-ID"],
                "status": "queued",
            },
            status_code=202,
        )

    def fake_get(url, headers=None, timeout=None):
        captured["get"] = headers
        return DummyResponse(
            {
                "execution_id": "exec-2",
                "run_id": headers["X-Run-ID"],
                "status": "succeeded",
                "result": {"ok": True},
            }
        )

    import haxen_sdk.client as client_mod

    monkeypatch.setattr(client_mod.requests, "post", fake_post)
    monkeypatch.setattr(client_mod.requests, "get", fake_get)

    client = HaxenClient(base_url="http://example.com")
    result = client.execute_sync(
        "node.reasoner",
        {"payload": 1},
        headers={"X-Run-ID": "run-parent", "X-Parent-Execution-ID": "exec-parent"},
    )

    assert result["status"] == "succeeded"
    assert captured["post"]["X-Run-ID"] == "run-parent"
    assert captured["post"]["X-Parent-Execution-ID"] == "exec-parent"
    assert captured["get"]["X-Parent-Execution-ID"] == "exec-parent"


def test_execute_async_uses_httpx(monkeypatch):
    calls = []

    def on_request(method, url, **kwargs):
        calls.append((method, url))
        if method == "POST":
            return DummyResponse(
                {"execution_id": "exec-async", "run_id": "run-123", "status": "queued"},
                status_code=202,
            )
        return DummyResponse(
            {
                "execution_id": "exec-async",
                "run_id": "run-123",
                "status": "succeeded",
                "result": {"async": True},
            },
        )

    install_httpx_stub(monkeypatch, on_request=on_request)

    client = HaxenClient(base_url="http://example.com")
    result = asyncio.run(client.execute("node.reasoner", {"payload": 1}))

    assert result["result"] == {"async": True}
    assert calls[0][0] == "POST"
    assert calls[1][0] == "GET"


@pytest.mark.asyncio
async def test_execute_async_falls_back_to_requests(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "httpx":
            raise ImportError("no httpx")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.delitem(sys.modules, "httpx", raising=False)

    captured = {}

    def fake_post(url, json=None, headers=None, timeout=None, **kwargs):
        captured["post"] = headers
        return DummyResponse(
            {
                "execution_id": "exec-fallback",
                "run_id": headers["X-Run-ID"],
                "status": "queued",
            },
            status_code=202,
        )

    def fake_get(url, headers=None, timeout=None, **kwargs):
        captured["get"] = headers
        return DummyResponse(
            {
                "execution_id": "exec-fallback",
                "run_id": headers["X-Run-ID"],
                "status": "succeeded",
                "result": {"ok": True},
            }
        )

    import haxen_sdk.client as client_mod

    client_mod.httpx = None
    monkeypatch.setattr(
        client_mod, "_ensure_httpx", lambda force_reload=False: None, raising=False
    )
    monkeypatch.setattr(client_mod.requests, "post", fake_post)
    monkeypatch.setattr(client_mod.requests, "get", fake_get)
    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr(requests, "get", fake_get)

    def fake_session_request(self, method, url, **kwargs):
        if method.upper() == "POST":
            return fake_post(url, **kwargs)
        return fake_get(url, **kwargs)

    monkeypatch.setattr(requests.Session, "request", fake_session_request)

    client = HaxenClient(base_url="http://example.com")
    result = await client.execute("node.reasoner", {"payload": 1})

    assert result["status"] == "succeeded"
    assert captured["post"]["Content-Type"] == "application/json"
    assert captured["post"]["X-Run-ID"].startswith("run_")
    assert captured["get"]["X-Run-ID"] == captured["post"]["X-Run-ID"]


@pytest.mark.asyncio
async def test_async_heartbeat(monkeypatch):
    calls = []

    def on_request(method, url, **kwargs):
        calls.append((method, url))
        return DummyResponse({}, 200)

    install_httpx_stub(monkeypatch, on_request=on_request)

    import haxen_sdk.client as client_mod

    monkeypatch.setattr(
        client_mod.requests, "post", lambda *args, **kwargs: DummyResponse({}, 200)
    )

    client = HaxenClient(base_url="http://example.com")
    heartbeat = HeartbeatData(status=AgentStatus.READY, mcp_servers=[], timestamp="now")

    assert await client.send_enhanced_heartbeat("node", heartbeat) is True
    assert calls and calls[0][1].endswith("/nodes/node/heartbeat")
    assert await client.notify_graceful_shutdown("node") is True


def test_sync_heartbeat(monkeypatch):
    urls = []

    class DummyResp:
        def raise_for_status(self):
            return None

    def fake_post(url, json=None, headers=None, timeout=None):
        urls.append(url)
        return DummyResp()

    import haxen_sdk.client as client_mod

    monkeypatch.setattr(client_mod.requests, "post", fake_post)

    client = HaxenClient(base_url="http://example.com")
    heartbeat = HeartbeatData(status=AgentStatus.READY, mcp_servers=[], timestamp="now")

    assert client.send_enhanced_heartbeat_sync("node", heartbeat) is True
    assert client.notify_graceful_shutdown_sync("node") is True
    assert urls[0].endswith("/nodes/node/heartbeat")
    assert urls[1].endswith("/nodes/node/shutdown")


def test_register_node_and_health(monkeypatch):
    class DummyResp:
        def __init__(self, payload):
            self._payload = payload
            self.headers = {}

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    calls = {}

    def fake_post(url, json=None, **kwargs):
        calls.setdefault("post", []).append((url, json))
        return DummyResp({"ok": True})

    def fake_put(url, json=None, **kwargs):
        calls.setdefault("put", []).append((url, json))
        return DummyResp({"status": "updated"})

    def fake_get(url, **kwargs):
        calls.setdefault("get", []).append(url)
        return DummyResp({"nodes": ["n1"]})

    import haxen_sdk.client as client_mod

    monkeypatch.setattr(client_mod.requests, "post", fake_post)
    monkeypatch.setattr(client_mod.requests, "put", fake_put)
    monkeypatch.setattr(client_mod.requests, "get", fake_get)

    client = HaxenClient(base_url="http://example.com")
    assert client.register_node({"id": "n1"}) == {"ok": True}
    assert client.update_health("n1", {"status": "up"}) == {"status": "updated"}
    assert client.get_nodes() == {"nodes": ["n1"]}

    assert calls["post"][0][0].endswith("/nodes/register")
    assert calls["put"][0][0].endswith("/nodes/n1/health")
    assert calls["get"][0].endswith("/nodes")


@pytest.mark.asyncio
async def test_register_agent(monkeypatch):
    posted = []

    def on_request(method, url, **kwargs):
        posted.append((method, url, kwargs.get("json")))
        return DummyResponse({}, 200)

    install_httpx_stub(monkeypatch, on_request=on_request)

    client = HaxenClient(base_url="http://example.com")
    ok, payload = await client.register_agent("node-1", [], [], base_url="http://agent")
    assert ok is True
    assert payload == {}
    assert posted[0][0] == "POST"
    assert posted[0][1].endswith("/nodes/register")
