import asyncio
import sys
import types

from agentfield.client import AgentFieldClient
from agentfield.types import AgentStatus, HeartbeatData


class DummyResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {}
        self.content = b"{}"
        self.text = "{}"

    def raise_for_status(self):
        if not (200 <= self.status_code < 400):
            raise RuntimeError("bad status")

    def json(self):
        return self._payload


def test_send_enhanced_heartbeat_sync_success_and_failure(monkeypatch):
    sent = {"calls": 0}

    def ok_post(url, json, headers, timeout):
        sent["calls"] += 1
        return DummyResponse(200)

    import agentfield.client as client_mod

    monkeypatch.setattr(client_mod.requests, "post", ok_post)

    bc = AgentFieldClient(base_url="http://example")
    hb = HeartbeatData(status=AgentStatus.READY, mcp_servers=[], timestamp="now")
    assert bc.send_enhanced_heartbeat_sync("node1", hb) is True

    def bad_post(url, json, headers, timeout):
        raise RuntimeError("boom")

    monkeypatch.setattr(client_mod.requests, "post", bad_post)
    assert bc.send_enhanced_heartbeat_sync("node1", hb) is False


def test_notify_graceful_shutdown_sync(monkeypatch):
    import agentfield.client as client_mod

    def ok_post(url, headers, timeout):
        return DummyResponse(200)

    monkeypatch.setattr(client_mod.requests, "post", ok_post)
    bc = AgentFieldClient(base_url="http://example")
    assert bc.notify_graceful_shutdown_sync("node1") is True

    def bad_post(url, headers, timeout):
        raise RuntimeError("x")

    monkeypatch.setattr(client_mod.requests, "post", bad_post)
    assert bc.notify_graceful_shutdown_sync("node1") is False


def test_register_agent_with_status_async(monkeypatch):
    # Provide a dummy httpx module that AgentFieldClient will use
    from agentfield import client as client_mod

    class DummyAsyncClient:
        def __init__(self, *args, **kwargs):
            self.is_closed = False

        async def request(self, method, url, **kwargs):
            return DummyResponse(status_code=201, payload={})

        async def aclose(self):
            self.is_closed = True

    stub_httpx = types.SimpleNamespace(
        AsyncClient=DummyAsyncClient,
        Limits=lambda *a, **k: None,
        Timeout=lambda *a, **k: None,
        HTTPStatusError=Exception,
    )

    monkeypatch.setitem(sys.modules, "httpx", stub_httpx)
    client_mod.httpx = stub_httpx
    monkeypatch.setattr(
        client_mod,
        "_ensure_httpx",
        lambda force_reload=False: stub_httpx,
        raising=False,
    )

    bc = AgentFieldClient(base_url="http://example")

    async def run():
        return await bc.register_agent_with_status(
            node_id="n1", reasoners=[], skills=[], base_url="http://agent"
        )

    success, payload = asyncio.run(run())
    assert success is True
    assert payload == {}
