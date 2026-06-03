import json
from types import SimpleNamespace

import httpx
import pytest
import requests

from agentfield.memory import (
    GlobalMemoryClient,
    MemoryClient,
    MemoryInterface,
    ScopedMemoryClient,
)


class DummySyncResponse:
    def raise_for_status(self):
        return None


class DummyAsyncResponse:
    def __init__(self, status_code: int, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("error", request=None, response=None)


@pytest.mark.functional
@pytest.mark.asyncio
async def test_memory_round_trip(monkeypatch, dummy_headers):
    store: dict[str, dict[str, object]] = {}

    def _scope(payload):
        return payload.get("scope") or "default"

    def fake_post(url, json=None, headers=None, timeout=None):  # type: ignore[override]
        assert url.endswith("/memory/set")
        scope = _scope(json)
        store.setdefault(scope, {})[json["key"]] = json["data"]
        return DummySyncResponse()

    class AsyncClientStub:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, json=None, headers=None, timeout=None):  # type: ignore[override]
            scope = _scope(json)
            data = store.get(scope, {}).get(json["key"])
            if data is None:
                return DummyAsyncResponse(404, {})
            payload = {
                "data": json_module.dumps(data)
                if not isinstance(data, (dict, list))
                else data
            }
            return DummyAsyncResponse(200, payload)

        async def request(
            self, method, url, json=None, headers=None, params=None, timeout=None
        ):  # type: ignore[override]
            method_upper = method.upper()
            if url.endswith("/memory/delete"):
                scope = _scope(json or {})
                store.get(scope, {}).pop((json or {}).get("key"), None)
                return DummyAsyncResponse(200, {"ok": True})
            if method_upper == "DELETE":
                scope = _scope(json or {})
                store.get(scope, {}).pop((json or {}).get("key"), None)
                return DummyAsyncResponse(200, {"ok": True})
            if method_upper == "GET":
                return await self.get(
                    url, params=params, headers=headers, timeout=timeout
                )
            # Default to POST semantics
            return await self.post(url, json=json, headers=headers, timeout=timeout)

        async def get(self, url, params=None, headers=None, timeout=None):  # type: ignore[override]
            scope = (params or {}).get("scope") or "default"
            keys = [{"key": key} for key in sorted(store.get(scope, {}))]
            return DummyAsyncResponse(200, keys)

    json_module = json

    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr(httpx, "AsyncClient", lambda *args, **kwargs: AsyncClientStub())
    monkeypatch.setattr("agentfield.logger.log_debug", lambda *args, **kwargs: None)

    context = SimpleNamespace(to_headers=lambda: dict(dummy_headers))
    agentfield_client = SimpleNamespace(api_base="http://agentfield.local/api/v1")
    memory_client = MemoryClient(agentfield_client, context)
    interface = MemoryInterface(memory_client, SimpleNamespace())  # type: ignore[arg-type]

    # Default scope round-trip
    await interface.set("user.profile", {"name": "Ada"})
    profile = await interface.get("user.profile")
    assert profile == {"name": "Ada"}

    # Scoped client behaviour
    scoped = ScopedMemoryClient(memory_client, scope="session", scope_id="abc")
    await scoped.set("flag", True)
    assert await scoped.get("flag") is True
    assert await scoped.list_keys() == ["flag"]
    await scoped.delete("flag")
    assert await scoped.list_keys() == []

    # Global scope helper
    global_client = GlobalMemoryClient(memory_client)
    await global_client.set("counter", 3)
    assert await global_client.get("counter") == 3


@pytest.mark.functional
@pytest.mark.asyncio
async def test_memory_client_uses_agentfield_async_request(dummy_headers):
    calls: list[tuple[str, str, dict]] = []

    class DummyAgentFieldClient:
        api_base = "http://agentfield.local/api/v1"

        async def _async_request(self, method, url, **kwargs):
            calls.append((method, url, kwargs))
            if url.endswith("/memory/get"):
                return DummyAsyncResponse(200, {"data": json.dumps({"value": 42})})
            return DummyAsyncResponse(200, {"ok": True})

    context = SimpleNamespace(to_headers=lambda: dict(dummy_headers))
    memory_client = MemoryClient(DummyAgentFieldClient(), context)

    await memory_client.set("answer", 42)
    value = await memory_client.get("answer")
    await memory_client.delete("answer")

    assert value == {"value": 42}
    methods = [call[0] for call in calls]
    assert methods == ["POST", "POST", "POST"]
    assert calls[0][1].endswith("/memory/set")
    assert calls[1][1].endswith("/memory/get")
    assert calls[2][1].endswith("/memory/delete")
