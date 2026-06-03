from __future__ import annotations

import sys
import types
import builtins
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agentfield.memory import GlobalMemoryClient, MemoryClient, ScopedMemoryClient
from agentfield.exceptions import MemoryAccessError


class DummyResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"http {self.status_code}")

    def json(self):
        return self._payload


@pytest.fixture
def execution_context():
    return SimpleNamespace(to_headers=lambda: {"X-Workflow-ID": "wf-1"})


@pytest.fixture
def memory_client(execution_context):
    return MemoryClient(
        agentfield_client=SimpleNamespace(api_base="http://agentfield.test/api/v1"),
        execution_context=execution_context,
        agent_node_id="node-9",
    )


@pytest.mark.asyncio
async def test_async_request_uses_httpx_then_requests_fallback(monkeypatch, execution_context):
    captured = {}

    class FakeAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def request(self, method, url, **kwargs):
            captured["httpx"] = (method, url, kwargs)
            return "httpx-response"

    httpx_module = types.SimpleNamespace(AsyncClient=lambda: FakeAsyncClient())
    monkeypatch.setitem(sys.modules, "httpx", httpx_module)

    client = MemoryClient(
        agentfield_client=SimpleNamespace(api_base="http://agentfield.test/api/v1"),
        execution_context=execution_context,
    )

    result = await client._async_request("GET", "http://example.test/a", timeout=1.0)
    assert result == "httpx-response"
    assert captured["httpx"][0] == "GET"

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "httpx":
            raise ImportError("httpx unavailable")
        if name == "requests":
            return requests_module
        return real_import(name, *args, **kwargs)

    async def fake_to_thread(func, method, url, **kwargs):
        captured["requests"] = (func, method, url, kwargs)
        return "requests-response"

    requests_module = types.SimpleNamespace(request=object())
    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr("agentfield.memory._to_thread", fake_to_thread)

    result = await client._async_request("POST", "http://example.test/b", json={"ok": True})
    assert result == "requests-response"
    assert captured["requests"][1] == "POST"


@pytest.mark.asyncio
async def test_set_vector_delete_vector_list_keys_and_similarity(memory_client, monkeypatch):
    calls = []

    async def fake_request(method, url, **kwargs):
        calls.append((method, url, kwargs))
        if url.endswith("/memory/list"):
            return DummyResponse(payload=[{"key": "alpha"}, {"key": "beta"}, {"ignore": True}])
        if url.endswith("/memory/vector/search"):
            return DummyResponse(payload=[{"key": "alpha", "score": 0.9}])
        return DummyResponse()

    monkeypatch.setattr(memory_client, "_async_request", fake_request)

    await memory_client.set_vector(
        "emb-1",
        (1, "2.5"),
        metadata={"kind": "demo"},
        scope="session",
        scope_id="sess-1",
    )
    await memory_client.delete_vector("emb-1", scope="session", scope_id="sess-1")
    keys = await memory_client.list_keys("session", scope_id="sess-1")
    results = await memory_client.similarity_search(
        [0, "3"],
        top_k=2,
        scope="session",
        scope_id="sess-1",
        filters={"kind": "demo"},
    )

    assert calls[0][2]["json"]["embedding"] == [1.0, 2.5]
    assert calls[0][2]["headers"]["X-Session-ID"] == "sess-1"
    assert calls[1][2]["json"] == {"key": "emb-1", "scope": "session"}
    assert keys == ["alpha", "beta"]
    assert results == [{"key": "alpha", "score": 0.9}]
    assert calls[3][2]["json"]["query_embedding"] == [0.0, 3.0]


@pytest.mark.asyncio
async def test_memory_operations_wrap_errors(memory_client, monkeypatch):
    monkeypatch.setattr(memory_client, "_async_request", AsyncMock(side_effect=RuntimeError("boom")))

    with pytest.raises(MemoryAccessError, match="Failed to delete memory key 'alpha'"):
        await memory_client.delete("alpha")

    with pytest.raises(MemoryAccessError, match="Failed to set vector key 'vec'"):
        await memory_client.set_vector("vec", [1])

    with pytest.raises(MemoryAccessError, match="Failed to delete vector key 'vec'"):
        await memory_client.delete_vector("vec")

    with pytest.raises(MemoryAccessError, match="Failed to list keys for scope 'global'"):
        await memory_client.list_keys("global")

    with pytest.raises(MemoryAccessError, match="Failed to perform similarity search"):
        await memory_client.similarity_search([1])


@pytest.mark.asyncio
async def test_exists_and_scoped_clients_delegate(memory_client, monkeypatch):
    monkeypatch.setattr(memory_client, "get", AsyncMock(side_effect=[{"ok": True}, RuntimeError("missing")]))
    monkeypatch.setattr(memory_client, "set", AsyncMock())
    monkeypatch.setattr(memory_client, "delete", AsyncMock())
    monkeypatch.setattr(memory_client, "list_keys", AsyncMock(return_value=["k1"]))
    monkeypatch.setattr(memory_client, "set_vector", AsyncMock())
    monkeypatch.setattr(memory_client, "delete_vector", AsyncMock())
    monkeypatch.setattr(memory_client, "similarity_search", AsyncMock(return_value=[{"id": 1}]))

    assert await memory_client.exists("present") is True
    assert await memory_client.exists("missing") is False

    scoped = ScopedMemoryClient(memory_client, "actor", "actor-1", event_client=None)
    global_client = GlobalMemoryClient(memory_client, event_client=None)

    await scoped.set("name", {"value": 1})
    await scoped.delete("name")
    assert await scoped.list_keys() == ["k1"]
    await scoped.set_vector("vec", [1, 2])
    await scoped.delete_vector("vec")
    assert await scoped.similarity_search([1, 2]) == [{"id": 1}]
    assert await global_client.list_keys() == ["k1"]

    memory_client.set.assert_awaited_once_with("name", {"value": 1}, scope="actor", scope_id="actor-1")
    memory_client.delete.assert_awaited_once_with("name", scope="actor", scope_id="actor-1")
    memory_client.list_keys.assert_any_await("actor", scope_id="actor-1")
    memory_client.list_keys.assert_any_await("global")


def test_scoped_on_change_decorator_without_events(memory_client):
    scoped = ScopedMemoryClient(memory_client, "session", "sess-1", event_client=None)

    @scoped.on_change("prefs.*")
    async def listener(event):
        return event

    assert listener.__name__ == "listener"
