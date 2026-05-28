from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest
import requests

from agentfield.memory import (
    GlobalMemoryClient,
    MemoryClient,
    MemoryInterface,
    ScopedMemoryClient,
)


class DummyResponse:
    def __init__(self, status_code: int = 200, payload: dict | str | None = None):
        self.status_code = status_code
        self._payload = payload or {}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("error", request=None, response=None)


@pytest.fixture(autouse=True)
def mute_debug_logs(monkeypatch):
    monkeypatch.setattr("agentfield.logger.log_debug", lambda *args, **kwargs: None)


@pytest.fixture
def memory_client(dummy_headers):
    context = SimpleNamespace(to_headers=lambda: dict(dummy_headers))
    agentfield_client = SimpleNamespace(api_base="http://agentfield.local/api/v1")
    return MemoryClient(agentfield_client, context)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_set_posts_payload(memory_client, monkeypatch, dummy_headers):
    captured: dict[str, object] = {}

    def fake_post(url, json=None, headers=None, timeout=None):  # type: ignore[override]
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        return DummyResponse()

    monkeypatch.setattr(requests, "post", fake_post)

    await memory_client.set("user.profile", {"name": "Ada"})

    assert captured["url"].endswith("/memory/set")
    payload = captured["json"]  # type: ignore[assignment]
    assert payload["key"] == "user.profile"
    assert payload["data"] == {"name": "Ada"}
    assert captured["headers"] == dummy_headers


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_decodes_json_payload(memory_client, monkeypatch):
    record = {}

    class DummyAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def request(self, method, url, json=None, headers=None, timeout=None):  # type: ignore[override]
            record["url"] = url
            record["json"] = json
            record["headers"] = headers
            return DummyResponse(200, {"data": json_module.dumps({"count": 1})})

    json_module = json
    monkeypatch.setattr(
        httpx, "AsyncClient", lambda *args, **kwargs: DummyAsyncClient()
    )

    result = await memory_client.get("stats")

    assert result == {"count": 1}
    assert record["url"].endswith("/memory/get")
    assert record["json"]["key"] == "stats"  # type: ignore[index]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_returns_default_on_404(memory_client, monkeypatch):
    class DummyAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def request(self, method, url, json=None, headers=None, timeout=None):  # type: ignore[override]
            return DummyResponse(404, {})

    monkeypatch.setattr(
        httpx, "AsyncClient", lambda *args, **kwargs: DummyAsyncClient()
    )

    result = await memory_client.get("missing", default="fallback")
    assert result == "fallback"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_delete_uses_post_endpoint(memory_client, monkeypatch):
    captured = {}

    class DummyAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def request(self, method, url, json=None, headers=None, timeout=None):  # type: ignore[override]
            captured["method"] = method
            captured["url"] = url
            captured["json"] = json
            return DummyResponse()

    monkeypatch.setattr(
        httpx, "AsyncClient", lambda *args, **kwargs: DummyAsyncClient()
    )

    await memory_client.delete("temp")

    assert captured["method"] == "POST"
    assert captured["url"].endswith("/memory/delete")
    assert captured["json"]["key"] == "temp"  # type: ignore[index]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_set_vector_calls_vector_endpoint(memory_client, monkeypatch):
    captured: dict[str, Any] = {}

    class DummyResponse:
        def raise_for_status(self):
            return None

    async def fake_request(method, url, json=None, headers=None, timeout=None):  # type: ignore[override]
        captured["method"] = method
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return DummyResponse()

    monkeypatch.setattr(memory_client, "_async_request", fake_request)

    await memory_client.set_vector("chunk_1", [0.1, 0.2], metadata={"source": "doc"})

    assert captured["url"].endswith("/memory/vector/set")
    assert captured["json"]["key"] == "chunk_1"  # type: ignore[index]
    assert captured["json"]["metadata"] == {"source": "doc"}  # type: ignore[index]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_delete_vector_calls_vector_endpoint(memory_client, monkeypatch):
    captured: dict[str, Any] = {}

    class DummyResponse:
        def raise_for_status(self):
            return None

    async def fake_request(method, url, json=None, headers=None, timeout=None):  # type: ignore[override]
        captured["url"] = url
        captured["json"] = json
        return DummyResponse()

    monkeypatch.setattr(memory_client, "_async_request", fake_request)

    await memory_client.delete_vector("chunk_1")

    assert captured["url"].endswith("/memory/vector/delete")
    assert captured["json"]["key"] == "chunk_1"  # type: ignore[index]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_similarity_search_returns_results(memory_client, monkeypatch):
    expected = [{"key": "chunk_1", "score": 0.9}]

    class DummyResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    async def fake_request(method, url, json=None, headers=None, timeout=None):  # type: ignore[override]
        assert json["query_embedding"] == [0.5, 0.2]  # type: ignore[index]
        assert json["top_k"] == 5  # type: ignore[index]
        return DummyResponse(expected)

    monkeypatch.setattr(memory_client, "_async_request", fake_request)

    result = await memory_client.similarity_search([0.5, 0.2], top_k=5)
    assert result == expected


@pytest.mark.unit
@pytest.mark.asyncio
async def test_list_keys_returns_names(memory_client, monkeypatch):
    class DummyAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def request(self, method, url, params=None, headers=None, timeout=None):  # type: ignore[override]
            assert params == {"scope": "session"}
            keys = [{"key": "a"}, {"key": "b"}]
            return DummyResponse(200, keys)

    monkeypatch.setattr(
        httpx, "AsyncClient", lambda *args, **kwargs: DummyAsyncClient()
    )

    result = await memory_client.list_keys("session")
    assert result == ["a", "b"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_exists_handles_errors(memory_client, monkeypatch):
    async def _raise(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(memory_client, "get", _raise)

    assert await memory_client.exists("missing") is False

    async def _value(*args, **kwargs):
        return "ok"

    monkeypatch.setattr(memory_client, "get", _value)

    assert await memory_client.exists("present") is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_scoped_client_delegates(memory_client):
    base = memory_client
    base.set = AsyncMock()  # type: ignore[assignment]
    base.get = AsyncMock(return_value=True)  # type: ignore[assignment]
    base.exists = AsyncMock(return_value=True)  # type: ignore[assignment]
    base.delete = AsyncMock()  # type: ignore[assignment]
    base.list_keys = AsyncMock(return_value=["a"])  # type: ignore[assignment]
    base.set_vector = AsyncMock()  # type: ignore[assignment]
    base.delete_vector = AsyncMock()  # type: ignore[assignment]
    base.similarity_search = AsyncMock(return_value=[{"key": "chunk"}])  # type: ignore[assignment]

    scoped = ScopedMemoryClient(base, scope="session", scope_id="abc")

    await scoped.set("key", 1)
    await scoped.get("key")
    await scoped.exists("key")
    await scoped.delete("key")
    await scoped.list_keys()
    await scoped.set_vector("chunk", [0.1])
    await scoped.delete_vector("chunk")
    await scoped.similarity_search([0.2])

    base.set.assert_awaited_once_with(
        "key", 1, scope="session", scope_id="abc"
    )
    base.get.assert_awaited_once_with(
        "key", default=None, scope="session", scope_id="abc"
    )
    base.exists.assert_awaited_once_with("key", scope="session", scope_id="abc")
    base.delete.assert_awaited_once_with("key", scope="session", scope_id="abc")
    base.list_keys.assert_awaited_once_with("session", scope_id="abc")
    base.set_vector.assert_awaited_once_with(
        "chunk", [0.1], metadata=None, scope="session", scope_id="abc"
    )
    base.delete_vector.assert_awaited_once_with(
        "chunk", scope="session", scope_id="abc"
    )
    base.similarity_search.assert_awaited_once_with(
        [0.2],
        top_k=10,
        scope="session",
        scope_id="abc",
        filters=None,
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_global_client_delegates(memory_client):
    base = memory_client
    base.set = AsyncMock()  # type: ignore[assignment]
    base.get = AsyncMock(return_value=True)  # type: ignore[assignment]
    base.exists = AsyncMock(return_value=True)  # type: ignore[assignment]
    base.delete = AsyncMock()  # type: ignore[assignment]
    base.list_keys = AsyncMock(return_value=["g"])  # type: ignore[assignment]
    base.set_vector = AsyncMock()  # type: ignore[assignment]
    base.delete_vector = AsyncMock()  # type: ignore[assignment]
    base.similarity_search = AsyncMock(return_value=[{"key": "chunk"}])  # type: ignore[assignment]

    global_client = GlobalMemoryClient(base)

    await global_client.set("key", 1)
    await global_client.get("key")
    await global_client.exists("key")
    await global_client.delete("key")
    await global_client.list_keys()
    await global_client.set_vector("chunk", [0.2])
    await global_client.delete_vector("chunk")
    await global_client.similarity_search([0.3])

    base.set.assert_awaited_once_with("key", 1, scope="global")
    base.get.assert_awaited_once_with("key", default=None, scope="global")
    base.exists.assert_awaited_once_with("key", scope="global")
    base.delete.assert_awaited_once_with("key", scope="global")
    base.list_keys.assert_awaited_once_with("global")
    base.set_vector.assert_awaited_once_with(
        "chunk", [0.2], metadata=None, scope="global"
    )
    base.delete_vector.assert_awaited_once_with("chunk", scope="global")
    base.similarity_search.assert_awaited_once_with(
        [0.3], top_k=10, scope="global", filters=None
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_memory_interface_uses_underlying_client(memory_client):
    base = memory_client
    base.set = AsyncMock()  # type: ignore[assignment]
    base.get = AsyncMock(return_value={})  # type: ignore[assignment]
    base.exists = AsyncMock(return_value=False)  # type: ignore[assignment]
    base.delete = AsyncMock()  # type: ignore[assignment]
    base.set_vector = AsyncMock()  # type: ignore[assignment]
    base.delete_vector = AsyncMock()  # type: ignore[assignment]
    base.similarity_search = AsyncMock(return_value=[])  # type: ignore[assignment]

    events = SimpleNamespace()
    interface = MemoryInterface(base, events)  # type: ignore[arg-type]

    await interface.set("key", 1)
    await interface.get("key")
    await interface.exists("key")
    await interface.delete("key")
    await interface.set_vector("chunk", [0.4])
    await interface.delete_vector("chunk")
    await interface.similarity_search([0.4])

    base.set.assert_awaited_once_with("key", 1)
    base.get.assert_awaited_once_with("key", default=None)
    base.exists.assert_awaited_once_with("key")
    base.delete.assert_awaited_once_with("key")
    base.set_vector.assert_awaited_once_with("chunk", [0.4], metadata=None)
    base.delete_vector.assert_awaited_once_with("chunk")
    base.similarity_search.assert_awaited_once_with([0.4], top_k=10, filters=None)
    assert interface.events is events


@pytest.mark.unit
@pytest.mark.asyncio
async def test_set_uses_async_request_when_available(dummy_headers):
    captured: dict[str, object] = {}

    async def fake_async_request(method, url, **kwargs):
        captured["method"] = method
        captured["url"] = url
        captured["kwargs"] = kwargs

        class OkResponse:
            def raise_for_status(self):
                return None

        return OkResponse()

    context = SimpleNamespace(to_headers=lambda: dict(dummy_headers))
    agentfield_client = SimpleNamespace(
        api_base="http://agentfield.local/api/v1",
        _async_request=fake_async_request,
    )
    client = MemoryClient(agentfield_client, context)

    await client.set("key", {"value": 1})

    assert captured["method"] == "POST"
    assert captured["url"].endswith("/memory/set")
    payload = captured["kwargs"]["json"]  # type: ignore[index]
    assert payload == {"key": "key", "data": {"value": 1}}
    assert captured["kwargs"]["headers"] == dummy_headers  # type: ignore[index]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_set_raises_on_unserializable_payload(memory_client):
    with pytest.raises(TypeError):
        await memory_client.set("key", {"bad": {1, 2, 3}})


@pytest.mark.unit
@pytest.mark.asyncio
async def test_async_request_falls_back_to_requests(memory_client, monkeypatch):
    import builtins

    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "httpx":
            raise ImportError("missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    calls: dict[str, object] = {}

    def fake_request(method, url, **kwargs):  # type: ignore[override]
        calls["method"] = method
        calls["url"] = url
        calls["kwargs"] = kwargs
        return DummyResponse()

    monkeypatch.setattr("requests.request", fake_request)

    response = await memory_client._async_request(
        "GET", "http://example.com", params={"a": 1}
    )

    assert isinstance(response, DummyResponse)
    assert calls["method"] == "GET"
    assert calls["url"] == "http://example.com"
    assert calls["kwargs"]["params"] == {"a": 1}  # type: ignore[index]


@pytest.mark.unit
def test_scoped_on_change_without_events(memory_client):
    scoped = ScopedMemoryClient(
        memory_client, scope="session", scope_id="abc", event_client=None
    )

    @scoped.on_change("*")
    def handler():
        return "ok"

    assert handler() == "ok"
