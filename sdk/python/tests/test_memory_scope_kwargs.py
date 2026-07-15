"""Unit tests for the convenience ``scope``/``scope_id`` kwargs on MemoryInterface.

These cover the developer-facing ``app.memory.set/get/delete/similarity_search``
convenience layer added to mirror the TypeScript SDK. The kwargs delegate to the same scoped clients
the accessor API (``global_scope``/``session()``/``actor()``/``workflow()``) uses,
so the tests assert the delegation reaches the underlying MemoryClient with the
right scope/scope_id.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agentfield.memory import MemoryClient, MemoryInterface


@pytest.fixture
def memory_client(dummy_headers):
    context = SimpleNamespace(to_headers=lambda: dict(dummy_headers))
    agentfield_client = SimpleNamespace(api_base="http://agentfield.local/api/v1")
    return MemoryClient(agentfield_client, context)


@pytest.fixture
def interface(memory_client):
    """MemoryInterface with a stubbed low-level client and no event client."""
    memory_client.set = AsyncMock()  # type: ignore[assignment]
    memory_client.get = AsyncMock(return_value="stored")  # type: ignore[assignment]
    memory_client.delete = AsyncMock()  # type: ignore[assignment]
    memory_client.similarity_search = AsyncMock(  # type: ignore[assignment]
        return_value=[{"key": "match", "score": 0.9}]
    )
    return MemoryInterface(memory_client, None)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Default behavior unchanged (scope=None)
# --------------------------------------------------------------------------- #


@pytest.mark.unit
@pytest.mark.asyncio
async def test_set_default_scope_unchanged(interface):
    await interface.set("key", {"v": 1})
    interface.memory_client.set.assert_awaited_once_with("key", {"v": 1})


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_default_scope_unchanged(interface):
    result = await interface.get("key", default="fallback")
    interface.memory_client.get.assert_awaited_once_with("key", default="fallback")
    assert result == "stored"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_delete_default_scope_unchanged(interface):
    await interface.delete("key")
    interface.memory_client.delete.assert_awaited_once_with("key")


# --------------------------------------------------------------------------- #
# Explicit scope kwarg delegation — set
# --------------------------------------------------------------------------- #


@pytest.mark.unit
@pytest.mark.asyncio
async def test_set_scope_global(interface):
    await interface.set("config", {"v": 1}, scope="global")
    interface.memory_client.set.assert_awaited_once_with(
        "config", {"v": 1}, scope="global"
    )


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("scope", ["session", "actor", "workflow"])
async def test_set_scope_with_id(interface, scope):
    await interface.set("ctx", {"v": 1}, scope=scope, scope_id="id-123")
    interface.memory_client.set.assert_awaited_once_with(
        "ctx", {"v": 1}, scope=scope, scope_id="id-123"
    )


# --------------------------------------------------------------------------- #
# Explicit scope kwarg delegation — get
# --------------------------------------------------------------------------- #


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_scope_global(interface):
    result = await interface.get("config", scope="global")
    interface.memory_client.get.assert_awaited_once_with(
        "config", default=None, scope="global"
    )
    assert result == "stored"


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("scope", ["session", "actor", "workflow"])
async def test_get_scope_with_id(interface, scope):
    await interface.get("ctx", default="d", scope=scope, scope_id="id-123")
    interface.memory_client.get.assert_awaited_once_with(
        "ctx", default="d", scope=scope, scope_id="id-123"
    )


# --------------------------------------------------------------------------- #
# Explicit scope kwarg delegation — delete
# --------------------------------------------------------------------------- #


@pytest.mark.unit
@pytest.mark.asyncio
async def test_delete_scope_global(interface):
    await interface.delete("config", scope="global")
    interface.memory_client.delete.assert_awaited_once_with("config", scope="global")


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("scope", ["session", "actor", "workflow"])
async def test_delete_scope_with_id(interface, scope):
    await interface.delete("ctx", scope=scope, scope_id="id-123")
    interface.memory_client.delete.assert_awaited_once_with(
        "ctx", scope=scope, scope_id="id-123"
    )


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


@pytest.mark.unit
@pytest.mark.asyncio
async def test_invalid_scope_raises_value_error(interface):
    with pytest.raises(ValueError) as exc_info:
        await interface.set("k", 1, scope="agent")

    message = str(exc_info.value)
    assert "agent" in message
    # Error lists the valid scopes so the caller can self-correct.
    for valid in ("global", "session", "actor", "workflow"):
        assert valid in message
    interface.memory_client.set.assert_not_awaited()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_invalid_scope_raises_on_get(interface):
    with pytest.raises(ValueError):
        await interface.get("k", scope="run")
    interface.memory_client.get.assert_not_awaited()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_invalid_scope_raises_on_delete(interface):
    with pytest.raises(ValueError):
        await interface.delete("k", scope="bogus")
    interface.memory_client.delete.assert_not_awaited()


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("scope", ["session", "actor", "workflow"])
async def test_set_scope_without_id_uses_current_context(interface, scope):
    await interface.set("k", 1, scope=scope)
    interface.memory_client.set.assert_awaited_once_with(
        "k", 1, scope=scope, scope_id=None
    )


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("scope", ["session", "actor", "workflow"])
async def test_get_scope_without_id_uses_current_context(interface, scope):
    await interface.get("k", scope=scope)
    interface.memory_client.get.assert_awaited_once_with(
        "k", default=None, scope=scope, scope_id=None
    )


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("scope", ["session", "actor", "workflow"])
async def test_delete_scope_without_id_uses_current_context(interface, scope):
    await interface.delete("k", scope=scope)
    interface.memory_client.delete.assert_awaited_once_with(
        "k", scope=scope, scope_id=None
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_global_scope_ignores_scope_id(interface):
    # scope_id is accepted but irrelevant for global; it must not be forwarded.
    await interface.set("config", {"v": 1}, scope="global", scope_id="ignored")
    interface.memory_client.set.assert_awaited_once_with(
        "config", {"v": 1}, scope="global"
    )


# --------------------------------------------------------------------------- #
# Explicit scope kwarg delegation — similarity_search
# --------------------------------------------------------------------------- #


@pytest.mark.unit
@pytest.mark.asyncio
async def test_similarity_search_default_scope_unchanged(interface):
    result = await interface.similarity_search([0.1, 0.2], top_k=5)
    interface.memory_client.similarity_search.assert_awaited_once_with(
        [0.1, 0.2], top_k=5, filters=None
    )
    assert result == [{"key": "match", "score": 0.9}]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_similarity_search_scope_global(interface):
    result = await interface.similarity_search([0.1, 0.2], scope="global")
    interface.memory_client.similarity_search.assert_awaited_once_with(
        [0.1, 0.2], top_k=10, scope="global", filters=None
    )
    assert result == [{"key": "match", "score": 0.9}]


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("scope", ["session", "actor", "workflow"])
async def test_similarity_search_scope_with_id(interface, scope):
    await interface.similarity_search(
        [0.3],
        top_k=3,
        filters={"kind": "doc"},
        scope=scope,
        scope_id="id-123",
    )
    interface.memory_client.similarity_search.assert_awaited_once_with(
        [0.3], top_k=3, scope=scope, scope_id="id-123", filters={"kind": "doc"}
    )


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("scope", ["session", "actor", "workflow"])
async def test_similarity_search_scope_without_id_uses_current_context(
    interface, scope
):
    await interface.similarity_search([0.3], scope=scope)
    interface.memory_client.similarity_search.assert_awaited_once_with(
        [0.3], top_k=10, scope=scope, scope_id=None, filters=None
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_invalid_scope_raises_on_similarity_search(interface):
    with pytest.raises(ValueError):
        await interface.similarity_search([0.1], scope="run")
    interface.memory_client.similarity_search.assert_not_awaited()
