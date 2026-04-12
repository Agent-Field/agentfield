"""
Behavioral invariant tests for the MemoryClient and related memory classes.

These tests verify structural properties of the memory system that must always
hold regardless of implementation changes. They use mocks to isolate the
client from HTTP and verify behavioral contracts at the API boundary.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_memory_client(api_base: str = "http://localhost:8080/api/v1"):
    """Build a MemoryClient with a fully mocked AgentFieldClient and ExecutionContext."""
    from agentfield.memory import MemoryClient
    from agentfield.execution_context import ExecutionContext

    mock_af_client = MagicMock()
    mock_af_client.api_base = api_base

    # Provide an async _async_request on the mock
    mock_af_client._async_request = AsyncMock()

    exec_ctx = ExecutionContext(
        run_id="run-test",
        execution_id="exec-test",
        agent_instance=None,
        reasoner_name="test-reasoner",
        workflow_id="wf-test",
        session_id="sess-test",
        actor_id="actor-test",
    )

    return MemoryClient(mock_af_client, exec_ctx, agent_node_id="test-node")


def _ok_response(data: Any = None, status_code: int = 200):
    """Build a mock HTTP response that looks like a successful memory response."""
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.json.return_value = {"data": data} if data is not None else {}
    mock_resp.raise_for_status = MagicMock()
    return mock_resp


def _not_found_response():
    mock_resp = MagicMock()
    mock_resp.status_code = 404
    mock_resp.json.return_value = {"error": "not_found"}
    mock_resp.raise_for_status = MagicMock()
    return mock_resp


# ---------------------------------------------------------------------------
# 1. Scope hierarchy order
# ---------------------------------------------------------------------------

class TestScopeHierarchyOrder:
    """
    The hierarchical lookup MUST check workflow → session → actor → global.
    The scope is communicated via headers, so we verify that when a context
    has all IDs populated, the payload sent uses the headers that encode the
    most-specific scope (letting the server resolve hierarchy).
    """

    @pytest.mark.asyncio
    async def test_invariant_get_passes_all_context_headers_for_hierarchy(self):
        """
        When ExecutionContext has workflow, session, and actor IDs populated,
        a plain get() must include ALL those IDs in the headers so the server
        can resolve workflow → session → actor → global ordering.
        """
        client = _make_memory_client()

        client.agentfield_client._async_request.return_value = _ok_response("value")

        await client.get("my-key")

        assert client.agentfield_client._async_request.called, (
            "INVARIANT VIOLATION: _async_request was never called for get()"
        )
        call_kwargs = client.agentfield_client._async_request.call_args
        headers = call_kwargs.kwargs.get("headers", {})

        # All context IDs must be present so the server can do hierarchy resolution.
        # The header may be X-Workflow-ID or X-Run-ID depending on implementation.
        has_workflow_context = (
            "X-Workflow-ID" in headers
            or "X-Run-ID" in headers
            or "X-Workflow-Run-ID" in headers
        )
        assert has_workflow_context, (
            f"INVARIANT VIOLATION: No workflow context header found in get() headers. "
            f"Server cannot perform workflow → session → actor → global resolution. "
            f"Headers present: {set(headers.keys())}"
        )
        assert "X-Session-ID" in headers, (
            f"INVARIANT VIOLATION: X-Session-ID missing from get() headers. "
            f"Headers present: {set(headers.keys())}"
        )

    @pytest.mark.asyncio
    async def test_invariant_scope_override_sends_scope_in_payload(self):
        """
        When explicit scope is provided to get(), it must appear in the request
        payload, allowing the server to restrict the search to that scope.
        """
        client = _make_memory_client()
        client.agentfield_client._async_request.return_value = _ok_response("v")

        await client.get("key", scope="global")

        call_kwargs = client.agentfield_client._async_request.call_args
        payload = call_kwargs.kwargs.get("json", {})

        assert payload.get("scope") == "global", (
            f"INVARIANT VIOLATION: explicit scope 'global' not in request payload. "
            f"Got: {payload}"
        )


# ---------------------------------------------------------------------------
# 2. Scope independence
# ---------------------------------------------------------------------------

class TestScopeIndependence:
    """Setting a key in one scope must not affect the same key in another scope."""

    @pytest.mark.asyncio
    async def test_invariant_workflow_scope_set_uses_different_header_than_global(self):
        """
        set() with scope='workflow' and scope_id must include X-Workflow-ID header.
        The payload must contain scope='workflow'.
        """
        client = _make_memory_client()
        client.agentfield_client._async_request.return_value = _ok_response()
        client.agentfield_client._async_request.return_value.raise_for_status = MagicMock()

        # Call with workflow scope override
        await client.set("x", "workflow-value", scope="workflow", scope_id="wf-123")

        call_kwargs = client.agentfield_client._async_request.call_args
        headers = call_kwargs.kwargs.get("headers", {})
        payload = call_kwargs.kwargs.get("json", {})

        assert headers.get("X-Workflow-ID") == "wf-123", (
            f"INVARIANT VIOLATION: Workflow scope ID 'wf-123' not in headers. Headers: {headers}"
        )
        assert payload.get("scope") == "workflow", (
            f"INVARIANT VIOLATION: scope='workflow' not in payload. Payload: {payload}"
        )

    @pytest.mark.asyncio
    async def test_invariant_global_scope_set_does_not_inject_workflow_header(self):
        """
        GlobalMemoryClient.set() must use scope='global' in the payload.
        """
        from agentfield.memory import GlobalMemoryClient

        inner_client = _make_memory_client()
        inner_client.agentfield_client._async_request.return_value = _ok_response()
        inner_client.agentfield_client._async_request.return_value.raise_for_status = MagicMock()

        global_client = GlobalMemoryClient(inner_client)
        await global_client.set("x", "global-value")

        call_kwargs = inner_client.agentfield_client._async_request.call_args
        payload = call_kwargs.kwargs.get("json", {})

        assert payload.get("scope") == "global", (
            f"INVARIANT VIOLATION: GlobalMemoryClient.set() did not use scope='global'. "
            f"Payload: {payload}"
        )

    @pytest.mark.asyncio
    async def test_invariant_scoped_clients_send_different_scope_ids(self):
        """
        ScopedMemoryClient for workflow-A and workflow-B must send different
        X-Workflow-ID headers — they are independent namespaces.
        """
        from agentfield.memory import ScopedMemoryClient

        client = _make_memory_client()
        client.agentfield_client._async_request.return_value = _ok_response()
        client.agentfield_client._async_request.return_value.raise_for_status = MagicMock()

        wf_a = ScopedMemoryClient(client, "workflow", "wf-AAA")
        wf_b = ScopedMemoryClient(client, "workflow", "wf-BBB")

        await wf_a.set("key", "value-a")
        call_a = client.agentfield_client._async_request.call_args
        headers_a = call_a.kwargs.get("headers", {})

        await wf_b.set("key", "value-b")
        call_b = client.agentfield_client._async_request.call_args
        headers_b = call_b.kwargs.get("headers", {})

        assert headers_a.get("X-Workflow-ID") == "wf-AAA", (
            f"INVARIANT VIOLATION: workflow-A scope not in headers. Headers: {headers_a}"
        )
        assert headers_b.get("X-Workflow-ID") == "wf-BBB", (
            f"INVARIANT VIOLATION: workflow-B scope not in headers. Headers: {headers_b}"
        )
        assert headers_a.get("X-Workflow-ID") != headers_b.get("X-Workflow-ID"), (
            "INVARIANT VIOLATION: Two different workflow scopes produced the same header value."
        )


# ---------------------------------------------------------------------------
# 3. Key namespacing
# ---------------------------------------------------------------------------

class TestKeyNamespacing:
    """Two different keys in the same scope are fully independent."""

    @pytest.mark.asyncio
    async def test_invariant_different_keys_produce_different_payloads(self):
        """
        Calling set("key-A", ...) and set("key-B", ...) must produce
        different key values in the request payload.
        """
        client = _make_memory_client()
        client.agentfield_client._async_request.return_value = _ok_response()
        client.agentfield_client._async_request.return_value.raise_for_status = MagicMock()

        await client.set("key-A", "value-1")
        call_a = client.agentfield_client._async_request.call_args
        payload_a = call_a.kwargs.get("json", {})

        await client.set("key-B", "value-2")
        call_b = client.agentfield_client._async_request.call_args
        payload_b = call_b.kwargs.get("json", {})

        assert payload_a["key"] == "key-A", (
            f"INVARIANT VIOLATION: set('key-A') sent key='{payload_a['key']}'"
        )
        assert payload_b["key"] == "key-B", (
            f"INVARIANT VIOLATION: set('key-B') sent key='{payload_b['key']}'"
        )
        assert payload_a["key"] != payload_b["key"], (
            "INVARIANT VIOLATION: Two different keys produced the same request key field."
        )

    @pytest.mark.asyncio
    async def test_invariant_get_passes_exact_requested_key(self):
        """
        get("specific-key") must pass exactly that key in the payload,
        not a transformed or hashed version.
        """
        client = _make_memory_client()
        client.agentfield_client._async_request.return_value = _ok_response("some-data")

        await client.get("specific-key")

        call_kwargs = client.agentfield_client._async_request.call_args
        payload = call_kwargs.kwargs.get("json", {})

        assert payload.get("key") == "specific-key", (
            f"INVARIANT VIOLATION: get('specific-key') sent key='{payload.get('key')}' "
            "instead of 'specific-key'."
        )


# ---------------------------------------------------------------------------
# 4. Null safety
# ---------------------------------------------------------------------------

class TestNullSafety:
    """Getting a non-existent key must return None/default, never crash."""

    @pytest.mark.asyncio
    async def test_invariant_missing_key_returns_default_not_raises(self):
        """
        A 404 response from the server must result in the default value being
        returned, NOT an exception propagating to the caller.
        """
        client = _make_memory_client()
        client.agentfield_client._async_request.return_value = _not_found_response()

        result = await client.get("nonexistent-key", default="my-default")

        assert result == "my-default", (
            f"INVARIANT VIOLATION: Missing key returned '{result}' instead of default 'my-default'."
        )

    @pytest.mark.asyncio
    async def test_invariant_missing_key_returns_none_when_no_default(self):
        """
        A 404 response with no default must return None (not raise).
        """
        client = _make_memory_client()
        client.agentfield_client._async_request.return_value = _not_found_response()

        result = await client.get("nonexistent-key")

        assert result is None, (
            f"INVARIANT VIOLATION: Missing key without default returned '{result}' instead of None."
        )

    @pytest.mark.asyncio
    async def test_invariant_missing_key_does_not_raise_exception(self):
        """
        A 404 must never propagate as an uncaught exception from get().
        """
        client = _make_memory_client()
        client.agentfield_client._async_request.return_value = _not_found_response()

        try:
            await client.get("no-such-key")
            # Should reach here without raising
        except Exception as exc:
            pytest.fail(
                f"INVARIANT VIOLATION: get() raised {type(exc).__name__}: {exc} "
                "for a 404 response. It should return the default value."
            )

    @pytest.mark.asyncio
    async def test_invariant_scoped_memory_client_missing_key_returns_default(self):
        """
        ScopedMemoryClient.get() for a missing key must return the default value.
        """
        from agentfield.memory import ScopedMemoryClient

        client = _make_memory_client()
        client.agentfield_client._async_request.return_value = _not_found_response()

        scoped = ScopedMemoryClient(client, "session", "sess-xyz")
        result = await scoped.get("nonexistent", default=42)

        assert result == 42, (
            f"INVARIANT VIOLATION: ScopedMemoryClient.get() returned '{result}' "
            "instead of default 42 for a missing key."
        )
