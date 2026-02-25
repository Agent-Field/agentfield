"""Tests for the AgentField SDK domain-specific exception hierarchy."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
import requests
import responses as responses_lib

from agentfield.exceptions import (
    AgentFieldClientError,
    AgentFieldError,
    ExecutionTimeoutError,
    MemoryAccessError,
    RegistrationError,
    ValidationError,
)
from agentfield.client import AgentFieldClient


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------


class TestExceptionHierarchy:
    """Verify the inheritance chain and basic semantics."""

    def test_base_is_exception(self):
        assert issubclass(AgentFieldError, Exception)

    @pytest.mark.parametrize(
        "cls",
        [
            AgentFieldClientError,
            ExecutionTimeoutError,
            MemoryAccessError,
            RegistrationError,
            ValidationError,
        ],
    )
    def test_subclass_of_base(self, cls):
        assert issubclass(cls, AgentFieldError)

    @pytest.mark.parametrize(
        "cls",
        [
            AgentFieldClientError,
            ExecutionTimeoutError,
            MemoryAccessError,
            RegistrationError,
            ValidationError,
        ],
    )
    def test_catchable_as_base(self, cls):
        with pytest.raises(AgentFieldError):
            raise cls("test")

    def test_distinct_types(self):
        """Each exception should be independently catchable."""
        with pytest.raises(RegistrationError):
            raise RegistrationError("reg")

        with pytest.raises(ExecutionTimeoutError):
            raise ExecutionTimeoutError("timeout")

        with pytest.raises(MemoryAccessError):
            raise MemoryAccessError("mem")

        with pytest.raises(ValidationError):
            raise ValidationError("val")

        with pytest.raises(AgentFieldClientError):
            raise AgentFieldClientError("client")

    def test_exception_chaining(self):
        """Verify 'raise X from Y' preserves the cause."""
        original = ValueError("original cause")
        try:
            raise AgentFieldClientError("wrapped") from original
        except AgentFieldClientError as exc:
            assert exc.__cause__ is original

    def test_message_preserved(self):
        exc = MemoryAccessError("key 'foo' not found")
        assert str(exc) == "key 'foo' not found"


# ---------------------------------------------------------------------------
# Top-level imports
# ---------------------------------------------------------------------------


class TestTopLevelImports:
    """Exceptions should be importable from the agentfield package root."""

    def test_import_from_package(self):
        import agentfield

        assert hasattr(agentfield, "AgentFieldError")
        assert hasattr(agentfield, "AgentFieldClientError")
        assert hasattr(agentfield, "ExecutionTimeoutError")
        assert hasattr(agentfield, "MemoryAccessError")
        assert hasattr(agentfield, "RegistrationError")
        assert hasattr(agentfield, "ValidationError")

    def test_all_exports(self):
        import agentfield

        for name in [
            "AgentFieldError",
            "AgentFieldClientError",
            "ExecutionTimeoutError",
            "MemoryAccessError",
            "RegistrationError",
            "ValidationError",
        ]:
            assert name in agentfield.__all__


# ---------------------------------------------------------------------------
# Client — registration
# ---------------------------------------------------------------------------


class TestClientRegistrationError:
    @responses_lib.activate
    def test_register_node_network_error_raises_registration_error(self):
        responses_lib.add(
            responses_lib.POST,
            "http://localhost:8080/api/v1/nodes/register",
            body=ConnectionError("refused"),
        )
        client = AgentFieldClient(base_url="http://localhost:8080")
        with pytest.raises(RegistrationError, match="Failed to register node"):
            client.register_node({"node_id": "test"})

    @responses_lib.activate
    def test_register_node_http_500_raises_registration_error(self):
        responses_lib.add(
            responses_lib.POST,
            "http://localhost:8080/api/v1/nodes/register",
            json={"error": "internal"},
            status=500,
        )
        client = AgentFieldClient(base_url="http://localhost:8080")
        with pytest.raises(RegistrationError):
            client.register_node({"node_id": "test"})

    @responses_lib.activate
    def test_register_node_bad_json_raises_registration_error(self):
        """JSONDecodeError from response.json() should be wrapped."""
        responses_lib.add(
            responses_lib.POST,
            "http://localhost:8080/api/v1/nodes/register",
            body="not-json",
            status=200,
            content_type="application/json",
        )
        client = AgentFieldClient(base_url="http://localhost:8080")
        with pytest.raises(RegistrationError):
            client.register_node({"node_id": "test"})

    @responses_lib.activate
    def test_register_node_success(self):
        responses_lib.add(
            responses_lib.POST,
            "http://localhost:8080/api/v1/nodes/register",
            json={"status": "registered"},
            status=200,
        )
        client = AgentFieldClient(base_url="http://localhost:8080")
        result = client.register_node({"node_id": "test"})
        assert result == {"status": "registered"}


# ---------------------------------------------------------------------------
# Client — execution submission
# ---------------------------------------------------------------------------


class TestClientExecutionErrors:
    @responses_lib.activate
    def test_submit_execution_network_error_raises_client_error(self):
        responses_lib.add(
            responses_lib.POST,
            "http://localhost:8080/api/v1/execute/async/agent.skill",
            body=requests.ConnectionError("refused"),
        )
        client = AgentFieldClient(base_url="http://localhost:8080")
        with pytest.raises(AgentFieldClientError, match="Failed to submit execution"):
            client._submit_execution_sync(
                "agent.skill", {"key": "value"}, {}
            )

    @responses_lib.activate
    def test_parse_submission_missing_ids_raises_client_error(self):
        client = AgentFieldClient(base_url="http://localhost:8080")
        with pytest.raises(
            AgentFieldClientError, match="missing identifiers"
        ):
            client._parse_submission(
                {"status": "pending"},  # no execution_id or run_id
                {"X-Run-ID": ""},
                "agent.skill",
            )


# ---------------------------------------------------------------------------
# Client — execution timeout
# ---------------------------------------------------------------------------


class TestClientExecutionTimeout:
    @responses_lib.activate
    def test_sync_poll_timeout_raises_execution_timeout(self):
        """When polling exceeds max_execution_timeout, raise ExecutionTimeoutError."""
        responses_lib.add(
            responses_lib.POST,
            "http://localhost:8080/api/v1/execute/async/agent.skill",
            json={
                "execution_id": "exec-1",
                "run_id": "run-1",
                "status": "pending",
            },
            status=200,
        )
        # Always return 'running' to trigger timeout
        responses_lib.add(
            responses_lib.GET,
            "http://localhost:8080/api/v1/executions/exec-1",
            json={"status": "running"},
            status=200,
        )

        client = AgentFieldClient(base_url="http://localhost:8080")
        client.async_config.max_execution_timeout = 0.01
        client.async_config.initial_poll_interval = 0.005

        with pytest.raises(ExecutionTimeoutError):
            client.execute_sync(
                target="agent.skill",
                input_data={"key": "value"},
            )


# ---------------------------------------------------------------------------
# Client — async execution disabled
# ---------------------------------------------------------------------------


class TestAsyncExecutionDisabled:
    """Methods that require async execution should raise AgentFieldClientError
    when it is disabled."""

    @pytest.fixture
    def client(self):
        c = AgentFieldClient(base_url="http://localhost:8080")
        c.async_config.enable_async_execution = False
        return c

    @pytest.mark.parametrize(
        "method,args",
        [
            ("execute_async", ("agent.skill", {})),
            ("poll_execution_status", ("exec-1",)),
            ("batch_check_statuses", (["exec-1"],)),
            ("wait_for_execution_result", ("exec-1",)),
            ("cancel_async_execution", ("exec-1",)),
            ("list_async_executions", ()),
            ("get_async_execution_metrics", ()),
            ("cleanup_async_executions", ()),
        ],
    )
    def test_disabled_raises_client_error(self, client, method, args):
        coro = getattr(client, method)(*args)
        with pytest.raises(AgentFieldClientError, match="disabled"):
            asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# Client — validation
# ---------------------------------------------------------------------------


class TestClientValidation:
    def test_batch_check_empty_ids_raises_validation_error(self):
        client = AgentFieldClient(base_url="http://localhost:8080")
        client.async_config.enable_async_execution = True

        coro = client.batch_check_statuses([])
        with pytest.raises(ValidationError, match="cannot be empty"):
            asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# Memory — MemoryAccessError wrapping
# ---------------------------------------------------------------------------


class TestMemoryAccessErrors:
    """MemoryClient methods should wrap transport errors as MemoryAccessError."""

    @pytest.fixture
    def memory_client(self):
        from agentfield.memory import MemoryClient
        from agentfield.execution_context import ExecutionContext

        af_client = MagicMock()
        af_client.api_base = "http://localhost:8080/api/v1"

        ctx = MagicMock(spec=ExecutionContext)
        ctx.to_headers.return_value = {}

        return MemoryClient(af_client, ctx, agent_node_id="test-agent")

    @pytest.mark.asyncio
    async def test_set_wraps_transport_error(self, memory_client):
        memory_client._async_request = AsyncMock(
            side_effect=ConnectionError("refused")
        )
        with pytest.raises(MemoryAccessError, match="Failed to set memory key"):
            await memory_client.set("key", "value")

    @pytest.mark.asyncio
    async def test_get_wraps_transport_error(self, memory_client):
        memory_client._async_request = AsyncMock(
            side_effect=ConnectionError("refused")
        )
        with pytest.raises(MemoryAccessError, match="Failed to get memory key"):
            await memory_client.get("key")

    @pytest.mark.asyncio
    async def test_delete_wraps_transport_error(self, memory_client):
        memory_client._async_request = AsyncMock(
            side_effect=ConnectionError("refused")
        )
        with pytest.raises(MemoryAccessError, match="Failed to delete memory key"):
            await memory_client.delete("key")

    @pytest.mark.asyncio
    async def test_list_keys_wraps_transport_error(self, memory_client):
        memory_client._async_request = AsyncMock(
            side_effect=ConnectionError("refused")
        )
        with pytest.raises(MemoryAccessError, match="Failed to list keys"):
            await memory_client.list_keys("global")

    @pytest.mark.asyncio
    async def test_set_vector_wraps_transport_error(self, memory_client):
        memory_client._async_request = AsyncMock(
            side_effect=ConnectionError("refused")
        )
        with pytest.raises(MemoryAccessError, match="Failed to set vector key"):
            await memory_client.set_vector("key", [0.1, 0.2])

    @pytest.mark.asyncio
    async def test_delete_vector_wraps_transport_error(self, memory_client):
        memory_client._async_request = AsyncMock(
            side_effect=ConnectionError("refused")
        )
        with pytest.raises(MemoryAccessError, match="Failed to delete vector key"):
            await memory_client.delete_vector("key")

    @pytest.mark.asyncio
    async def test_similarity_search_wraps_transport_error(self, memory_client):
        memory_client._async_request = AsyncMock(
            side_effect=ConnectionError("refused")
        )
        with pytest.raises(MemoryAccessError, match="Failed to perform similarity"):
            await memory_client.similarity_search([0.1, 0.2])

    @pytest.mark.asyncio
    async def test_get_returns_default_on_404(self, memory_client):
        """404 responses should return the default value, not raise."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        memory_client._async_request = AsyncMock(return_value=mock_response)

        result = await memory_client.get("missing", default="fallback")
        assert result == "fallback"

    @pytest.mark.asyncio
    async def test_exists_returns_false_on_error(self, memory_client):
        """exists() intentionally suppresses errors and returns False."""
        memory_client._async_request = AsyncMock(
            side_effect=ConnectionError("refused")
        )
        result = await memory_client.exists("key")
        assert result is False


# ---------------------------------------------------------------------------
# Memory — no double-wrapping
# ---------------------------------------------------------------------------


class TestMemoryNoDoubleWrap:
    """MemoryAccessError raised internally should not be double-wrapped."""

    @pytest.fixture
    def memory_client(self):
        from agentfield.memory import MemoryClient
        from agentfield.execution_context import ExecutionContext

        af_client = MagicMock()
        af_client.api_base = "http://localhost:8080/api/v1"

        ctx = MagicMock(spec=ExecutionContext)
        ctx.to_headers.return_value = {}

        return MemoryClient(af_client, ctx, agent_node_id="test-agent")

    @pytest.mark.asyncio
    async def test_set_does_not_double_wrap(self, memory_client):
        original = MemoryAccessError("inner error")
        memory_client._async_request = AsyncMock(side_effect=original)

        # Patch hasattr check for _async_request on the af_client
        memory_client.agentfield_client._async_request = memory_client._async_request

        with pytest.raises(MemoryAccessError) as exc_info:
            await memory_client.set("key", "value")

        # Should be the original, not a wrapper
        assert exc_info.value is original or exc_info.value.__cause__ is None

    @pytest.mark.asyncio
    async def test_get_does_not_double_wrap(self, memory_client):
        original = MemoryAccessError("inner error")
        memory_client._async_request = AsyncMock(side_effect=original)

        with pytest.raises(MemoryAccessError) as exc_info:
            await memory_client.get("key")

        assert exc_info.value is original or exc_info.value.__cause__ is None

    @pytest.mark.asyncio
    async def test_delete_does_not_double_wrap(self, memory_client):
        original = MemoryAccessError("inner error")
        memory_client._async_request = AsyncMock(side_effect=original)

        with pytest.raises(MemoryAccessError) as exc_info:
            await memory_client.delete("key")

        assert exc_info.value is original or exc_info.value.__cause__ is None

    @pytest.mark.asyncio
    async def test_list_keys_does_not_double_wrap(self, memory_client):
        original = MemoryAccessError("inner error")
        memory_client._async_request = AsyncMock(side_effect=original)

        with pytest.raises(MemoryAccessError) as exc_info:
            await memory_client.list_keys("global")

        assert exc_info.value is original or exc_info.value.__cause__ is None
