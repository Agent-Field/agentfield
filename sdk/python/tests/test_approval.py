"""Tests for approval workflow and async helper methods on AgentFieldClient."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agentfield.client import (
    AgentFieldClient,
    ApprovalRequestResponse,
    ApprovalStatusResponse,
)
from agentfield.exceptions import AgentFieldClientError, ExecutionTimeoutError
from agentfield.execution_state import ExecutionStatus


BASE_URL = "http://localhost:8080"
API_BASE = f"{BASE_URL}/api/v1"
NODE_ID = "test-node"
EXECUTION_ID = "exec-123"


class FakeResponse:
    def __init__(self, status_code: int = 200, payload: dict | None = None, text: str = ""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text or "{}"

    def json(self) -> dict:
        return self._payload


@pytest.fixture
def client() -> AgentFieldClient:
    c = AgentFieldClient(base_url=BASE_URL, api_key="test-key")
    c.caller_agent_id = NODE_ID
    return c


@pytest.mark.asyncio
async def test_request_approval_returns_typed_response_and_payload(client: AgentFieldClient):
    http_client = SimpleNamespace(
        post=AsyncMock(
            return_value=FakeResponse(
                payload={
                    "approval_request_id": "req-abc",
                    "approval_request_url": "https://hub.example.com/r/req-abc",
                }
            )
        )
    )
    client.get_async_http_client = AsyncMock(return_value=http_client)

    result = await client.request_approval(
        execution_id=EXECUTION_ID,
        approval_request_id="req-abc",
        approval_request_url="https://hub.example.com/r/req-abc",
        callback_url="https://callback.example.com/approval",
        expires_in_hours=24,
    )

    assert isinstance(result, ApprovalRequestResponse)
    assert result.approval_request_id == "req-abc"
    assert result.approval_request_url == "https://hub.example.com/r/req-abc"
    http_client.post.assert_awaited_once()
    assert http_client.post.await_args.args[0] == (
        f"{API_BASE}/agents/{NODE_ID}/executions/{EXECUTION_ID}/request-approval"
    )
    assert http_client.post.await_args.kwargs["json"] == {
        "approval_request_id": "req-abc",
        "approval_request_url": "https://hub.example.com/r/req-abc",
        "callback_url": "https://callback.example.com/approval",
        "expires_in_hours": 24,
    }


@pytest.mark.asyncio
async def test_request_approval_wraps_transport_errors(client: AgentFieldClient):
    http_client = SimpleNamespace(post=AsyncMock(side_effect=RuntimeError("boom")))
    client.get_async_http_client = AsyncMock(return_value=http_client)

    with pytest.raises(AgentFieldClientError, match="Failed to request approval: boom"):
        await client.request_approval(EXECUTION_ID, approval_request_id="req-fail")


@pytest.mark.asyncio
async def test_request_approval_raises_on_http_error(client: AgentFieldClient):
    http_client = SimpleNamespace(
        post=AsyncMock(return_value=FakeResponse(status_code=404, text='{"error":"missing"}'))
    )
    client.get_async_http_client = AsyncMock(return_value=http_client)

    with pytest.raises(AgentFieldClientError, match="404"):
        await client.request_approval(EXECUTION_ID, approval_request_id="req-fail")


@pytest.mark.asyncio
async def test_get_approval_status_returns_typed_response(client: AgentFieldClient):
    http_client = SimpleNamespace(
        get=AsyncMock(
            return_value=FakeResponse(
                payload={
                    "status": "approved",
                    "response": {"decision": "approved", "feedback": "LGTM"},
                    "request_url": "https://hub.example.com/r/req-abc",
                    "requested_at": "2026-02-25T10:00:00Z",
                    "responded_at": "2026-02-25T11:00:00Z",
                }
            )
        )
    )
    client.get_async_http_client = AsyncMock(return_value=http_client)

    result = await client.get_approval_status(EXECUTION_ID)

    assert isinstance(result, ApprovalStatusResponse)
    assert result.status == "approved"
    assert result.response == {"decision": "approved", "feedback": "LGTM"}
    assert result.request_url == "https://hub.example.com/r/req-abc"
    assert result.requested_at == "2026-02-25T10:00:00Z"
    assert result.responded_at == "2026-02-25T11:00:00Z"
    http_client.get.assert_awaited_once()
    assert http_client.get.await_args.args[0] == (
        f"{API_BASE}/agents/{NODE_ID}/executions/{EXECUTION_ID}/approval-status"
    )


@pytest.mark.asyncio
async def test_get_approval_status_wraps_transport_errors(client: AgentFieldClient):
    http_client = SimpleNamespace(get=AsyncMock(side_effect=RuntimeError("boom")))
    client.get_async_http_client = AsyncMock(return_value=http_client)

    with pytest.raises(AgentFieldClientError, match="Failed to get approval status: boom"):
        await client.get_approval_status(EXECUTION_ID)


@pytest.mark.asyncio
async def test_get_approval_status_raises_on_http_error(client: AgentFieldClient):
    http_client = SimpleNamespace(
        get=AsyncMock(return_value=FakeResponse(status_code=500, text='{"error":"internal"}'))
    )
    client.get_async_http_client = AsyncMock(return_value=http_client)

    with pytest.raises(AgentFieldClientError, match="500"):
        await client.get_approval_status(EXECUTION_ID)


@pytest.mark.asyncio
async def test_wait_for_approval_returns_first_resolved_status(
    client: AgentFieldClient, monkeypatch: pytest.MonkeyPatch
):
    pending = ApprovalStatusResponse(status="pending")
    approved = ApprovalStatusResponse(status="approved", response={"decision": "approved"})
    client.get_approval_status = AsyncMock(side_effect=[pending, approved])

    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr("agentfield.client.asyncio.sleep", no_sleep)

    result = await client.wait_for_approval(EXECUTION_ID, poll_interval=0.01, max_interval=0.02)

    assert result.status == "approved"
    assert client.get_approval_status.await_count == 2


@pytest.mark.asyncio
async def test_wait_for_approval_retries_transient_client_errors(
    client: AgentFieldClient, monkeypatch: pytest.MonkeyPatch
):
    approved = ApprovalStatusResponse(status="approved", response={"decision": "approved"})
    client.get_approval_status = AsyncMock(
        side_effect=[AgentFieldClientError("transient"), approved]
    )

    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr("agentfield.client.asyncio.sleep", no_sleep)

    result = await client.wait_for_approval(EXECUTION_ID, poll_interval=0.01, max_interval=0.02)

    assert result.status == "approved"
    assert client.get_approval_status.await_count == 2


@pytest.mark.asyncio
async def test_wait_for_approval_times_out(
    client: AgentFieldClient, monkeypatch: pytest.MonkeyPatch
):
    import agentfield.client as client_module

    client.get_approval_status = AsyncMock(return_value=ApprovalStatusResponse(status="pending"))

    async def no_sleep(_: float) -> None:
        return None

    times = iter([0.0, 0.02, 0.06])
    monkeypatch.setattr("agentfield.client.asyncio.sleep", no_sleep)
    monkeypatch.setattr(client_module.time, "time", lambda: next(times))

    with pytest.raises(ExecutionTimeoutError, match="timed out"):
        await client.wait_for_approval(
            EXECUTION_ID,
            poll_interval=0.01,
            max_interval=0.02,
            timeout=0.05,
        )


@pytest.mark.asyncio
async def test_batch_check_statuses_uses_batched_path(client: AgentFieldClient):
    manager = SimpleNamespace(
        get_execution_status=AsyncMock(side_effect=[{"status": "queued"}, {"status": "running"}])
    )
    client._get_async_execution_manager = AsyncMock(return_value=manager)
    client.async_config.enable_async_execution = True
    client.async_config.enable_batch_polling = True
    client.async_config.batch_size = 2

    result = await client.batch_check_statuses(["exec-1", "exec-2"])

    assert result == {"exec-1": {"status": "queued"}, "exec-2": {"status": "running"}}
    assert manager.get_execution_status.await_count == 2


@pytest.mark.asyncio
async def test_batch_check_statuses_uses_individual_path(client: AgentFieldClient):
    manager = SimpleNamespace(get_execution_status=AsyncMock(return_value={"status": "done"}))
    client._get_async_execution_manager = AsyncMock(return_value=manager)
    client.async_config.enable_async_execution = True
    client.async_config.enable_batch_polling = False

    result = await client.batch_check_statuses(["exec-1"])

    assert result == {"exec-1": {"status": "done"}}
    manager.get_execution_status.assert_awaited_once_with("exec-1")


@pytest.mark.asyncio
async def test_list_async_executions_normalizes_status_filter(client: AgentFieldClient):
    manager = SimpleNamespace(list_executions=AsyncMock(return_value=[{"id": "exec-1"}]))
    client._get_async_execution_manager = AsyncMock(return_value=manager)
    client.async_config.enable_async_execution = True

    result = await client.list_async_executions(status_filter="RUNNING", limit=5)

    assert result == [{"id": "exec-1"}]
    manager.list_executions.assert_awaited_once_with(ExecutionStatus.RUNNING, 5)


@pytest.mark.asyncio
async def test_list_async_executions_returns_empty_for_invalid_status(client: AgentFieldClient):
    manager = SimpleNamespace(list_executions=AsyncMock())
    client._get_async_execution_manager = AsyncMock(return_value=manager)
    client.async_config.enable_async_execution = True

    result = await client.list_async_executions(status_filter="not-a-status")

    assert result == []
    manager.list_executions.assert_not_called()


@pytest.mark.asyncio
async def test_close_async_execution_manager_stops_and_clears_manager(client: AgentFieldClient):
    manager = SimpleNamespace(stop=AsyncMock())
    client._async_execution_manager = manager

    await client.close_async_execution_manager()

    manager.stop.assert_awaited_once()
    assert client._async_execution_manager is None


@pytest.mark.asyncio
async def test_aclose_cleans_up_manager_and_http_client(client: AgentFieldClient):
    manager = SimpleNamespace(stop=AsyncMock())
    http_client = SimpleNamespace(aclose=AsyncMock())
    client._async_execution_manager = manager
    client._async_http_client = http_client
    client._async_http_client_lock = object()

    await client.aclose()

    manager.stop.assert_awaited_once()
    http_client.aclose.assert_awaited_once()
    assert client._async_execution_manager is None
    assert client._async_http_client is None
    assert client._async_http_client_lock is None
