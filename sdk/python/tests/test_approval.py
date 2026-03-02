"""Tests for approval workflow helpers on AgentFieldClient."""

import pytest

pytest.importorskip("pytest_httpx", reason="pytest-httpx requires Python >=3.10")

from agentfield.client import (
    AgentFieldClient,
    ApprovalRequestResponse,
    ApprovalStatusResponse,
)
from agentfield.exceptions import AgentFieldClientError, ExecutionTimeoutError


BASE_URL = "http://localhost:8080"
API_BASE = f"{BASE_URL}/api/v1"
NODE_ID = "test-node"
EXECUTION_ID = "exec-123"


@pytest.fixture
def client():
    """Create an AgentFieldClient pointed at a mock control plane."""
    c = AgentFieldClient(base_url=BASE_URL, api_key="test-key")
    c.caller_agent_id = NODE_ID
    return c


# ---------------------------------------------------------------------------
# request_approval
# ---------------------------------------------------------------------------


async def test_request_approval_returns_typed_response(client, httpx_mock):
    """request_approval should return an ApprovalRequestResponse dataclass."""
    url = f"{API_BASE}/agents/{NODE_ID}/executions/{EXECUTION_ID}/request-approval"
    httpx_mock.add_response(
        method="POST",
        url=url,
        json={
            "approval_request_id": "req-abc",
            "approval_request_url": "https://hub.example.com/r/req-abc",
        },
    )

    result = await client.request_approval(
        execution_id=EXECUTION_ID,
        approval_request_id="req-abc",
        approval_request_url="https://hub.example.com/r/req-abc",
    )

    assert isinstance(result, ApprovalRequestResponse)
    assert result.approval_request_id == "req-abc"
    assert result.approval_request_url == "https://hub.example.com/r/req-abc"


async def test_request_approval_raises_on_http_error(client, httpx_mock):
    """request_approval should raise AgentFieldClientError on 4xx/5xx."""
    url = f"{API_BASE}/agents/{NODE_ID}/executions/{EXECUTION_ID}/request-approval"
    httpx_mock.add_response(
        method="POST",
        url=url,
        json={"error": "execution not found"},
        status_code=404,
    )

    with pytest.raises(AgentFieldClientError, match="404"):
        await client.request_approval(
                execution_id=EXECUTION_ID,
                approval_request_id="req-fail",
            )


# ---------------------------------------------------------------------------
# get_approval_status
# ---------------------------------------------------------------------------


async def test_get_approval_status_returns_typed_response(client, httpx_mock):
    """get_approval_status should return an ApprovalStatusResponse dataclass."""
    url = f"{API_BASE}/agents/{NODE_ID}/executions/{EXECUTION_ID}/approval-status"
    httpx_mock.add_response(
        method="GET",
        url=url,
        json={
            "status": "approved",
            "response": {"decision": "approved", "feedback": "LGTM"},
            "request_url": "https://hub.example.com/r/req-abc",
            "requested_at": "2026-02-25T10:00:00Z",
            "responded_at": "2026-02-25T11:00:00Z",
        },
    )

    result = await client.get_approval_status(EXECUTION_ID)

    assert isinstance(result, ApprovalStatusResponse)
    assert result.status == "approved"
    assert result.response == {"decision": "approved", "feedback": "LGTM"}
    assert result.request_url == "https://hub.example.com/r/req-abc"
    assert result.requested_at == "2026-02-25T10:00:00Z"
    assert result.responded_at == "2026-02-25T11:00:00Z"


async def test_get_approval_status_pending(client, httpx_mock):
    """get_approval_status should return pending when not yet resolved."""
    url = f"{API_BASE}/agents/{NODE_ID}/executions/{EXECUTION_ID}/approval-status"
    httpx_mock.add_response(
        method="GET",
        url=url,
        json={
            "status": "pending",
            "request_url": "https://hub.example.com/r/req-abc",
            "requested_at": "2026-02-25T10:00:00Z",
        },
    )

    result = await client.get_approval_status(EXECUTION_ID)

    assert isinstance(result, ApprovalStatusResponse)
    assert result.status == "pending"
    assert result.responded_at is None
    assert result.response is None


async def test_get_approval_status_expired(client, httpx_mock):
    """get_approval_status should return expired when request times out."""
    url = f"{API_BASE}/agents/{NODE_ID}/executions/{EXECUTION_ID}/approval-status"
    httpx_mock.add_response(
        method="GET",
        url=url,
        json={
            "status": "expired",
            "request_url": "https://hub.example.com/r/req-abc",
            "requested_at": "2026-02-25T10:00:00Z",
            "responded_at": "2026-02-28T10:00:00Z",
        },
    )

    result = await client.get_approval_status(EXECUTION_ID)

    assert isinstance(result, ApprovalStatusResponse)
    assert result.status == "expired"
    assert result.responded_at == "2026-02-28T10:00:00Z"


async def test_get_approval_status_raises_on_http_error(client, httpx_mock):
    """get_approval_status should raise on server errors."""
    url = f"{API_BASE}/agents/{NODE_ID}/executions/{EXECUTION_ID}/approval-status"
    httpx_mock.add_response(
        method="GET",
        url=url,
        json={"error": "internal"},
        status_code=500,
    )

    with pytest.raises(AgentFieldClientError, match="500"):
        await client.get_approval_status(EXECUTION_ID)


# ---------------------------------------------------------------------------
# wait_for_approval
# ---------------------------------------------------------------------------


async def test_wait_for_approval_resolves_on_approved(client, httpx_mock):
    """wait_for_approval should return once status is no longer pending."""
    url = f"{API_BASE}/agents/{NODE_ID}/executions/{EXECUTION_ID}/approval-status"

    # First call returns pending, second returns approved
    httpx_mock.add_response(method="GET", url=url, json={"status": "pending"})
    httpx_mock.add_response(
        method="GET",
        url=url,
        json={"status": "approved", "response": {"decision": "approved"}},
    )

    result = await client.wait_for_approval(
        EXECUTION_ID,
        poll_interval=0.01,
        max_interval=0.02,
    )

    assert isinstance(result, ApprovalStatusResponse)
    assert result.status == "approved"


async def test_wait_for_approval_resolves_on_rejected(client, httpx_mock):
    """wait_for_approval should return on rejected status."""
    url = f"{API_BASE}/agents/{NODE_ID}/executions/{EXECUTION_ID}/approval-status"
    httpx_mock.add_response(
        method="GET",
        url=url,
        json={"status": "rejected", "response": {"feedback": "needs work"}},
    )

    result = await client.wait_for_approval(EXECUTION_ID, poll_interval=0.01)

    assert result.status == "rejected"


async def test_wait_for_approval_resolves_on_expired(client, httpx_mock):
    """wait_for_approval should return on expired status."""
    url = f"{API_BASE}/agents/{NODE_ID}/executions/{EXECUTION_ID}/approval-status"
    httpx_mock.add_response(
        method="GET",
        url=url,
        json={"status": "expired", "request_url": "https://hub.example.com/r/req-abc"},
    )

    result = await client.wait_for_approval(EXECUTION_ID, poll_interval=0.01)

    assert result.status == "expired"


@pytest.mark.httpx_mock(assert_all_responses_were_requested=False)
async def test_wait_for_approval_timeout(client, httpx_mock):
    """wait_for_approval should raise ExecutionTimeoutError on timeout."""
    url = f"{API_BASE}/agents/{NODE_ID}/executions/{EXECUTION_ID}/approval-status"

    # Always return pending (add enough responses for the polling loop)
    for _ in range(20):
        httpx_mock.add_response(method="GET", url=url, json={"status": "pending"})

    with pytest.raises(ExecutionTimeoutError, match="timed out"):
        await client.wait_for_approval(
            EXECUTION_ID,
            poll_interval=0.01,
            max_interval=0.01,
            timeout=0.05,
        )


async def test_wait_for_approval_retries_on_transient_error(client, httpx_mock):
    """wait_for_approval should back off and retry on transient HTTP errors."""
    url = f"{API_BASE}/agents/{NODE_ID}/executions/{EXECUTION_ID}/approval-status"

    # First call fails, second succeeds
    httpx_mock.add_response(
        method="GET", url=url, json={"error": "transient"}, status_code=500
    )
    httpx_mock.add_response(method="GET", url=url, json={"status": "approved"})

    result = await client.wait_for_approval(EXECUTION_ID, poll_interval=0.01)

    assert result.status == "approved"
