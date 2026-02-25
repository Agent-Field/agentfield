"""Tests for approval workflow helpers on AgentFieldClient."""

import asyncio
import json
from unittest.mock import patch, MagicMock

import pytest
import responses as responses_lib

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


@responses_lib.activate
def test_request_approval_returns_typed_response(client):
    """request_approval should return an ApprovalRequestResponse dataclass."""
    url = f"{API_BASE}/agents/{NODE_ID}/executions/{EXECUTION_ID}/request-approval"
    responses_lib.add(
        responses_lib.POST,
        url,
        json={
            "approval_request_id": "req-abc",
            "approval_request_url": "https://hub.example.com/r/req-abc",
        },
        status=200,
    )

    result = asyncio.get_event_loop().run_until_complete(
        client.request_approval(
            execution_id=EXECUTION_ID,
            title="Plan Review",
            project_id="proj-1",
        )
    )

    assert isinstance(result, ApprovalRequestResponse)
    assert result.approval_request_id == "req-abc"
    assert result.approval_request_url == "https://hub.example.com/r/req-abc"


@responses_lib.activate
def test_request_approval_raises_on_http_error(client):
    """request_approval should raise AgentFieldClientError on 4xx/5xx."""
    url = f"{API_BASE}/agents/{NODE_ID}/executions/{EXECUTION_ID}/request-approval"
    responses_lib.add(
        responses_lib.POST,
        url,
        json={"error": "execution not found"},
        status=404,
    )

    with pytest.raises(AgentFieldClientError, match="404"):
        asyncio.get_event_loop().run_until_complete(
            client.request_approval(execution_id=EXECUTION_ID, project_id="p")
        )


# ---------------------------------------------------------------------------
# get_approval_status
# ---------------------------------------------------------------------------


@responses_lib.activate
def test_get_approval_status_returns_typed_response(client):
    """get_approval_status should return an ApprovalStatusResponse dataclass."""
    url = f"{API_BASE}/agents/{NODE_ID}/executions/{EXECUTION_ID}/approval-status"
    responses_lib.add(
        responses_lib.GET,
        url,
        json={
            "status": "approved",
            "response": {"decision": "approved", "feedback": "LGTM"},
            "request_url": "https://hub.example.com/r/req-abc",
            "requested_at": "2026-02-25T10:00:00Z",
            "responded_at": "2026-02-25T11:00:00Z",
        },
        status=200,
    )

    result = asyncio.get_event_loop().run_until_complete(
        client.get_approval_status(EXECUTION_ID)
    )

    assert isinstance(result, ApprovalStatusResponse)
    assert result.status == "approved"
    assert result.response == {"decision": "approved", "feedback": "LGTM"}
    assert result.request_url == "https://hub.example.com/r/req-abc"
    assert result.requested_at == "2026-02-25T10:00:00Z"
    assert result.responded_at == "2026-02-25T11:00:00Z"


@responses_lib.activate
def test_get_approval_status_pending(client):
    """get_approval_status should return pending when not yet resolved."""
    url = f"{API_BASE}/agents/{NODE_ID}/executions/{EXECUTION_ID}/approval-status"
    responses_lib.add(
        responses_lib.GET,
        url,
        json={
            "status": "pending",
            "request_url": "https://hub.example.com/r/req-abc",
            "requested_at": "2026-02-25T10:00:00Z",
        },
        status=200,
    )

    result = asyncio.get_event_loop().run_until_complete(
        client.get_approval_status(EXECUTION_ID)
    )

    assert isinstance(result, ApprovalStatusResponse)
    assert result.status == "pending"
    assert result.responded_at is None
    assert result.response is None


@responses_lib.activate
def test_get_approval_status_raises_on_http_error(client):
    """get_approval_status should raise on server errors."""
    url = f"{API_BASE}/agents/{NODE_ID}/executions/{EXECUTION_ID}/approval-status"
    responses_lib.add(
        responses_lib.GET,
        url,
        json={"error": "internal"},
        status=500,
    )

    with pytest.raises(AgentFieldClientError, match="500"):
        asyncio.get_event_loop().run_until_complete(
            client.get_approval_status(EXECUTION_ID)
        )


# ---------------------------------------------------------------------------
# wait_for_approval
# ---------------------------------------------------------------------------


@responses_lib.activate
def test_wait_for_approval_resolves_on_approved(client):
    """wait_for_approval should return once status is no longer pending."""
    url = f"{API_BASE}/agents/{NODE_ID}/executions/{EXECUTION_ID}/approval-status"

    # First call returns pending, second returns approved
    responses_lib.add(
        responses_lib.GET, url,
        json={"status": "pending"},
        status=200,
    )
    responses_lib.add(
        responses_lib.GET, url,
        json={"status": "approved", "response": {"decision": "approved"}},
        status=200,
    )

    result = asyncio.get_event_loop().run_until_complete(
        client.wait_for_approval(
            EXECUTION_ID,
            poll_interval=0.01,
            max_interval=0.02,
        )
    )

    assert isinstance(result, ApprovalStatusResponse)
    assert result.status == "approved"


@responses_lib.activate
def test_wait_for_approval_resolves_on_rejected(client):
    """wait_for_approval should return on rejected status."""
    url = f"{API_BASE}/agents/{NODE_ID}/executions/{EXECUTION_ID}/approval-status"

    responses_lib.add(
        responses_lib.GET, url,
        json={"status": "rejected", "response": {"feedback": "needs work"}},
        status=200,
    )

    result = asyncio.get_event_loop().run_until_complete(
        client.wait_for_approval(EXECUTION_ID, poll_interval=0.01)
    )

    assert result.status == "rejected"


@responses_lib.activate
def test_wait_for_approval_timeout(client):
    """wait_for_approval should raise ExecutionTimeoutError on timeout."""
    url = f"{API_BASE}/agents/{NODE_ID}/executions/{EXECUTION_ID}/approval-status"

    # Always return pending
    for _ in range(10):
        responses_lib.add(
            responses_lib.GET, url,
            json={"status": "pending"},
            status=200,
        )

    with pytest.raises(ExecutionTimeoutError, match="timed out"):
        asyncio.get_event_loop().run_until_complete(
            client.wait_for_approval(
                EXECUTION_ID,
                poll_interval=0.01,
                max_interval=0.01,
                timeout=0.05,
            )
        )


@responses_lib.activate
def test_wait_for_approval_retries_on_transient_error(client):
    """wait_for_approval should back off and retry on transient HTTP errors."""
    url = f"{API_BASE}/agents/{NODE_ID}/executions/{EXECUTION_ID}/approval-status"

    # First call fails, second succeeds
    responses_lib.add(
        responses_lib.GET, url,
        json={"error": "transient"},
        status=500,
    )
    responses_lib.add(
        responses_lib.GET, url,
        json={"status": "approved"},
        status=200,
    )

    result = asyncio.get_event_loop().run_until_complete(
        client.wait_for_approval(EXECUTION_ID, poll_interval=0.01)
    )

    assert result.status == "approved"
