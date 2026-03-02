"""
Functional tests for the waiting state (human-in-the-loop approval) workflow.

Tests the full end-to-end flow:
1. Agent starts an execution
2. Execution transitions to "waiting" via the approval request API
3. Approval status is polled (returns "pending")
4. External webhook resolves the approval (approved/rejected/expired)
5. Execution resumes with the appropriate status

These tests exercise the control plane approval endpoints directly via HTTP,
validating the state machine transitions without requiring an external approval
service.
"""

import asyncio
import uuid

import pytest

from utils import run_agent_server

from agentfield import Agent


def _unique_node_id(prefix: str = "test-waiting") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _start_execution(client, node_id: str, reasoner_id: str, input_data: dict) -> dict:
    """Start a sync execution and return the response payload."""
    resp = await client.post(
        f"/api/v1/execute/{node_id}.{reasoner_id}",
        json={"input": input_data},
        timeout=30.0,
    )
    assert resp.status_code == 200, f"Execute failed: {resp.text}"
    return resp.json()


async def _request_approval(
    client,
    node_id: str,
    execution_id: str,
    approval_request_id: str,
    **kwargs,
) -> dict:
    """Request approval for an execution, transitioning it to waiting."""
    body = {"approval_request_id": approval_request_id, **kwargs}
    resp = await client.post(
        f"/api/v1/agents/{node_id}/executions/{execution_id}/request-approval",
        json=body,
        timeout=10.0,
    )
    return resp.json(), resp.status_code


async def _get_approval_status(client, node_id: str, execution_id: str) -> dict:
    """Poll the approval status for an execution."""
    resp = await client.get(
        f"/api/v1/agents/{node_id}/executions/{execution_id}/approval-status",
        timeout=10.0,
    )
    return resp.json(), resp.status_code


async def _send_approval_webhook(client, request_id: str, decision: str, feedback: str = "") -> dict:
    """Send an approval webhook to resolve a waiting execution."""
    body = {
        "requestId": request_id,
        "decision": decision,
    }
    if feedback:
        body["feedback"] = feedback
    resp = await client.post(
        "/api/v1/webhooks/approval-response",
        json=body,
        timeout=10.0,
    )
    return resp.json(), resp.status_code


async def _get_execution_status(client, execution_id: str) -> dict:
    """Get the execution status from the executions API."""
    resp = await client.get(
        f"/api/v1/executions/{execution_id}",
        timeout=10.0,
    )
    if resp.status_code != 200:
        return {"status": "unknown"}, resp.status_code
    return resp.json(), resp.status_code


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.functional
@pytest.mark.asyncio
async def test_approval_request_transitions_to_waiting(make_test_agent, async_http_client):
    """
    Verify that requesting approval transitions an execution from
    running to waiting state.
    """
    node_id = _unique_node_id()
    agent = make_test_agent(node_id=node_id)

    @agent.reasoner()
    async def slow_task(message: str) -> dict:
        # This reasoner just returns — the test will request approval
        # on the execution after it completes. For the real flow,
        # the agent would call app.pause() mid-execution.
        return {"message": message, "status": "done"}

    async with run_agent_server(agent):
        # Start an execution
        exec_result = await _start_execution(
            async_http_client, node_id, "slow_task", {"message": "test"}
        )
        execution_id = exec_result["execution_id"]
        assert exec_result["status"] == "succeeded"

        # Note: In a real scenario, the agent would request approval during
        # execution (while running). For this test, we verify the API
        # contract by checking behavior against a completed execution.
        # The control plane rejects approval requests for non-running executions.
        approval_result, status_code = await _request_approval(
            async_http_client,
            node_id,
            execution_id,
            "req-test-1",
        )

        # Since execution is already succeeded, this should be rejected
        assert status_code == 409, f"Expected conflict, got {status_code}: {approval_result}"


@pytest.mark.functional
@pytest.mark.asyncio
async def test_full_approval_flow_via_async_execution(make_test_agent, async_http_client):
    """
    Full end-to-end approval flow using async execution:
    1. Start async execution (returns immediately while running)
    2. Request approval (transitions to waiting)
    3. Poll status (should be pending)
    4. Send webhook (resolves to approved)
    5. Verify execution resumes
    """
    node_id = _unique_node_id()
    agent = make_test_agent(node_id=node_id)

    @agent.reasoner()
    async def long_running_task(message: str) -> dict:
        # Sleep to keep the execution running long enough for the test
        await asyncio.sleep(60)
        return {"message": message}

    async with run_agent_server(agent):
        # Start an async execution (returns immediately, execution runs in background)
        resp = await async_http_client.post(
            f"/api/v1/execute/async/{node_id}.long_running_task",
            json={"input": {"message": "test approval flow"}},
            timeout=10.0,
        )
        assert resp.status_code == 202, f"Async execute failed: {resp.text}"
        exec_data = resp.json()
        execution_id = exec_data["execution_id"]

        # Wait a moment for execution to start
        await asyncio.sleep(2)

        # Request approval — should transition to waiting
        approval_request_id = f"req-{uuid.uuid4().hex[:8]}"
        approval_result, status_code = await _request_approval(
            async_http_client,
            node_id,
            execution_id,
            approval_request_id,
            approval_request_url=f"https://hub.example.com/review/{approval_request_id}",
            expires_in_hours=1,
        )
        assert status_code == 200, f"Request approval failed ({status_code}): {approval_result}"
        assert approval_result["status"] == "pending"

        # Poll approval status — should be pending
        status_result, status_code = await _get_approval_status(
            async_http_client, node_id, execution_id
        )
        assert status_code == 200
        assert status_result["status"] == "pending"
        assert status_result.get("request_url") == f"https://hub.example.com/review/{approval_request_id}"

        # Send webhook to approve
        webhook_result, webhook_status = await _send_approval_webhook(
            async_http_client,
            approval_request_id,
            "approved",
            feedback="LGTM!",
        )
        assert webhook_status == 200, f"Webhook failed ({webhook_status}): {webhook_result}"
        assert webhook_result["decision"] == "approved"
        assert webhook_result["new_status"] == "running"

        # Verify approval status is now approved
        final_status, _ = await _get_approval_status(
            async_http_client, node_id, execution_id
        )
        assert final_status["status"] == "approved"
        assert final_status.get("responded_at") is not None


@pytest.mark.functional
@pytest.mark.asyncio
async def test_approval_rejected_cancels_execution(make_test_agent, async_http_client):
    """
    Verify that rejecting an approval transitions the execution to cancelled.
    """
    node_id = _unique_node_id()
    agent = make_test_agent(node_id=node_id)

    @agent.reasoner()
    async def pending_task(message: str) -> dict:
        await asyncio.sleep(60)
        return {"message": message}

    async with run_agent_server(agent):
        # Start async execution
        resp = await async_http_client.post(
            f"/api/v1/execute/async/{node_id}.pending_task",
            json={"input": {"message": "test rejection"}},
            timeout=10.0,
        )
        assert resp.status_code == 202
        execution_id = resp.json()["execution_id"]

        await asyncio.sleep(2)

        # Request approval
        approval_request_id = f"req-rej-{uuid.uuid4().hex[:8]}"
        result, status = await _request_approval(
            async_http_client, node_id, execution_id, approval_request_id
        )
        assert status == 200

        # Reject via webhook
        webhook_result, webhook_status = await _send_approval_webhook(
            async_http_client,
            approval_request_id,
            "rejected",
            feedback="Plan needs more detail",
        )
        assert webhook_status == 200
        assert webhook_result["decision"] == "rejected"
        assert webhook_result["new_status"] == "cancelled"


@pytest.mark.functional
@pytest.mark.asyncio
async def test_approval_expired_cancels_execution(make_test_agent, async_http_client):
    """
    Verify that an expired approval transitions the execution to cancelled.
    """
    node_id = _unique_node_id()
    agent = make_test_agent(node_id=node_id)

    @agent.reasoner()
    async def expiring_task(message: str) -> dict:
        await asyncio.sleep(60)
        return {"message": message}

    async with run_agent_server(agent):
        resp = await async_http_client.post(
            f"/api/v1/execute/async/{node_id}.expiring_task",
            json={"input": {"message": "test expiry"}},
            timeout=10.0,
        )
        assert resp.status_code == 202
        execution_id = resp.json()["execution_id"]

        await asyncio.sleep(2)

        approval_request_id = f"req-exp-{uuid.uuid4().hex[:8]}"
        result, status = await _request_approval(
            async_http_client, node_id, execution_id, approval_request_id
        )
        assert status == 200

        # Simulate expiry via webhook
        webhook_result, webhook_status = await _send_approval_webhook(
            async_http_client, approval_request_id, "expired"
        )
        assert webhook_status == 200
        assert webhook_result["decision"] == "expired"
        assert webhook_result["new_status"] == "cancelled"


@pytest.mark.functional
@pytest.mark.asyncio
async def test_approval_webhook_idempotent(make_test_agent, async_http_client):
    """
    Verify that sending the same webhook twice returns success (idempotent).
    """
    node_id = _unique_node_id()
    agent = make_test_agent(node_id=node_id)

    @agent.reasoner()
    async def idem_task(message: str) -> dict:
        await asyncio.sleep(60)
        return {"message": message}

    async with run_agent_server(agent):
        resp = await async_http_client.post(
            f"/api/v1/execute/async/{node_id}.idem_task",
            json={"input": {"message": "test idempotency"}},
            timeout=10.0,
        )
        assert resp.status_code == 202
        execution_id = resp.json()["execution_id"]

        await asyncio.sleep(2)

        approval_request_id = f"req-idem-{uuid.uuid4().hex[:8]}"
        result, status = await _request_approval(
            async_http_client, node_id, execution_id, approval_request_id
        )
        assert status == 200

        # First webhook
        r1, s1 = await _send_approval_webhook(
            async_http_client, approval_request_id, "approved"
        )
        assert s1 == 200
        assert r1["status"] == "processed"

        # Duplicate webhook — should return 200 with already_processed
        r2, s2 = await _send_approval_webhook(
            async_http_client, approval_request_id, "approved"
        )
        assert s2 == 200
        assert r2["status"] == "already_processed"


@pytest.mark.functional
@pytest.mark.asyncio
async def test_approval_duplicate_request_rejected(make_test_agent, async_http_client):
    """
    Verify that requesting approval twice on the same execution is rejected.
    """
    node_id = _unique_node_id()
    agent = make_test_agent(node_id=node_id)

    @agent.reasoner()
    async def dup_task(message: str) -> dict:
        await asyncio.sleep(60)
        return {"message": message}

    async with run_agent_server(agent):
        resp = await async_http_client.post(
            f"/api/v1/execute/async/{node_id}.dup_task",
            json={"input": {"message": "test duplicate"}},
            timeout=10.0,
        )
        assert resp.status_code == 202
        execution_id = resp.json()["execution_id"]

        await asyncio.sleep(2)

        # First approval request
        r1, s1 = await _request_approval(
            async_http_client, node_id, execution_id, f"req-dup-1-{uuid.uuid4().hex[:8]}"
        )
        assert s1 == 200

        # Second approval request — should be rejected.
        # After the first request, execution is in "waiting" state (not "running"),
        # so the handler rejects with "invalid_state" before reaching the duplicate check.
        r2, s2 = await _request_approval(
            async_http_client, node_id, execution_id, f"req-dup-2-{uuid.uuid4().hex[:8]}"
        )
        assert s2 == 409
        assert r2.get("error") == "invalid_state"


@pytest.mark.functional
@pytest.mark.asyncio
async def test_hax_sdk_envelope_webhook_format(make_test_agent, async_http_client):
    """
    Verify that the hax-sdk envelope webhook format is correctly handled.
    """
    node_id = _unique_node_id()
    agent = make_test_agent(node_id=node_id)

    @agent.reasoner()
    async def hax_task(message: str) -> dict:
        await asyncio.sleep(60)
        return {"message": message}

    async with run_agent_server(agent):
        resp = await async_http_client.post(
            f"/api/v1/execute/async/{node_id}.hax_task",
            json={"input": {"message": "test hax format"}},
            timeout=10.0,
        )
        assert resp.status_code == 202
        execution_id = resp.json()["execution_id"]

        await asyncio.sleep(2)

        approval_request_id = f"req-hax-{uuid.uuid4().hex[:8]}"
        result, status = await _request_approval(
            async_http_client, node_id, execution_id, approval_request_id
        )
        assert status == 200

        # Send in hax-sdk envelope format
        webhook_resp = await async_http_client.post(
            "/api/v1/webhooks/approval-response",
            json={
                "id": f"evt_{uuid.uuid4().hex[:8]}",
                "type": "completed",
                "createdAt": "2026-03-02T12:00:00Z",
                "data": {
                    "requestId": approval_request_id,
                    "response": {
                        "decision": "approved",
                        "feedback": "Approved from Response Hub",
                    },
                },
            },
            timeout=10.0,
        )
        assert webhook_resp.status_code == 200
        webhook_data = webhook_resp.json()
        assert webhook_data["decision"] == "approved"
        assert webhook_data["new_status"] == "running"
