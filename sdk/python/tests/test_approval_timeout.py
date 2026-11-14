"""Approval polling deadlines through the public client API."""

import asyncio
from unittest.mock import AsyncMock

import httpx
import pytest
import respx

from agentfield.client import AgentFieldClient
from agentfield.exceptions import ExecutionTimeoutError


@pytest.fixture
def router():
    return respx.MockRouter(assert_all_mocked=True)


@pytest.fixture
async def client(router, monkeypatch):
    client = AgentFieldClient(base_url="https://control-plane.test")
    client.caller_agent_id = "test-node"
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(router.async_handler)
    ) as http_client:
        monkeypatch.setattr(
            client, "get_async_http_client", AsyncMock(return_value=http_client)
        )
        yield client
    await client.aclose()


@pytest.mark.asyncio
async def test_timeout_interrupts_initial_poll_delay(client, router):
    route = router.get(
        "https://control-plane.test/api/v1/agents/test-node/executions/exec-1/approval-status"
    ).respond(200, json={"status": "approved"})

    with pytest.raises(ExecutionTimeoutError, match="exec-1 timed out after 0.01s"):
        await asyncio.wait_for(
            client.wait_for_approval("exec-1", poll_interval=1, timeout=0.01),
            timeout=0.5,
        )

    assert not route.called


@pytest.mark.asyncio
async def test_timeout_cancels_in_flight_status_request(client, router):
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def slow_response(request):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            await asyncio.sleep(0)
            cancelled.set()

    router.get(
        "https://control-plane.test/api/v1/agents/test-node/executions/exec-1/approval-status"
    ).mock(side_effect=slow_response)

    with pytest.raises(ExecutionTimeoutError, match="timed out"):
        await asyncio.wait_for(
            client.wait_for_approval("exec-1", poll_interval=0, timeout=0.05),
            timeout=0.5,
        )

    assert started.is_set()
    assert cancelled.is_set()


@pytest.mark.asyncio
@pytest.mark.parametrize("first_status", [200, 503])
async def test_timeout_interrupts_pending_or_error_backoff(
    client, router, first_status
):
    route = router.get(
        "https://control-plane.test/api/v1/agents/test-node/executions/exec-1/approval-status"
    ).mock(
        side_effect=[
            httpx.Response(first_status, json={"status": "pending"}),
            httpx.Response(200, json={"status": "approved"}),
        ]
    )

    with pytest.raises(ExecutionTimeoutError, match="timed out"):
        await asyncio.wait_for(
            client.wait_for_approval("exec-1", poll_interval=0.05, timeout=0.08),
            timeout=0.5,
        )

    assert route.call_count <= 1


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout", [0, -1])
async def test_elapsed_deadline_does_not_poll(client, router, timeout):
    route = router.get(
        "https://control-plane.test/api/v1/agents/test-node/executions/exec-1/approval-status"
    ).respond(200, json={"status": "approved"})

    with pytest.raises(ExecutionTimeoutError, match="timed out"):
        await client.wait_for_approval("exec-1", poll_interval=0, timeout=timeout)

    assert not route.called


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout", [None, 1])
@pytest.mark.parametrize(
    "status", ["approved", "rejected", "expired", "request_changes"]
)
async def test_resolved_status_before_deadline_is_unchanged(
    client, router, timeout, status
):
    route = router.get(
        "https://control-plane.test/api/v1/agents/test-node/executions/exec-1/approval-status"
    ).respond(200, json={"status": status, "feedback": "human response"})

    result = await asyncio.wait_for(
        client.wait_for_approval("exec-1", poll_interval=0, timeout=timeout),
        timeout=0.5,
    )

    assert result.status == status
    assert result.feedback == "human response"
    assert route.call_count == 1


@pytest.mark.asyncio
async def test_caller_cancellation_cleans_up_status_request(client, router):
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def slow_response(request):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    router.get(
        "https://control-plane.test/api/v1/agents/test-node/executions/exec-1/approval-status"
    ).mock(side_effect=slow_response)
    task = asyncio.create_task(
        client.wait_for_approval("exec-1", poll_interval=0, timeout=10)
    )
    try:
        await asyncio.wait_for(started.wait(), timeout=0.5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert cancelled.is_set()
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_caller_cancellation_wins_when_status_resolves(client, router):
    waiting = asyncio.current_task()

    async def approved_response(request):
        assert waiting.cancel()
        return httpx.Response(200, json={"status": "approved"})

    route = router.get(
        "https://control-plane.test/api/v1/agents/test-node/executions/exec-1/approval-status"
    ).mock(side_effect=approved_response)

    with pytest.raises(asyncio.CancelledError):
        await client.wait_for_approval("exec-1", poll_interval=0, timeout=1)

    assert route.call_count == 1


@pytest.mark.asyncio
async def test_repeated_cancellation_does_not_leave_a_status_request(client, router):
    started = asyncio.Event()
    cleanup_started = asyncio.Event()
    cleanup_finished = asyncio.Event()
    finish_cleanup = asyncio.Event()

    async def slow_response(request):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleanup_started.set()
            try:
                await finish_cleanup.wait()
            finally:
                cleanup_finished.set()

    router.get(
        "https://control-plane.test/api/v1/agents/test-node/executions/exec-1/approval-status"
    ).mock(side_effect=slow_response)
    task = asyncio.create_task(
        client.wait_for_approval("exec-1", poll_interval=0, timeout=10)
    )
    try:
        await asyncio.wait_for(started.wait(), timeout=0.5)
        task.cancel()
        await asyncio.wait_for(cleanup_started.wait(), timeout=0.5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=0.5)
        assert cleanup_finished.is_set()
    finally:
        finish_cleanup.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
