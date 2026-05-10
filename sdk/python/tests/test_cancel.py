"""Tests for the cooperative-cancel transport on the Python SDK.

The full Agent class drags in FastAPI middleware, AI clients, memory
backends, and the registration loop — so we use a lightweight stand-in
for unit-testing the cancel module in isolation. The contract tested is:

  * register_execution_task / deregister_execution / cancel_execution
    behave correctly under concurrency.
  * cancel_execution raises CancelledError into the registered task.
  * The HTTP route returns 200 with `cancelled: True/False` per state.
"""

from __future__ import annotations

import asyncio
import logging

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agentfield.cancel import (
    cancel_execution,
    deregister_execution,
    install_cancel_route,
    is_execution_cancelled,
    register_execution_task,
)


class _FakeAgent(FastAPI):
    """Stand-in for Agent that satisfies the cancel module's contract.

    The cancel module only touches three attributes (`_cancel_tasks`,
    `_cancel_lock`) and the FastAPI `.post(...)` decorator. Subclassing
    FastAPI gives us the route-registration plumbing for free.
    """

    def __init__(self) -> None:
        super().__init__()
        self._cancel_tasks: dict[str, asyncio.Task] = {}
        self._cancel_lock = asyncio.Lock()


@pytest.mark.asyncio
async def test_cancel_execution_raises_into_task():
    agent = _FakeAgent()
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def long_running() -> None:
        started.set()
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    task = asyncio.create_task(long_running())
    await register_execution_task(agent, "exec-1", task)
    await started.wait()

    assert await cancel_execution(agent, "exec-1") is True
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_cancel_execution_unknown_id_returns_false():
    agent = _FakeAgent()
    assert await cancel_execution(agent, "missing") is False
    assert await cancel_execution(agent, "") is False


@pytest.mark.asyncio
async def test_deregister_execution_removes_entry():
    agent = _FakeAgent()
    task = asyncio.create_task(asyncio.sleep(0.01))
    await register_execution_task(agent, "exec-2", task)
    await deregister_execution(agent, "exec-2")
    assert "exec-2" not in agent._cancel_tasks
    assert await cancel_execution(agent, "exec-2") is False
    await task  # let the sleep complete cleanly


@pytest.mark.asyncio
async def test_register_replaces_existing_id():
    agent = _FakeAgent()
    first = asyncio.create_task(asyncio.sleep(10))
    second = asyncio.create_task(asyncio.sleep(10))
    await register_execution_task(agent, "exec-3", first)
    await register_execution_task(agent, "exec-3", second)
    assert agent._cancel_tasks["exec-3"] is second

    # Cancelling cleans up the second task; the first is still running.
    assert await cancel_execution(agent, "exec-3") is True
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    with pytest.raises(asyncio.CancelledError):
        await second


@pytest.mark.asyncio
async def test_concurrent_register_and_cancel_settles_clean():
    agent = _FakeAgent()
    n = 20
    tasks = [asyncio.create_task(asyncio.sleep(10)) for _ in range(n)]
    register_jobs = [
        register_execution_task(agent, f"exec-{i}", tasks[i]) for i in range(n)
    ]
    await asyncio.gather(*register_jobs)

    cancel_jobs = [cancel_execution(agent, f"exec-{i}") for i in range(n)]
    results = await asyncio.gather(*cancel_jobs)
    assert all(results)

    for task in tasks:
        with pytest.raises(asyncio.CancelledError):
            await task


def test_cancel_route_returns_200_for_unknown_execution():
    agent = _FakeAgent()
    install_cancel_route(agent)
    client = TestClient(agent)
    resp = client.post("/_internal/executions/exec-missing/cancel")
    assert resp.status_code == 200
    body = resp.json()
    assert body == {
        "cancelled": False,
        "execution_id": "exec-missing",
        "reason": "execution_not_active",
    }


@pytest.mark.asyncio
async def test_cancel_route_emits_info_log_on_active_cancel(caplog):
    """The route logs a structured info line when it actually cancelled
    something. We register the task and call the route handler directly
    rather than via TestClient because TestClient creates a fresh event
    loop per request, so a task registered in one request isn't visible
    from the next.
    """
    agent = _FakeAgent()
    install_cancel_route(agent)

    async def long() -> None:
        await asyncio.sleep(10)

    task = asyncio.create_task(long())
    await register_execution_task(agent, "exec-active", task)

    # Find the route handler we just installed.
    handler = None
    for route in agent.router.routes:
        if (
            getattr(route, "path", "")
            == "/_internal/executions/{execution_id}/cancel"
        ):
            handler = route.endpoint
            break
    assert handler is not None, "cancel route was not installed"

    # Build a minimal request stand-in. The handler only reads one header.
    class _Headers:
        def __init__(self, h):
            self._h = h

        def get(self, name, default=None):
            return self._h.get(name, default)

    class _Request:
        def __init__(self, headers):
            self.headers = _Headers(headers)

    with caplog.at_level(logging.INFO, logger="agentfield.cancel"):
        resp = await handler(
            execution_id="exec-active",
            request=_Request({"X-AgentField-Source": "cancel-dispatcher"}),
        )

    # FastAPI JSONResponse — body is bytes, status is on the object.
    import json as _json

    assert resp.status_code == 200
    body = _json.loads(resp.body.decode())
    assert body == {"cancelled": True, "execution_id": "exec-active"}
    assert any("cancel-callback fired" in rec.message for rec in caplog.records)

    with pytest.raises(asyncio.CancelledError):
        await task


def test_cancel_route_rejects_path_with_slash():
    agent = _FakeAgent()
    install_cancel_route(agent)
    client = TestClient(agent)
    # FastAPI treats this as a missing route (path doesn't match the
    # parameterized handler), so 404 from the framework rather than from
    # our explicit guard. Either way the user-visible behaviour is "no
    # match".
    resp = client.post("/_internal/executions/has/extra/cancel")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_is_execution_cancelled_helper():
    agent = _FakeAgent()
    assert is_execution_cancelled(agent, None) is False
    assert is_execution_cancelled(agent, "missing") is False

    async def loop():
        try:
            while True:
                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            raise

    task = asyncio.create_task(loop())
    await register_execution_task(agent, "exec-poll", task)
    assert is_execution_cancelled(agent, "exec-poll") is False

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    # After natural cancellation the helper reflects it.
    assert is_execution_cancelled(agent, "exec-poll") is True
