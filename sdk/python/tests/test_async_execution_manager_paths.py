import asyncio
from types import SimpleNamespace
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import pytest

from agentfield.async_config import AsyncConfig
from agentfield.async_execution_manager import AsyncExecutionManager
from agentfield.execution_state import ExecutionState, ExecutionStatus


class _DummyResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    async def json(self):
        await asyncio.sleep(0)
        return self._payload


@pytest.mark.asyncio
async def test_poll_single_execution_targets_canonical_endpoint():
    cfg = AsyncConfig(enable_async_execution=True, enable_batch_polling=False)
    manager = AsyncExecutionManager("http://example", cfg)
    execution = ExecutionState(
        execution_id="exec-single", target="node.skill", input_data={}
    )

    request_mock = AsyncMock(return_value=_DummyResponse({"status": "succeeded"}))
    manager.connection_manager = SimpleNamespace(
        request=request_mock, batch_request=AsyncMock()
    )

    async def noop_process(self, exec_state, response, duration):
        return None

    manager._process_poll_response = noop_process.__get__(
        manager, AsyncExecutionManager
    )

    await manager._poll_single_execution(execution)

    assert request_mock.await_count == 1
    call = request_mock.await_args
    assert call.args[0] == "GET"
    assert call.args[1].endswith(f"/api/v1/executions/{execution.execution_id}")
    assert call.kwargs["timeout"] == cfg.polling_timeout


@pytest.mark.asyncio
async def test_batch_poll_uses_canonical_endpoint():
    cfg = AsyncConfig(enable_async_execution=True, enable_batch_polling=True)
    cfg.batch_size = 5
    manager = AsyncExecutionManager("http://example", cfg)

    executions = [
        ExecutionState(execution_id=f"exec-{idx}", target="node.skill", input_data={})
        for idx in range(3)
    ]

    batch_mock = AsyncMock(
        return_value=[_DummyResponse({"status": "succeeded"}) for _ in executions]
    )
    manager.connection_manager = SimpleNamespace(
        request=AsyncMock(), batch_request=batch_mock
    )

    async def noop_process(self, exec_state, response, duration):
        return None

    manager._process_poll_response = noop_process.__get__(
        manager, AsyncExecutionManager
    )

    await manager._batch_poll_executions(executions)

    assert batch_mock.await_count == 1
    call = batch_mock.await_args
    batch_requests = call.args[0]
    assert len(batch_requests) == len(executions)
    for exec_state, request in zip(executions, batch_requests):
        assert request["method"] == "GET"
        assert request["url"].endswith(f"/api/v1/executions/{exec_state.execution_id}")
        assert request["timeout"] == cfg.polling_timeout


@pytest.mark.asyncio
async def test_submit_execution_wraps_payload(monkeypatch):
    cfg = AsyncConfig(enable_async_execution=True)
    manager = AsyncExecutionManager("http://example", cfg)

    # Pretend the manager is already running
    manager._polling_task = object()

    # Provide a dummy execution lock context (already initialized)
    session_post = AsyncMock(
        return_value=_DummyResponse(
            {
                "execution_id": "exec-xyz",
                "status": "queued",
            }
        )
    )

    class DummySession:
        post = session_post

    class DummyConnectionManager:
        def __init__(self, session):
            self._session = session

        @asynccontextmanager
        async def get_session(self):
            yield self._session

    manager.connection_manager = DummyConnectionManager(DummySession())

    execution_id = await manager.submit_execution(
        target="node.reasoner",
        input_data={"foo": "bar"},
    )

    assert execution_id == "exec-xyz"

    assert session_post.await_count == 1
    call = session_post.await_args
    assert call.args[0] == "http://example/api/v1/execute/async/node.reasoner"
    assert call.kwargs["json"] == {"input": {"foo": "bar"}}


@pytest.mark.asyncio
async def test_submit_execution_ignores_completed_entries_for_capacity():
    cfg = AsyncConfig(enable_async_execution=True, max_concurrent_executions=2)
    manager = AsyncExecutionManager("http://example", cfg)

    # Pretend the manager is already running
    manager._polling_task = object()

    session_post = AsyncMock(
        return_value=_DummyResponse(
            {
                "execution_id": "exec-cap",
                "status": "queued",
            }
        )
    )

    class DummySession:
        post = session_post

    class DummyConnectionManager:
        def __init__(self, session):
            self._session = session

        @asynccontextmanager
        async def get_session(self):
            yield self._session

    manager.connection_manager = DummyConnectionManager(DummySession())

    # Pre-fill the execution map with completed executions beyond the configured limit.
    async with manager._execution_lock:
        for idx in range(3):
            exec_state = ExecutionState(
                execution_id=f"exec-done-{idx}",
                target="node.skill",
                input_data={},
            )
            exec_state.update_status(ExecutionStatus.SUCCEEDED)
            manager._executions[exec_state.execution_id] = exec_state

        manager.metrics.total_executions = len(manager._executions)
        manager.metrics.completed_executions = len(manager._executions)
        manager.metrics.active_executions = 0

    # Should still accept a new submission because there are no active executions.
    execution_id = await manager.submit_execution(
        target="node.reasoner",
        input_data={"foo": "bar"},
    )

    assert execution_id == "exec-cap"
    assert session_post.await_count == 1


@pytest.mark.asyncio
async def test_update_execution_from_status_normalizes_whitespace():
    cfg = AsyncConfig(enable_async_execution=True)
    manager = AsyncExecutionManager("http://example", cfg)

    execution = ExecutionState(
        execution_id="exec-status",
        target="node.skill",
        input_data={},
    )

    # Inject execution into manager cache so metrics updates succeed
    async with manager._execution_lock:
        manager._executions[execution.execution_id] = execution

    status_payload = {
        "status": " succeeded \n",
        "result": {"ok": True},
    }

    await manager._update_execution_from_status(execution, status_payload)

    assert execution.status == ExecutionStatus.SUCCEEDED
    assert execution.result == {"ok": True}
