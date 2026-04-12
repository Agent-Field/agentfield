from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from agentfield.async_config import AsyncConfig
from agentfield.async_execution_manager import AsyncExecutionManager
from agentfield.execution_state import ExecutionState, ExecutionStatus


class _DummyTask:
    def __init__(self):
        self.cancelled = False

    def cancel(self):
        self.cancelled = True

    def done(self):
        return False

    def __await__(self):
        async def _wait():
            raise asyncio.CancelledError

        return _wait().__await__()


class _Response:
    def __init__(self, payload=None):
        self.payload = payload or {}

    def raise_for_status(self):
        return None

    async def json(self):
        return self.payload


@pytest.fixture
def manager():
    cfg = AsyncConfig(enable_async_execution=True)
    cfg.completed_execution_retention_seconds = 1
    cfg.max_completed_executions = 1
    mgr = AsyncExecutionManager("http://example", cfg)
    mgr.connection_manager = SimpleNamespace(
        start=AsyncMock(),
        close=AsyncMock(),
        request=AsyncMock(),
        batch_request=AsyncMock(),
        get_metrics=lambda: SimpleNamespace(requests=1),
    )
    mgr.result_cache = SimpleNamespace(
        start=AsyncMock(),
        stop=AsyncMock(),
        get_execution_result=Mock(return_value=None),
        set_execution_result=Mock(),
        get_stats=Mock(return_value={"entries": 0}),
    )
    return mgr


@pytest.mark.asyncio
async def test_start_stop_and_event_stream_headers(manager, monkeypatch):
    created_tasks = []

    def fake_create_task(coro):
        coro.close()
        task = _DummyTask()
        created_tasks.append(task)
        return task

    monkeypatch.setattr("agentfield.async_execution_manager.asyncio.create_task", fake_create_task)

    manager.config.enable_performance_logging = True
    manager.config.enable_event_stream = True
    manager.set_event_stream_headers({"Authorization": "Bearer token", "X-Null": None})
    assert manager._event_stream_headers == {"Authorization": "Bearer token"}

    await manager.start()
    assert len(created_tasks) == 4
    manager.connection_manager.start.assert_awaited_once()
    manager.result_cache.start.assert_awaited_once()

    execution = ExecutionState("exec-1", "node.skill", {})
    async with manager._execution_lock:
        manager._executions["exec-1"] = execution
    await manager.stop()

    assert execution.status == ExecutionStatus.CANCELLED
    manager.connection_manager.close.assert_awaited_once()
    manager.result_cache.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_wait_for_result_handles_success_failure_cancel_timeout(manager, monkeypatch):
    success = ExecutionState("success", "node.skill", {})
    success.set_result({"ok": True})

    failed = ExecutionState("failed", "node.skill", {})
    failed.set_error("boom", {"detail": True})

    cancelled = ExecutionState("cancelled", "node.skill", {})
    cancelled.cancel("user")

    pending = ExecutionState("pending", "node.skill", {}, timeout=0.01)

    async with manager._execution_lock:
        manager._executions.update(
            {
                "success": success,
                "failed": failed,
                "cancelled": cancelled,
                "pending": pending,
            }
        )

    assert await manager.wait_for_result("success") == {"ok": True}
    manager.result_cache.set_execution_result.assert_called_with("success", {"ok": True})

    with pytest.raises(RuntimeError, match="boom"):
        await manager.wait_for_result("failed")

    with pytest.raises(RuntimeError, match="cancelled"):
        await manager.wait_for_result("cancelled")

    async def fast_sleep(_seconds):
        return None

    tick = {"count": 0}

    def fake_time():
        tick["count"] += 1
        return 1000.0 + tick["count"] * 0.02

    monkeypatch.setattr("agentfield.async_execution_manager.asyncio.sleep", fast_sleep)
    monkeypatch.setattr("agentfield.async_execution_manager.time.time", fake_time)

    with pytest.raises(TimeoutError, match="Wait timeout reached"):
        await manager.wait_for_result("pending", timeout=0.01)

    assert pending.status == ExecutionStatus.TIMEOUT


@pytest.mark.asyncio
async def test_cleanup_polling_and_event_payload_paths(manager, monkeypatch):
    old_success = ExecutionState("old-success", "node.skill", {})
    old_success.set_result({"old": True})
    old_success.metrics.end_time = time.time() - 5

    recent_success = ExecutionState("recent-success", "node.skill", {})
    recent_success.set_result({"recent": True})
    recent_success.metrics.end_time = time.time()

    overflow = ExecutionState("overflow", "node.skill", {})
    overflow.set_result({"overflow": True})
    overflow.metrics.end_time = time.time() - 2

    running = ExecutionState("running", "node.skill", {})
    running.update_status(ExecutionStatus.RUNNING)
    running._last_poll_time = time.time() - 100

    overdue = ExecutionState("overdue", "node.skill", {}, timeout=0.01)
    overdue._last_poll_time = time.time() - 100
    overdue.metrics.submit_time = time.time() - 100

    async with manager._execution_lock:
        manager._executions = {
            "old-success": old_success,
            "recent-success": recent_success,
            "overflow": overflow,
            "running": running,
            "overdue": overdue,
        }
        manager.metrics.active_executions = 2

    cleaned = await manager.cleanup_completed_executions()
    assert cleaned == 2
    assert "recent-success" in manager._executions
    assert "old-success" not in manager._executions
    assert "overflow" not in manager._executions
    assert manager.result_cache.set_execution_result.call_count >= 2

    batch_polled = AsyncMock()
    individual_polled = AsyncMock()
    manager._batch_poll_executions = batch_polled
    manager._individual_poll_executions = individual_polled

    await manager._poll_active_executions()
    batch_polled.assert_not_awaited()
    individual_polled.assert_awaited_once()
    assert overdue.status == ExecutionStatus.TIMEOUT

    manager._poll_single_execution = AsyncMock()
    create_task_calls = []

    def fake_create_task(coro):
        create_task_calls.append(coro)
        coro.close()
        return _DummyTask()

    monkeypatch.setattr("agentfield.async_execution_manager.asyncio.create_task", fake_create_task)

    await manager._handle_event_stream_payload({"execution_id": "running", "status": "running"})
    assert running.status == ExecutionStatus.RUNNING

    await manager._handle_event_stream_payload({"executionId": "running", "status": "succeeded"})
    assert running.status == ExecutionStatus.SUCCEEDED
    assert create_task_calls


@pytest.mark.asyncio
async def test_process_poll_response_metrics_and_circuit_breaker(manager, monkeypatch):
    execution = ExecutionState("exec-1", "node.skill", {})
    update_mock = AsyncMock()
    manager._update_execution_from_status = update_mock

    old_interval = execution.current_poll_interval
    await manager._process_poll_response(execution, asyncio.TimeoutError(), 0.5)
    assert manager.metrics.polling_metrics.timeout_polls == 1
    assert execution.current_poll_interval > old_interval

    await manager._process_poll_response(execution, _Response({"status": "running"}), 0.2)
    update_mock.assert_awaited_once()
    assert manager.metrics.polling_metrics.successful_polls == 1

    for _ in range(manager.config.circuit_breaker_failure_threshold):
        manager._record_circuit_breaker_failure()
    assert manager._circuit_breaker_open is True

    monkeypatch.setattr(
        "agentfield.async_execution_manager.time.time",
        lambda: manager._circuit_breaker_last_failure
        + manager.config.circuit_breaker_recovery_timeout
        + 1,
    )
    assert manager._is_circuit_breaker_open() is False
    assert "success_rate" in repr(manager)
