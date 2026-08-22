import asyncio
from contextlib import suppress

import pytest

from agentfield.async_config import AsyncConfig
from agentfield.async_execution_manager import AsyncExecutionManager
from agentfield.execution_state import ExecutionState, ExecutionStatus
from agentfield.http_connection_manager import ConnectionManager
from agentfield.result_cache import ResultCache


async def _failing_on_cancel(started: asyncio.Event, failure: RuntimeError) -> None:
    started.set()
    try:
        await asyncio.Future()
    except asyncio.CancelledError:
        raise failure


async def _wait_forever(started: asyncio.Event) -> None:
    started.set()
    await asyncio.Future()


async def _cancel_task(task: asyncio.Task | None) -> None:
    if task is not None and not task.done():
        task.cancel()
        with suppress(asyncio.CancelledError, RuntimeError):
            await task


@pytest.mark.asyncio
async def test_connection_manager_close_finishes_after_task_cleanup_error():
    manager = ConnectionManager(AsyncConfig(enable_performance_logging=False))
    await manager.start()

    failure = RuntimeError("health cleanup failed")
    failing_started = asyncio.Event()
    remaining_started = asyncio.Event()
    manager._health_check_task = asyncio.create_task(
        _failing_on_cancel(failing_started, failure)
    )
    manager._cleanup_task = asyncio.create_task(_wait_forever(remaining_started))
    await asyncio.gather(failing_started.wait(), remaining_started.wait())

    try:
        with pytest.raises(RuntimeError) as raised:
            await manager.close()
        assert raised.value is failure

        assert manager._closed is True
        assert manager._health_check_task is None
        assert manager._cleanup_task is None
        assert manager._session is None
        assert manager._connector is None
        assert manager._loop is None
    finally:
        await _cancel_task(manager._health_check_task)
        await _cancel_task(manager._cleanup_task)
        if manager._session is not None:
            await manager._session.close()
        if manager._connector is not None:
            await manager._connector.close()


@pytest.mark.asyncio
async def test_async_execution_manager_stop_finishes_after_task_cleanup_error():
    config = AsyncConfig(
        enable_async_execution=True,
        enable_result_caching=True,
        enable_performance_logging=False,
        enable_event_stream=False,
    )
    manager = AsyncExecutionManager("http://example", config)
    await manager.start()

    # Replace the real polling task with a task whose cancellation cleanup fails.
    original_polling_task = manager._polling_task
    await _cancel_task(original_polling_task)
    failure = RuntimeError("polling cleanup failed")
    failing_started = asyncio.Event()
    manager._polling_task = asyncio.create_task(
        _failing_on_cancel(failing_started, failure)
    )
    await failing_started.wait()

    execution = ExecutionState("active-execution", "node.skill", {})
    async with manager._execution_lock:
        manager._executions[execution.execution_id] = execution
        manager.metrics.active_executions = 1
    manager.result_cache.set("cached", {"ok": True})

    try:
        with pytest.raises(RuntimeError) as raised:
            await manager.stop()
        assert raised.value is failure

        assert execution.status == ExecutionStatus.CANCELLED
        assert manager._polling_task is None
        assert manager._cleanup_task is None
        assert manager._metrics_task is None
        assert manager._event_stream_task is None
        assert manager._loop is None
        assert manager.connection_manager._closed is True
        assert manager.connection_manager._session is None
        assert manager.result_cache.get("cached") is None
        assert manager.result_cache._cleanup_task is None
        assert manager.result_cache._shutdown_event is None
        assert manager.result_cache._loop is None
        assert manager.result_cache.metrics.size == 0
    finally:
        await _cancel_task(manager._polling_task)
        await _cancel_task(manager._cleanup_task)
        await _cancel_task(manager._metrics_task)
        await _cancel_task(manager._event_stream_task)
        if manager.connection_manager._session is not None:
            await manager.connection_manager._session.close()
        if manager.connection_manager._connector is not None:
            await manager.connection_manager._connector.close()
        await _cancel_task(manager.result_cache._cleanup_task)


@pytest.mark.asyncio
async def test_result_cache_stop_finishes_after_task_cleanup_error():
    cache = ResultCache(
        AsyncConfig(
            enable_result_caching=True,
            cleanup_interval=60.0,
        )
    )
    await cache.start()

    original_cleanup_task = cache._cleanup_task
    await _cancel_task(original_cleanup_task)
    failure = RuntimeError("cache cleanup failed")
    failing_started = asyncio.Event()
    cache._cleanup_task = asyncio.create_task(
        _failing_on_cancel(failing_started, failure)
    )
    await failing_started.wait()
    cache.set("cached", {"ok": True})

    try:
        with pytest.raises(RuntimeError) as raised:
            await cache.stop()
        assert raised.value is failure

        assert cache._cleanup_task is None
        assert cache._shutdown_event is None
        assert cache._loop is None
        assert cache.get("cached") is None
        assert cache.metrics.size == 0
    finally:
        await _cancel_task(cache._cleanup_task)
