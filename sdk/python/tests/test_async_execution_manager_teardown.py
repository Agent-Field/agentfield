"""AsyncExecutionManager.stop() must finish teardown even when a cancel raises.

stop() wraps its two cancel sweeps — the four background tasks, then the
active executions — in a single try each. One raising cancel therefore aborted
the whole sweep while the surrounding ``finally`` went on to drop the last
reference to every task, leaving the ones behind it pending and unreachable
and their executions stuck QUEUED with their capacity slots held.

These tests pin the sweeps down by their observable effect: every task is done
and every other execution is CANCELLED once stop() returns, whatever the first
cancel did. Companion to test_async_lifecycle_deadlock.py (#623) and the
finally-based structure from #909 — the cleanup below the sweeps must still
run, including for a BaseException.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agentfield.async_config import AsyncConfig
from agentfield.async_execution_manager import AsyncExecutionManager
from agentfield.execution_state import ExecutionState, ExecutionStatus


class _Abort(BaseException):
    """A BaseException that is not CancelledError."""


def _manager() -> AsyncExecutionManager:
    manager = AsyncExecutionManager(
        "http://example", AsyncConfig(enable_async_execution=True)
    )
    manager.connection_manager = SimpleNamespace(start=AsyncMock(), close=AsyncMock())
    manager.result_cache = SimpleNamespace(start=AsyncMock(), stop=AsyncMock())
    manager._loop = asyncio.get_running_loop()
    manager._shutdown_event = asyncio.Event()
    return manager


async def _idle() -> None:
    await asyncio.sleep(3600)


async def _explodes_on_cancel() -> None:
    try:
        await asyncio.sleep(3600)
    except asyncio.CancelledError:
        raise RuntimeError("cancel handler exploded")


async def _attach_background_tasks(manager, coros):
    tasks = [asyncio.create_task(coro) for coro in coros]
    await asyncio.sleep(0)  # let each task reach its first await
    (
        manager._polling_task,
        manager._cleanup_task,
        manager._metrics_task,
        manager._event_stream_task,
    ) = tasks
    return tasks


def _execution(execution_id: str, status: ExecutionStatus) -> ExecutionState:
    execution = ExecutionState(
        execution_id=execution_id, target="node.skill", input_data={}
    )
    execution.update_status(status)
    return execution


# ---------------------------------------------------------------------------
# Background-task sweep
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_stop_cancels_every_background_task_when_one_cancel_raises():
    manager = _manager()
    tasks = await _attach_background_tasks(
        manager, [_explodes_on_cancel(), _idle(), _idle(), _idle()]
    )

    with pytest.raises(RuntimeError, match="cancel handler exploded"):
        await manager.stop()

    assert all(task.done() for task in tasks), (
        "a raising cancel left the tasks behind it pending and unreachable"
    )
    manager.connection_manager.close.assert_awaited_once()
    manager.result_cache.stop.assert_awaited_once()


@pytest.mark.unit
async def test_stop_reports_the_first_task_cancel_failure():
    """Later failures must not mask the one that happened first."""
    manager = _manager()

    async def _explodes_differently():
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            raise ValueError("second failure")

    tasks = await _attach_background_tasks(
        manager, [_explodes_on_cancel(), _explodes_differently(), _idle(), _idle()]
    )

    with pytest.raises(RuntimeError, match="cancel handler exploded"):
        await manager.stop()

    assert all(task.done() for task in tasks)


@pytest.mark.unit
async def test_stop_without_failures_still_tears_everything_down():
    manager = _manager()
    tasks = await _attach_background_tasks(
        manager, [_idle(), _idle(), _idle(), _idle()]
    )

    await manager.stop()

    assert all(task.cancelled() for task in tasks)
    assert manager._polling_task is None
    assert manager._loop is None
    manager.connection_manager.close.assert_awaited_once()
    manager.result_cache.stop.assert_awaited_once()


# ---------------------------------------------------------------------------
# Execution sweep
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_stop_cancels_remaining_executions_when_one_cancel_raises():
    manager = _manager()

    stubborn = _execution("exec-stubborn", ExecutionStatus.RUNNING)

    def _refuse(reason=None):
        raise RuntimeError("cancel refused")

    stubborn.cancel = _refuse

    healthy = _execution("exec-healthy", ExecutionStatus.QUEUED)
    manager._executions = {
        "exec-stubborn": stubborn,
        "exec-healthy": healthy,
    }

    with pytest.raises(RuntimeError, match="cancel refused"):
        await manager.stop()

    assert healthy.status == ExecutionStatus.CANCELLED, (
        "a raising cancel left the executions behind it active"
    )
    manager.connection_manager.close.assert_awaited_once()
    manager.result_cache.stop.assert_awaited_once()


@pytest.mark.unit
async def test_stop_finishes_the_sweep_for_a_base_exception():
    """#909: cleanup must not be skipped when a cancel raises a BaseException."""
    manager = _manager()

    stubborn = _execution("exec-stubborn", ExecutionStatus.RUNNING)

    def _abort(reason=None):
        raise _Abort("not an Exception")

    stubborn.cancel = _abort

    healthy = _execution("exec-healthy", ExecutionStatus.QUEUED)
    manager._executions = {
        "exec-stubborn": stubborn,
        "exec-healthy": healthy,
    }

    with pytest.raises(_Abort):
        await manager.stop()

    assert healthy.status == ExecutionStatus.CANCELLED
    manager.connection_manager.close.assert_awaited_once()
    manager.result_cache.stop.assert_awaited_once()


@pytest.mark.unit
async def test_stop_leaves_terminal_executions_untouched():
    manager = _manager()
    finished = _execution("exec-done", ExecutionStatus.SUCCEEDED)
    manager._executions = {"exec-done": finished}

    await manager.stop()

    assert finished.status == ExecutionStatus.SUCCEEDED


@pytest.mark.unit
async def test_a_failing_task_cancel_does_not_skip_the_execution_sweep():
    """The two sweeps are independent: the first must not abort the second."""
    manager = _manager()
    tasks = await _attach_background_tasks(
        manager, [_explodes_on_cancel(), _idle(), _idle(), _idle()]
    )
    active = _execution("exec-active", ExecutionStatus.RUNNING)
    manager._executions = {"exec-active": active}

    with pytest.raises(RuntimeError, match="cancel handler exploded"):
        await manager.stop()

    assert all(task.done() for task in tasks)
    assert active.status == ExecutionStatus.CANCELLED
