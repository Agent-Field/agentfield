"""Regression tests for cross-loop teardown of async lifecycle components (#623).

Companion to test_result_cache_deadlock.py. The same loop-bound-primitive
hazard that affected ResultCache also lived in two sibling components that
sit above it in client.aclose() -> AsyncExecutionManager.stop() ->
ConnectionManager.close():

- AsyncExecutionManager.stop() cancelled + awaited four background tasks
  and set an asyncio.Event, all bound to the loop start() ran on.
- http_connection_manager.ConnectionManager.close() took an asyncio.Lock and
  awaited its background tasks + aiohttp session, all loop-bound.

Starting either on one loop and tearing it down from another raised
"got Future attached to a different loop" (or hung), so the end-to-end
sync/async mixing case (#620/#623) still failed one level up from the cache.

These tests start each component on a background thread's loop and tear it
down from a different loop, asserting no deadlock / RuntimeError and that the
component is left in a clean, closed state.
"""

import asyncio
import threading
import time

import pytest

from agentfield.async_config import AsyncConfig
from agentfield.async_execution_manager import AsyncExecutionManager
from agentfield.async_lifecycle import cancel_and_await_if_same_loop
from agentfield.http_connection_manager import ConnectionManager


def _run_loop_in_thread():
    """Spin up an event loop in a daemon thread and return (loop, thread)."""
    loop = asyncio.new_event_loop()

    def run():
        asyncio.set_event_loop(loop)
        loop.run_forever()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return loop, thread


def _shutdown_loop(loop, thread):
    # Give the owning loop a moment to process any scheduled cross-loop
    # cancels before stopping it.
    time.sleep(0.2)
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=2)
    if thread.is_alive():
        # Thread didn't stop in time — don't close the loop while it's
        # still running (would raise RuntimeError and obscure the real failure).
        return
    if not loop.is_closed():
        loop.close()


def test_connection_manager_cross_loop_close_no_deadlock():
    """start() on loop1, close() from loop2 must not hang or raise."""
    cfg = AsyncConfig()
    # Force the background health/cleanup tasks to start so there is
    # something loop-bound to tear down.
    cfg.enable_performance_logging = True
    cm = ConnectionManager(cfg)
    session_closed = threading.Event()

    loop1, thread = _run_loop_in_thread()
    try:
        asyncio.run_coroutine_threadsafe(cm.start(), loop1).result(timeout=5)
        assert cm._session is not None

        original_session_close = cm._session.close

        async def close_session():
            try:
                await original_session_close()
            finally:
                session_closed.set()

        cm._session.close = close_session
        time.sleep(0.15)

        async def close_here():
            await asyncio.wait_for(cm.close(), timeout=5.0)

        asyncio.run(close_here())

        assert session_closed.wait(timeout=5)
        # The owner loop completes the whole teardown before state is cleared.
        assert cm._closed is True
        assert cm._health_check_task is None
        assert cm._cleanup_task is None
        assert cm._session is None
    finally:
        _shutdown_loop(loop1, thread)


def test_connection_manager_cross_loop_close_is_idempotent_while_owner_lock_is_held():
    """A repeated foreign-loop close must not acquire the owner-loop lock."""
    cm = ConnectionManager()
    loop1, thread = _run_loop_in_thread()
    lock_held = threading.Event()
    release_lock = asyncio.Event()
    session_closed = threading.Event()
    holder = None

    async def hold_owner_lock():
        async with cm._lock:
            lock_held.set()
            await release_lock.wait()

    try:
        asyncio.run_coroutine_threadsafe(cm.start(), loop1).result(timeout=5)
        assert cm._session is not None
        original_session_close = cm._session.close

        async def close_session():
            try:
                await original_session_close()
            finally:
                session_closed.set()

        cm._session.close = close_session
        holder = asyncio.run_coroutine_threadsafe(hold_owner_lock(), loop1)
        assert lock_held.wait(timeout=5)

        asyncio.run(cm.close())
        # Before the fix this second call fell through to ``async with
        # self._lock`` after the first cross-loop close cleared ``_loop``.
        asyncio.run(cm.close())

        loop1.call_soon_threadsafe(release_lock.set)
        holder.result(timeout=5)
        assert session_closed.wait(timeout=5)
        assert cm._session is None
        assert cm.is_closed is True
    finally:
        if not release_lock.is_set():
            loop1.call_soon_threadsafe(release_lock.set)
        if holder is not None:
            holder.result(timeout=5)
        _shutdown_loop(loop1, thread)


@pytest.mark.asyncio
async def test_cancel_and_await_same_loop_propagates_cleanup_runtime_error():
    """Teardown must not hide unrelated RuntimeErrors from task cleanup."""
    started = asyncio.Event()

    async def task_body():
        try:
            started.set()
            await asyncio.Future()
        except asyncio.CancelledError:
            raise RuntimeError("cleanup failed")

    task = asyncio.create_task(task_body())
    await started.wait()

    with pytest.raises(RuntimeError, match="cleanup failed"):
        await cancel_and_await_if_same_loop(task, asyncio.get_running_loop())


@pytest.mark.asyncio
async def test_cancel_and_await_same_loop_swallows_loop_association_error():
    """The known cross-loop RuntimeError remains safe to suppress."""
    started = asyncio.Event()

    async def task_body():
        try:
            started.set()
            await asyncio.Future()
        except asyncio.CancelledError:
            raise RuntimeError("got Future attached to a different loop")

    task = asyncio.create_task(task_body())
    await started.wait()

    await cancel_and_await_if_same_loop(task, asyncio.get_running_loop())
    assert task.done()


@pytest.mark.asyncio
async def test_cancel_and_await_same_loop_swallows_loop_bound_primitive_error():
    """The asyncio/mixins.py wording of the same error class is absorbed too."""
    foreign_loop, thread = _run_loop_in_thread()
    try:
        foreign_event = asyncio.Event()

        async def bind_to_foreign_loop():
            # An Event binds to whichever loop first awaits it.
            try:
                await asyncio.wait_for(foreign_event.wait(), timeout=0.05)
            except asyncio.TimeoutError:
                pass

        asyncio.run_coroutine_threadsafe(bind_to_foreign_loop(), foreign_loop).result(
            timeout=5
        )

        # Precondition: this construction really does produce the mixins
        # wording, so the assertion below can't pass on the other one.
        with pytest.raises(RuntimeError, match="bound to a different event loop"):
            await foreign_event.wait()

        started = asyncio.Event()

        async def task_body():
            try:
                started.set()
                await asyncio.Future()
            except asyncio.CancelledError:
                await foreign_event.wait()

        task = asyncio.create_task(task_body())
        await started.wait()

        await cancel_and_await_if_same_loop(task, asyncio.get_running_loop())
        assert task.done()
    finally:
        _shutdown_loop(foreign_loop, thread)


def test_connection_manager_same_loop_close_still_works():
    """A same-loop start/close cycle behaves normally."""
    cfg = AsyncConfig()
    cfg.enable_performance_logging = True
    cm = ConnectionManager(cfg)

    async def run():
        await cm.start()
        assert cm._session is not None
        await cm.close()
        assert cm._closed is True
        assert cm._session is None

    asyncio.run(run())


def test_async_execution_manager_cross_loop_stop_no_deadlock():
    """start() on loop1, stop() from loop2 must not hang or raise."""
    cfg = AsyncConfig()
    mgr = AsyncExecutionManager(base_url="http://localhost:8080", config=cfg)

    loop1, thread = _run_loop_in_thread()
    try:
        asyncio.run_coroutine_threadsafe(mgr.start(), loop1).result(timeout=5)
        assert mgr._polling_task is not None
        time.sleep(0.15)

        async def stop_here():
            await asyncio.wait_for(mgr.stop(), timeout=5.0)

        asyncio.run(stop_here())

        # Cross-loop stop drops all background task refs.
        assert mgr._polling_task is None
        assert mgr._cleanup_task is None
        assert mgr._metrics_task is None
        assert mgr._event_stream_task is None
        assert mgr._loop is None
    finally:
        _shutdown_loop(loop1, thread)


def test_async_execution_manager_same_loop_start_stop():
    """A same-loop start/stop cycle behaves normally."""
    cfg = AsyncConfig()
    mgr = AsyncExecutionManager(base_url="http://localhost:8080", config=cfg)

    async def run():
        await mgr.start()
        assert mgr._polling_task is not None
        await mgr.stop()
        assert mgr._polling_task is None
        assert mgr._loop is None

    asyncio.run(run())
