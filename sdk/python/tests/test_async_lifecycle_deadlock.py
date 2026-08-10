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

from agentfield.async_config import AsyncConfig
from agentfield.async_execution_manager import AsyncExecutionManager
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

    loop1, thread = _run_loop_in_thread()
    try:
        asyncio.run_coroutine_threadsafe(cm.start(), loop1).result(timeout=5)
        assert cm._session is not None
        time.sleep(0.15)

        async def close_here():
            await asyncio.wait_for(cm.close(), timeout=5.0)

        asyncio.run(close_here())

        # Cross-loop close marks the manager closed and drops task refs.
        assert cm._closed is True
        assert cm._health_check_task is None
        assert cm._cleanup_task is None
    finally:
        _shutdown_loop(loop1, thread)


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
