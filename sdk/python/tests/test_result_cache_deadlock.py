"""Regression tests for ResultCache cross-loop deadlock (issue #623).

The ResultCache mixes a threading.RLock (for its data) with loop-bound
asyncio primitives (asyncio.Event for shutdown, asyncio.Task for the cleanup
loop). Before the fix, calling start() on one event loop and stop() on
another — which happens when the AgentFieldClient's sync and async execution
paths are mixed (issue #620) — raised "got Future attached to a different
loop" and could wedge the process.

These tests exercise the loop-aware lifecycle:
- cross-loop stop() cancels the cleanup task on its owning loop without
  awaiting it from the wrong loop
- start() is idempotent on the same loop and rebinds cleanly on a new loop
- the cache is always cleared on stop() regardless of loop
"""

import asyncio
import threading
import time

from agentfield.async_config import AsyncConfig
from agentfield.result_cache import ResultCache


def test_cross_loop_stop_does_not_deadlock():
    """start() on loop1, stop() on loop2 must not hang or raise cross-loop."""
    cfg = AsyncConfig(enable_result_caching=True, cleanup_interval=0.05)
    cache = ResultCache(cfg)

    # Run loop1 in its own thread and keep it alive so it can process the
    # thread-safe cancel scheduled by the cross-loop stop().
    loop1 = asyncio.new_event_loop()

    def run_loop1():
        asyncio.set_event_loop(loop1)
        loop1.run_forever()

    t = threading.Thread(target=run_loop1, daemon=True)
    t.start()
    try:
        fut = asyncio.run_coroutine_threadsafe(cache.start(), loop1)
        fut.result(timeout=3)
        assert cache._cleanup_task is not None
        time.sleep(0.15)  # let the cleanup loop iterate a couple of times

        # stop() from a *different* loop — this is the deadlock scenario.
        async def stop_here():
            await asyncio.wait_for(cache.stop(), timeout=3.0)

        asyncio.run(stop_here())

        # After a cross-loop stop, the task reference is cleared and the
        # cache is emptied.
        assert cache._cleanup_task is None
        assert cache._shutdown_event is None
        assert len(cache) == 0
    finally:
        time.sleep(0.2)  # let the loop process scheduled cross-loop cancels
        loop1.call_soon_threadsafe(loop1.stop)
        t.join(timeout=2)
        if not loop1.is_closed():
            loop1.close()


def test_same_loop_start_stop_awaits_task():
    """On a single loop, stop() cleanly cancels and awaits the cleanup task."""
    cfg = AsyncConfig(enable_result_caching=True, cleanup_interval=0.05)
    cache = ResultCache(cfg)

    async def run():
        await cache.start()
        task = cache._cleanup_task
        assert task is not None and not task.done()
        await cache.stop()
        # Task fully settled after same-loop stop.
        assert task.done()
        assert cache._cleanup_task is None
        assert cache._shutdown_event is None
        assert cache._loop is None

    asyncio.run(run())


def test_start_is_idempotent_on_same_loop():
    """Calling start() twice on the same loop reuses the running task."""
    cfg = AsyncConfig(enable_result_caching=True, cleanup_interval=0.05)
    cache = ResultCache(cfg)

    async def run():
        await cache.start()
        first = cache._cleanup_task
        await cache.start()
        assert cache._cleanup_task is first, "start() must not spawn a second task"
        await cache.stop()

    asyncio.run(run())


def test_start_rebinds_on_new_loop_without_error():
    """A client reused across loops rebinds its cleanup task cleanly."""
    cfg = AsyncConfig(enable_result_caching=True, cleanup_interval=0.05)
    cache = ResultCache(cfg)

    # First loop: start, then leave the loop without stopping (simulates a
    # short-lived asyncio.run() block that used the shared client).
    loop1 = asyncio.new_event_loop()

    def run_loop1():
        asyncio.set_event_loop(loop1)
        loop1.run_forever()

    t = threading.Thread(target=run_loop1, daemon=True)
    t.start()
    try:
        asyncio.run_coroutine_threadsafe(cache.start(), loop1).result(timeout=3)
        stale_task = cache._cleanup_task
        assert stale_task is not None

        # Second loop: start() again should discard the stale task (bound to
        # loop1) and create a fresh one bound to loop2 — no cross-loop error.
        async def restart_and_use():
            await cache.start()
            assert cache._cleanup_task is not stale_task
            cache.set("k", "v")
            assert cache.get("k") == "v"
            await cache.stop()

        asyncio.run(restart_and_use())
    finally:
        time.sleep(0.2)  # let the loop process scheduled cross-loop cancels
        loop1.call_soon_threadsafe(loop1.stop)
        t.join(timeout=2)
        if not loop1.is_closed():
            loop1.close()


def test_stop_without_start_is_safe():
    """stop() before any start() must not raise."""
    cfg = AsyncConfig(enable_result_caching=True)
    cache = ResultCache(cfg)
    cache.set("a", 1)

    async def run():
        await cache.stop()  # no task exists

    asyncio.run(run())
    assert len(cache) == 0


def test_disabled_cache_lifecycle_is_noop():
    """A caching-disabled cache never spawns a cleanup task."""
    cfg = AsyncConfig(enable_result_caching=False)
    cache = ResultCache(cfg)

    async def run():
        await cache.start()
        assert cache._cleanup_task is None
        await cache.stop()
        assert cache._cleanup_task is None

    asyncio.run(run())


def test_cleanup_loop_with_performance_logging():
    """Exercise the perf-logging branch of the cleanup loop under real ttl.

    Covers the get_stats() path inside _cleanup_loop that only runs when
    enable_performance_logging is set. Also verifies the loop still sweeps
    expired entries when stats logging is on.
    """
    cfg = AsyncConfig(
        enable_result_caching=True,
        result_cache_ttl=0.05,
        cleanup_interval=0.02,
        enable_performance_logging=True,
    )
    cache = ResultCache(cfg)

    async def run():
        await cache.start()
        cache.set("k", "v")
        await asyncio.sleep(0.15)  # let ttl expire + cleanup + stats logging run
        result = cache.get("k")
        await cache.stop()
        return result

    assert asyncio.run(run()) is None


def test_concurrent_sync_access_during_cleanup_no_deadlock():
    """Sync get/set from worker threads while the async cleanup loop runs.

    Exercises the threading.RLock under concurrent access from both the
    event-loop thread (cleanup sweep) and external worker threads (sync
    reasoner path). Must complete without deadlocking.
    """
    cfg = AsyncConfig(
        enable_result_caching=True,
        result_cache_ttl=0.05,
        cleanup_interval=0.02,
    )
    cache = ResultCache(cfg)

    stop_flag = threading.Event()

    def hammer():
        i = 0
        while not stop_flag.is_set():
            cache.set(f"k{i % 50}", i)
            cache.get(f"k{(i + 1) % 50}")
            cache.get_stats()
            i += 1

    async def run():
        await cache.start()
        workers = [threading.Thread(target=hammer, daemon=True) for _ in range(4)]
        for w in workers:
            w.start()
        # Let cleanup and workers contend for a bit.
        await asyncio.sleep(0.3)
        stop_flag.set()
        for w in workers:
            w.join(timeout=2)
            assert not w.is_alive(), "worker thread deadlocked on the cache lock"
        await cache.stop()

    asyncio.run(asyncio.wait_for(run(), timeout=10))
