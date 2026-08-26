"""Tests for agentfield.run_async — safe async-from-sync bridge (#620 slice 3).

Verifies that run_coroutine() and fire_and_forget() work correctly both
when no event loop is running (normal case) and when called from within
a running loop (the FastAPI/serverless case that previously raised
RuntimeError: asyncio.run() cannot be called from a running event loop).
"""

import asyncio
import gc
import threading
import time
from types import SimpleNamespace

from agentfield import run_async
from agentfield.run_async import run_coroutine, fire_and_forget


# ---------------------------------------------------------------------------
# run_coroutine tests
# ---------------------------------------------------------------------------


def test_run_coroutine_no_running_loop():
    """When no loop is running, behaves like asyncio.run()."""
    async def add(a, b):
        return a + b

    result = run_coroutine(add(3, 4))
    assert result == 7


def test_run_coroutine_inside_running_loop():
    """When called from within a running loop (via run_in_executor),
    still returns the correct result without raising RuntimeError."""
    async def multiply(a, b):
        await asyncio.sleep(0.01)
        return a * b

    async def main():
        loop = asyncio.get_running_loop()
        # Simulate a sync function called from within an async framework
        # (like a sync reasoner dispatched by FastAPI)
        result = await loop.run_in_executor(
            None, run_coroutine, multiply(5, 6)
        )
        return result

    result = asyncio.run(main())
    assert result == 30


def test_run_coroutine_propagates_exceptions():
    """Exceptions from the coroutine propagate to the caller."""
    async def fail():
        raise ValueError("test error")

    try:
        run_coroutine(fail())
        assert False, "Should have raised"
    except ValueError as e:
        assert str(e) == "test error"


def test_run_coroutine_exception_inside_running_loop():
    """Exceptions propagate even when called from within a running loop."""
    async def fail():
        raise RuntimeError("inner failure")

    async def main():
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, run_coroutine, fail())
            return "should not reach"
        except RuntimeError as e:
            return str(e)

    result = asyncio.run(main())
    assert result == "inner failure"


def test_run_coroutine_from_thread_with_running_main_loop():
    """A background thread can use run_coroutine while the main thread
    runs an event loop — schedules on the main loop via
    run_coroutine_threadsafe."""
    async def compute():
        await asyncio.sleep(0.01)
        return 42

    results = []

    async def main():
        def worker():
            # This thread sees the main loop running
            r = run_coroutine(compute())
            results.append(r)

        t = threading.Thread(target=worker)
        t.start()
        # Keep the main loop alive while the thread works
        await asyncio.sleep(0.2)
        t.join(timeout=2)

    asyncio.run(main())
    assert results == [42]


# ---------------------------------------------------------------------------
# fire_and_forget tests
# ---------------------------------------------------------------------------


def test_fire_and_forget_no_running_loop():
    """When no loop is running, spawns a thread and runs the coroutine."""
    flag = {"called": False}

    async def set_flag():
        flag["called"] = True

    fire_and_forget(set_flag())
    time.sleep(0.2)  # give the daemon thread time to complete
    assert flag["called"] is True


def test_fire_and_forget_with_running_loop():
    """When a loop is running, creates a task on it."""
    flag = {"called": False}

    async def set_flag():
        flag["called"] = True

    async def main():
        fire_and_forget(set_flag())
        await asyncio.sleep(0.1)  # let the task run
        return flag["called"]

    result = asyncio.run(main())
    assert result is True


def test_fire_and_forget_exception_does_not_propagate():
    """Exceptions in fire-and-forget coroutines don't crash the caller."""
    async def fail():
        raise ValueError("should not propagate")

    # Should not raise
    fire_and_forget(fail())
    time.sleep(0.1)


def test_fire_and_forget_with_running_loop_exception_does_not_crash():
    """Exceptions in tasks created by fire_and_forget don't crash the loop."""
    async def fail():
        raise ValueError("task failure")

    async def main():
        fire_and_forget(fail())
        await asyncio.sleep(0.1)
        return "loop survived"

    result = asyncio.run(main())
    assert result == "loop survived"


async def test_fire_and_forget_running_loop_failure_is_logged_not_noisy(monkeypatch):
    """C3: a failing fire-and-forget coroutine inside a running loop is
    reported once, at debug level, and never surfaces the asyncio
    'Task exception was never retrieved' warning."""
    logged = []
    monkeypatch.setattr(
        run_async,
        "logger",
        SimpleNamespace(debug=lambda message, **kwargs: logged.append((message, kwargs))),
    )

    unhandled = []
    loop = asyncio.get_running_loop()
    loop.set_exception_handler(lambda _loop, context: unhandled.append(context))

    async def fail():
        raise ValueError("task failure")

    fire_and_forget(fail())
    await asyncio.sleep(0.1)
    # Force collection of the finished task: the noisy warning is emitted
    # from Task.__del__ when nobody retrieved the exception.
    gc.collect()
    await asyncio.sleep(0.05)

    assert unhandled == []
    assert len(logged) == 1
    message, kwargs = logged[0]
    assert message == "fire_and_forget background task failed"
    assert isinstance(kwargs.get("exc_info"), ValueError)


async def test_fire_and_forget_running_loop_task_is_retained_until_done():
    """C4: the scheduled task is referenced by the module while pending (so
    it can't be garbage-collected mid-flight) and released once it finishes."""
    started = asyncio.Event()
    release = asyncio.Event()
    finished = {"value": False}

    async def work():
        started.set()
        await release.wait()
        finished["value"] = True

    before = set(run_async._BACKGROUND_TASKS)
    fire_and_forget(work())  # caller deliberately keeps no reference
    await started.wait()

    # Nothing outside the module holds the task at this point, so a
    # collection here would kill it if the module reference weren't there.
    gc.collect()

    scheduled = set(run_async._BACKGROUND_TASKS) - before
    assert len(scheduled) == 1
    task = next(iter(scheduled))
    assert not task.done()

    release.set()
    await asyncio.wait_for(task, timeout=2)
    await asyncio.sleep(0)

    assert finished["value"] is True
    assert task not in run_async._BACKGROUND_TASKS


def test_fire_and_forget_without_a_running_loop_reuses_one_open_loop():
    """Work scheduled from sync code shares one loop that is never closed.

    A loop created and closed per call strands anything the coroutine bound to
    it — notably the SDK's shared httpx.AsyncClient connection pool — so the
    next use on the caller's own loop fails with ``Event loop is closed``.
    """
    loops = []
    ran = threading.Semaphore(0)

    async def record():
        loops.append(asyncio.get_running_loop())
        ran.release()

    fire_and_forget(record())
    fire_and_forget(record())

    assert ran.acquire(timeout=5)
    assert ran.acquire(timeout=5)

    assert loops[0] is loops[1], "each call got its own event loop"
    assert not loops[0].is_closed(), "the background loop was closed after use"


def test_fire_and_forget_without_a_running_loop_survives_a_failure():
    """A failing coroutine must not stop later work from running."""
    ran = threading.Semaphore(0)

    async def fail():
        raise ValueError("boom")

    async def ok():
        ran.release()

    fire_and_forget(fail())
    fire_and_forget(ok())

    assert ran.acquire(timeout=5)


def test_background_dispatch_loop_names_the_loop_sync_work_runs_on():
    """Callers can recognise the loop best-effort work is running on.

    ``AgentFieldClient`` uses this to keep the background loop from taking
    ownership of the agent's shared HTTP client.
    """
    loops = []
    ran = threading.Semaphore(0)

    async def record():
        loops.append(asyncio.get_running_loop())
        ran.release()

    fire_and_forget(record())
    assert ran.acquire(timeout=5)

    assert run_async.background_dispatch_loop() is loops[0]


def test_background_dispatch_loop_does_not_start_one(monkeypatch):
    """Asking must never be what brings the background thread to life."""
    monkeypatch.setattr(run_async, "_BACKGROUND_LOOP", None)

    before = {thread.ident for thread in threading.enumerate()}
    assert run_async.background_dispatch_loop() is None
    after = {thread.ident for thread in threading.enumerate()}

    assert after - before == set(), "asking started a thread"


# ---------------------------------------------------------------------------
# The fallback loop must be recognisable too
# ---------------------------------------------------------------------------


def test_fallback_dispatch_loop_is_recognised_as_a_dispatch_loop(monkeypatch):
    """The last-resort loop is a dispatch loop, and the loudest kind.

    When the shared loop cannot be started, ``fire_and_forget`` runs the work
    on a one-shot loop that closes the moment it finishes. A caller that keeps
    loop-affine state must be able to tell — otherwise that loop looks like an
    ordinary caller, claims the state, and takes it to the grave (#620).
    """
    monkeypatch.setattr(run_async, "_background_loop", lambda: None)

    seen = {}
    ran = threading.Semaphore(0)

    async def record():
        loop = asyncio.get_running_loop()
        seen["loop"] = loop
        seen["is_dispatch"] = run_async.is_background_dispatch_loop(loop)
        ran.release()

    fire_and_forget(record())

    assert ran.acquire(timeout=5), "the fallback path never ran the work"
    assert seen["is_dispatch"] is True


def test_the_shared_background_loop_is_recognised_as_a_dispatch_loop():
    loops = []
    ran = threading.Semaphore(0)

    async def record():
        loops.append(asyncio.get_running_loop())
        ran.release()

    fire_and_forget(record())
    assert ran.acquire(timeout=5)

    assert run_async.is_background_dispatch_loop(loops[0]) is True


def test_an_ordinary_loop_is_not_a_dispatch_loop():
    loop = asyncio.new_event_loop()
    try:
        assert run_async.is_background_dispatch_loop(loop) is False
    finally:
        loop.close()


def test_recognising_a_dispatch_loop_starts_nothing(monkeypatch):
    """The yes/no answer must not be what brings a thread to life."""
    monkeypatch.setattr(run_async, "_BACKGROUND_LOOP", None)

    loop = asyncio.new_event_loop()
    before = {thread.ident for thread in threading.enumerate()}
    try:
        assert run_async.is_background_dispatch_loop(loop) is False
    finally:
        loop.close()

    assert {thread.ident for thread in threading.enumerate()} - before == set()


# ---------------------------------------------------------------------------
# A background loop that never signals readiness is not left running
# ---------------------------------------------------------------------------


def test_background_loop_that_never_starts_is_not_left_behind(monkeypatch):
    """A start that times out must not strand its loop and thread.

    ``_BACKGROUND_LOOP`` is only published on success, so a loop left running
    after a timeout is unreachable forever — and the next call builds another
    one beside it.
    """
    monkeypatch.setattr(run_async, "_BACKGROUND_LOOP", None)
    monkeypatch.setattr(run_async, "_BACKGROUND_LOOP_START_TIMEOUT", 0.25)

    real_new_event_loop = asyncio.new_event_loop
    park = threading.Event()
    arm = {"on": False}

    def _stalled_loop():
        loop = real_new_event_loop()
        if arm["on"]:
            arm["on"] = False
            # Queued ahead of the readiness callback, so the loop is busy
            # here when the start timeout expires.
            loop.call_soon_threadsafe(lambda: park.wait(timeout=10))
        return loop

    monkeypatch.setattr(asyncio, "new_event_loop", _stalled_loop)

    before = {
        t for t in threading.enumerate() if t.name == "agentfield-background-loop"
    }

    started = []
    try:
        for _ in range(3):
            arm["on"] = True
            assert run_async._background_loop() is None
            started = [
                t
                for t in threading.enumerate()
                if t.name == "agentfield-background-loop" and t not in before
            ]
    finally:
        park.set()

    assert started, "the test never created a background-loop thread to strand"

    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and any(t.is_alive() for t in started):
        time.sleep(0.01)

    assert not [t for t in started if t.is_alive()], (
        "a background loop that failed to start was left running"
    )
