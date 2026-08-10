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
