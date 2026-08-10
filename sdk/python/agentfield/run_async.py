"""Safe bridge for running async coroutines from synchronous code.

``asyncio.run()`` raises ``RuntimeError`` when called from within an
already-running event loop (e.g. a sync reasoner dispatched by FastAPI/
uvicorn, or a destructor on the event-loop thread). This module provides
helpers that detect the situation and handle it correctly:

- ``run_coroutine``: blocks until the coroutine completes. If a loop is
  already running, schedules the work on a **new thread** with its own loop
  so the caller can safely block without deadlocking the running loop.
- ``fire_and_forget``: schedules the coroutine without waiting for the
  result. If a loop is running, creates a task on it; otherwise spawns a
  daemon thread.

Part of #620 (slice 3: asyncio.run() inside a running loop).
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any, Coroutine, TypeVar

from .logger import get_logger

logger = get_logger(__name__)

T = TypeVar("T")

# Strong references to tasks scheduled by ``fire_and_forget`` on a running
# loop. asyncio only keeps a weak reference to a task, so without this the
# task can be garbage-collected mid-flight and silently never complete.
_BACKGROUND_TASKS: set[asyncio.Task[Any]] = set()


def _has_running_loop() -> bool:
    """Return True if there is a running event loop on the current thread."""
    try:
        asyncio.get_running_loop()
        return True
    except RuntimeError:
        return False


def _on_background_task_done(task: asyncio.Task[Any]) -> None:
    """Release the task reference and retrieve/log any failure quietly.

    Retrieving the exception here is what keeps a failed fire-and-forget
    task from emitting a noisy ``Task exception was never retrieved``
    traceback when it is garbage-collected.
    """
    _BACKGROUND_TASKS.discard(task)
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.debug("fire_and_forget background task failed", exc_info=exc)


def run_coroutine(coro: Coroutine[Any, Any, T]) -> T:
    """Run a coroutine from sync code, safe even inside a running loop.

    - **No running loop**: uses ``asyncio.run(coro)`` directly.
    - **Running loop**: runs the coroutine in a new daemon thread with its
      own event loop and blocks until it completes. This avoids the nested
      ``asyncio.run()`` RuntimeError while still delivering the result
      synchronously to the caller.

    Use this to replace bare ``asyncio.run(coro)`` calls in sync handlers
    that may be invoked from within a running loop (FastAPI, serverless
    wrappers, etc.).

    Args:
        coro: The coroutine to execute.

    Returns:
        The coroutine's return value.

    Raises:
        Whatever the coroutine raises (propagated from the worker thread).
    """
    if not _has_running_loop():
        return asyncio.run(coro)

    # A loop is running on this thread. We cannot call asyncio.run() here
    # (it would raise), and we cannot await (we're in sync code). Solution:
    # run the coroutine in a fresh loop on a new thread and block this
    # thread until it finishes.
    result: Any = None
    exception: BaseException | None = None

    def _worker() -> None:
        nonlocal result, exception
        try:
            result = asyncio.run(coro)
        except BaseException as exc:
            exception = exc

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    thread.join()

    if exception is not None:
        raise exception
    return result


def fire_and_forget(coro: Coroutine[Any, Any, Any]) -> None:
    """Schedule a coroutine without waiting for the result.

    - **Running loop**: creates a task on the current loop (no new thread).
      The task is retained until it finishes and its failure (if any) is
      logged at debug level.
    - **No running loop**: spawns a daemon thread that runs the coroutine.
      Note that a daemon thread is killed at interpreter exit, so callers
      that must see the work complete should not use this helper.

    Use this for best-effort background work (sending notes, cleanup) where
    the caller doesn't need the result and shouldn't block.

    Args:
        coro: The coroutine to schedule.
    """
    try:
        loop = asyncio.get_running_loop()
        task = loop.create_task(coro)
        _BACKGROUND_TASKS.add(task)
        task.add_done_callback(_on_background_task_done)
    except RuntimeError:
        # No running loop — run in a background thread with exception
        # handling so failures are logged cleanly rather than surfacing
        # as noisy unhandled-thread-exception tracebacks.
        def _worker() -> None:
            try:
                asyncio.run(coro)
            except Exception:
                logger.debug("fire_and_forget background task failed", exc_info=True)

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()
