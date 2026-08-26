"""Safe bridge for running async coroutines from synchronous code.

``asyncio.run()`` raises ``RuntimeError`` when called from within an
already-running event loop (e.g. a sync reasoner dispatched by FastAPI/
uvicorn, or a destructor on the event-loop thread). This module provides
helpers that detect the situation and handle it correctly:

- ``run_coroutine``: blocks until the coroutine completes. If a loop is
  already running, schedules the work on a **new thread** with its own loop
  so the caller can safely block without deadlocking the running loop.
- ``fire_and_forget``: schedules the coroutine without waiting for the
  result. If a loop is running, creates a task on it; otherwise hands it to
  a single long-lived background loop shared by the whole process.

Part of #620 (slice 3: asyncio.run() inside a running loop).
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
from typing import Any, Coroutine, Optional, TypeVar

from .logger import get_logger

logger = get_logger(__name__)

T = TypeVar("T")

# Strong references to tasks scheduled by ``fire_and_forget`` on a running
# loop. asyncio only keeps a weak reference to a task, so without this the
# task can be garbage-collected mid-flight and silently never complete.
_BACKGROUND_TASKS: set[asyncio.Task[Any]] = set()

# The single long-lived loop that serves ``fire_and_forget`` calls made from
# threads with no running loop. See ``_background_loop``.
_BACKGROUND_LOOP: Optional[asyncio.AbstractEventLoop] = None
_BACKGROUND_LOOP_LOCK = threading.Lock()
_BACKGROUND_LOOP_START_TIMEOUT = 5.0


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


def _on_background_future_done(
    future: "concurrent.futures.Future[Any]",
) -> None:
    """Retrieve/log the outcome of work submitted to the background loop.

    Mirrors :func:`_on_background_task_done` for the no-running-loop path: an
    unretrieved failure would otherwise be reported by asyncio's default
    handler when the future is collected.
    """
    if future.cancelled():
        return
    exc = future.exception()
    if exc is not None:
        logger.debug("fire_and_forget background task failed", exc_info=exc)


def _background_loop() -> Optional[asyncio.AbstractEventLoop]:
    """Return the shared background loop, starting it on first use.

    Every ``fire_and_forget`` call made without a running loop is served by
    this one loop, which stays open for the life of the process. That matters
    beyond thread economy: a loop created and *closed* per call leaves
    whatever the coroutine bound to it — most importantly the SDK's shared
    ``httpx.AsyncClient`` connection pool — attached to a dead loop, and the
    next use of that object on the caller's own loop then fails with
    ``RuntimeError: Event loop is closed`` (#620).

    Returns ``None`` if the loop could not be started, so callers can fall
    back rather than lose the work.
    """
    global _BACKGROUND_LOOP

    loop = _BACKGROUND_LOOP
    if loop is not None and not loop.is_closed():
        return loop

    with _BACKGROUND_LOOP_LOCK:
        loop = _BACKGROUND_LOOP
        if loop is not None and not loop.is_closed():
            return loop

        try:
            loop = asyncio.new_event_loop()
        except Exception:
            logger.debug("Could not create the background event loop", exc_info=True)
            return None

        running = threading.Event()

        def _run() -> None:
            asyncio.set_event_loop(loop)
            loop.call_soon(running.set)
            loop.run_forever()

        thread = threading.Thread(
            target=_run,
            name="agentfield-background-loop",
            daemon=True,
        )
        thread.start()

        if not running.wait(timeout=_BACKGROUND_LOOP_START_TIMEOUT):
            logger.debug("Background event loop did not start in time")
            return None

        _BACKGROUND_LOOP = loop
        return loop


def background_dispatch_loop() -> Optional[asyncio.AbstractEventLoop]:
    """Return the shared background loop, but only if it is already running.

    Unlike :func:`_background_loop` this never starts one. It exists so other
    modules can *recognise* the loop that best-effort work runs on without
    bringing a thread to life as a side effect of asking — see
    ``AgentFieldClient.get_async_http_client``, which must not let this loop
    take ownership of the agent's shared HTTP client.
    """
    loop = _BACKGROUND_LOOP
    if loop is None or loop.is_closed():
        return None
    return loop


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
    - **No running loop**: hands the coroutine to the SDK's shared background
      loop (see :func:`_background_loop`), which runs for the life of the
      process. The coroutine therefore never runs on a loop that is about to
      be closed, so anything it binds — connection pools, locks — stays
      usable. Note the background loop runs on a daemon thread, so callers
      that must see the work complete should not use this helper.

    Use this for best-effort background work (sending notes, cleanup) where
    the caller doesn't need the result and shouldn't block.

    Args:
        coro: The coroutine to schedule.
    """
    try:
        # create_task() is inside the guard as well: a loop that is mid-shutdown
        # raises here, and the background loop is a better home for the work
        # than an exception in the caller's face.
        task = asyncio.get_running_loop().create_task(coro)
    except RuntimeError:
        _submit_to_background_loop(coro)
        return

    _BACKGROUND_TASKS.add(task)
    task.add_done_callback(_on_background_task_done)


def _submit_to_background_loop(coro: Coroutine[Any, Any, Any]) -> None:
    """Run ``coro`` on the shared background loop, never raising."""
    loop = _background_loop()

    if loop is not None:
        try:
            future = asyncio.run_coroutine_threadsafe(coro, loop)
        except RuntimeError:
            # The loop stopped between the lookup and the submit.
            logger.debug("Could not submit work to the background loop", exc_info=True)
        else:
            future.add_done_callback(_on_background_future_done)
            return

    # Last resort: a one-shot loop on its own thread. Only reached when the
    # shared loop could not be started at all, in which case dropping the
    # work outright would be worse than the throwaway loop.
    def _worker() -> None:
        try:
            asyncio.run(coro)
        except Exception:
            logger.debug("fire_and_forget background task failed", exc_info=True)

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
