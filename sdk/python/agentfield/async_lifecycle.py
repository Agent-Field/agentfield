"""Loop-aware teardown helpers for background asyncio tasks.

Several SDK components (ResultCache, AsyncExecutionManager, the HTTP
ConnectionManager) start long-lived background tasks bound to whichever event
loop was running at ``start()`` time. When the owning object is later torn
down from a *different* loop — which happens whenever the client's sync and
async execution paths are mixed (issues #620 / #623) — the naive
``task.cancel(); await task`` pattern raises

    RuntimeError: got Future <...> attached to a different loop

and can wedge the caller waiting on a future its loop will never drive.

These helpers centralise the safe pattern:

* record the loop a task was created on (``current_running_loop``)
* only ``await`` a task when we're back on its owning loop
  (``cancel_and_await_if_same_loop``)
* otherwise cancel it on its owning loop without awaiting
  (``cancel_task_cross_loop``), so teardown never blocks or raises across
  loops.
"""

from __future__ import annotations

import asyncio
from typing import Optional

from .logger import get_logger

logger = get_logger(__name__)


def current_running_loop() -> Optional[asyncio.AbstractEventLoop]:
    """Return the running loop, or ``None`` when called outside one."""
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None


def cancel_task_cross_loop(
    task: Optional[asyncio.Task],
    owning_loop: Optional[asyncio.AbstractEventLoop],
) -> None:
    """Cancel ``task`` on its owning loop without awaiting it.

    Safe to call from any loop (or none). Never awaits — a cross-loop await is
    exactly what deadlocks. If the owning loop is still alive we schedule the
    cancel on it thread-safely; otherwise the task is unreachable garbage and
    dropping the reference is sufficient.
    """
    if task is None or task.done():
        return
    try:
        if owning_loop is not None and not owning_loop.is_closed():
            owning_loop.call_soon_threadsafe(task.cancel)
    except Exception:
        # The owning loop may be mid-teardown; the task is unreachable and
        # will be garbage-collected. Nothing actionable here.
        logger.debug("Could not cancel cross-loop task", exc_info=True)


async def cancel_and_await_if_same_loop(
    task: Optional[asyncio.Task],
    owning_loop: Optional[asyncio.AbstractEventLoop],
) -> None:
    """Cancel ``task`` and, only when safe, await it to completion.

    When called on the same loop the task is bound to, this cancels and awaits
    it (swallowing ``CancelledError``). When called from a *different* loop it
    delegates to :func:`cancel_task_cross_loop`, which cancels on the owning
    loop without awaiting — avoiding the ``got Future attached to a different
    loop`` RuntimeError.
    """
    if task is None:
        return

    if owning_loop is current_running_loop():
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except RuntimeError:
            # Defensive: if the loop association is somehow inconsistent,
            # don't let teardown wedge the caller.
            logger.debug(
                "Awaiting cancelled task raised RuntimeError during teardown",
                exc_info=True,
            )
        return

    cancel_task_cross_loop(task, owning_loop)
