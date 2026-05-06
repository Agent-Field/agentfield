"""Cooperative cancellation glue between the control plane and reasoner code.

The control plane's CancelDispatcher (Go service in
control-plane/internal/services/cancel_dispatcher.go) POSTs to
``/_internal/executions/{execution_id}/cancel`` whenever an execution
flips to cancelled — from per-execution cancel, the bottom-up cancel-tree
endpoint, or any future source publishing on the bus. This module wires
that callback into the running asyncio.Task for the matching execution
so ``CancelledError`` propagates into the user's reasoner coroutine.

The contract for reasoner authors:
- ``await``-based I/O (httpx, the official anthropic / openai SDKs,
  database drivers like asyncpg) honors task cancellation natively. No
  changes required.
- Long synchronous CPU loops or ``time.sleep`` won't see the cancellation
  until the next ``await`` checkpoint. For those, periodically check
  ``execution_cancelled()`` (or the ``cancel_event`` on the execution
  context) and bail.
- ``except Exception`` around the entire reasoner body will swallow
  ``CancelledError`` (which is a ``BaseException`` in 3.8+). Re-raise it.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Optional

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

if TYPE_CHECKING:
    from .agent import Agent

logger = logging.getLogger("agentfield.cancel")


async def register_execution_task(
    agent: "Agent", execution_id: str, task: asyncio.Task
) -> None:
    """Register an in-flight asyncio.Task against an execution_id.

    Called from the FastAPI endpoint after the reasoner task is created
    and before awaiting it. Idempotent: re-registering the same id replaces
    the prior entry, which is the right behaviour for retried executions.
    """
    if not execution_id:
        return
    async with agent._cancel_lock:
        agent._cancel_tasks[execution_id] = task


async def deregister_execution(agent: "Agent", execution_id: str) -> None:
    """Drop the registry entry for `execution_id`.

    Safe to call multiple times. Always called from a `finally` after
    the reasoner task completes (success, failure, or cancellation).
    """
    if not execution_id:
        return
    async with agent._cancel_lock:
        agent._cancel_tasks.pop(execution_id, None)


async def cancel_execution(agent: "Agent", execution_id: str) -> bool:
    """Cancel the asyncio.Task registered for `execution_id`.

    Returns True if a matching task was found and ``cancel()``-ed, False
    if there was no active execution with that id (already finished, or
    never dispatched here).
    """
    if not execution_id:
        return False
    async with agent._cancel_lock:
        task = agent._cancel_tasks.get(execution_id)
    if task is None or task.done():
        return False
    task.cancel()
    return True


def install_cancel_route(agent: "Agent") -> None:
    """Register the ``POST /_internal/executions/{execution_id}/cancel``
    handler on the agent's FastAPI app.

    The route bypasses DID verification by way of its path prefix
    (control-plane→worker notification, not a user-initiated DID-signed
    call). Bearer-token origin auth, if configured on the agent, still
    applies through the usual middleware path.
    """

    @agent.post("/_internal/executions/{execution_id}/cancel")
    async def _cancel_execution(execution_id: str, request: Request):
        # Path-level guards: HTTPException for shape errors so FastAPI
        # produces the right JSON shape automatically.
        if not execution_id or "/" in execution_id:
            raise HTTPException(status_code=404, detail="invalid execution_id")

        cancelled = await cancel_execution(agent, execution_id)
        body = {"cancelled": cancelled, "execution_id": execution_id}
        if not cancelled:
            # 200 with the marker — don't generate an alarm log on the
            # dispatcher side. The execution may have already finished
            # naturally, or never landed on this worker.
            body["reason"] = "execution_not_active"
        else:
            logger.info(
                "cancel-callback fired for execution_id=%s (source=%s)",
                execution_id,
                request.headers.get("X-AgentField-Source", "unknown"),
            )
        return JSONResponse(status_code=200, content=body)


def is_execution_cancelled(agent: "Agent", execution_id: Optional[str]) -> bool:
    """Synchronous helper for reasoner code to poll cancellation status.

    Useful inside CPU loops or sync blocks where ``CancelledError`` won't
    fire until the next ``await``. Returns False if there's no live
    registration (either never registered, or already cancelled and
    cleaned up).
    """
    if not execution_id:
        return False
    task = agent._cancel_tasks.get(execution_id)
    if task is None:
        return False
    return task.cancelled() or (task.done() and task.cancelling() > 0)
