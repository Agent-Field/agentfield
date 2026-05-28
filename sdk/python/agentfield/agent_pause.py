import asyncio
import time
from typing import Dict, Optional
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentfield.client import ApprovalResult


class PauseClock:
    """Tracks how long a single execution has spent inside ``Agent.pause()``.

    The reasoner-level wall-clock timeout in ``_execute_async_with_callback``
    must not count time spent waiting for an external approval — otherwise
    ``expires_in_hours`` is silently capped at the reasoner timeout.  The
    watchdog in ``_execute_async_with_callback`` reads ``total_paused()`` and
    subtracts it from elapsed wall-clock to decide whether to fire a timeout.

    A reasoner is a single asyncio task, so ``pause()`` calls cannot overlap
    on the same clock.  No locking is needed.
    """

    __slots__ = ("_total_paused", "_pause_started_at", "timed_out")

    def __init__(self) -> None:
        self._total_paused: float = 0.0
        self._pause_started_at: Optional[float] = None
        # Set by the watchdog when it cancels the reasoner for exceeding
        # the active-time budget.  Distinguishes timeout-cancel from an
        # external cooperative cancel arriving via the cancel dispatcher.
        self.timed_out: bool = False

    def start_pause(self) -> None:
        if self._pause_started_at is None:
            self._pause_started_at = time.time()

    def end_pause(self) -> None:
        if self._pause_started_at is not None:
            self._total_paused += time.time() - self._pause_started_at
            self._pause_started_at = None

    def total_paused(self) -> float:
        """Cumulative paused seconds, including any in-progress pause."""
        if self._pause_started_at is None:
            return self._total_paused
        return self._total_paused + (time.time() - self._pause_started_at)


class _PauseManager:
    """Manages pending execution pause futures resolved via webhook callback.

    Each call to ``Agent.pause()`` registers an ``asyncio.Future`` keyed by
    ``approval_request_id``.  When the webhook route receives a resolution
    callback from the control plane it resolves the matching future, unblocking
    the caller.
    """

    def __init__(self) -> None:
        self._pending: Dict[str, asyncio.Future] = {}
        # Also track execution_id → approval_request_id for fallback resolution
        self._exec_to_request: Dict[str, str] = {}
        self._lock = asyncio.Lock()

    async def register(
        self, approval_request_id: str, execution_id: str = ""
    ) -> asyncio.Future:
        """Register a new pending pause and return the Future to await."""
        async with self._lock:
            if approval_request_id in self._pending:
                return self._pending[approval_request_id]
            loop = asyncio.get_running_loop()
            future = loop.create_future()
            self._pending[approval_request_id] = future
            if execution_id:
                self._exec_to_request[execution_id] = approval_request_id
            return future

    async def resolve(self, approval_request_id: str, result: "ApprovalResult") -> bool:
        """Resolve a pending pause by approval_request_id.  Returns True if a waiter was found."""
        async with self._lock:
            future = self._pending.pop(approval_request_id, None)
            # Clean up execution mapping
            exec_id = None
            for eid, rid in self._exec_to_request.items():
                if rid == approval_request_id:
                    exec_id = eid
                    break
            if exec_id:
                self._exec_to_request.pop(exec_id, None)
            if future and not future.done():
                future.set_result(result)
                return True
            return False

    async def resolve_by_execution_id(
        self, execution_id: str, result: "ApprovalResult"
    ) -> bool:
        """Fallback: resolve by execution_id when approval_request_id is not in the callback."""
        async with self._lock:
            request_id = self._exec_to_request.pop(execution_id, None)
            if request_id:
                future = self._pending.pop(request_id, None)
                if future and not future.done():
                    future.set_result(result)
                    return True
            return False

    async def cancel_all(self) -> None:
        """Cancel all pending futures (for shutdown)."""
        async with self._lock:
            for future in self._pending.values():
                if not future.done():
                    future.cancel()
            self._pending.clear()
            self._exec_to_request.clear()
