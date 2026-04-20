import asyncio
from typing import Dict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentfield.client import ApprovalResult

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