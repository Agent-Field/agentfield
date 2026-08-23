"""Lock acquisition with timeout to prevent indefinite hangs (#620).

The SDK uses threading.Lock / threading.RLock in several places. Without a
timeout, a deadlocked or contended lock causes the process to hang
indefinitely — the "stuck for hours" symptom reported in #620.

This module provides a context-manager wrapper that acquires with a timeout
and raises a clear error instead of hanging forever.
"""

from __future__ import annotations

import math
import os
import threading
from contextlib import contextmanager
from typing import Optional, Union

from .logger import get_logger

logger = get_logger(__name__)

# Default timeout for lock acquisition (seconds). Long enough to never
# trip under normal contention, short enough to surface a real deadlock
# within minutes rather than hours. Configurable via the env var
# AGENTFIELD_LOCK_TIMEOUT_SECONDS.
FALLBACK_LOCK_TIMEOUT: float = 30.0

_LOCK_TIMEOUT_ENV_VAR = "AGENTFIELD_LOCK_TIMEOUT_SECONDS"


def _parse_lock_timeout(raw: Optional[str]) -> float:
    """Parse the lock timeout env var, falling back to the default.

    This runs at import time, so it must never raise: an unset-but-present
    (``AGENTFIELD_LOCK_TIMEOUT_SECONDS=``) or malformed value would otherwise
    break ``import agentfield`` entirely. A non-positive or non-finite value
    is rejected too — ``lock.acquire()`` refuses those on every call.
    """
    if raw is None or not raw.strip():
        return FALLBACK_LOCK_TIMEOUT

    try:
        timeout = float(raw)
    except ValueError:
        logger.warning(
            f"Ignoring invalid {_LOCK_TIMEOUT_ENV_VAR}={raw!r} "
            f"(not a number); using {FALLBACK_LOCK_TIMEOUT}s"
        )
        return FALLBACK_LOCK_TIMEOUT

    if not math.isfinite(timeout) or timeout <= 0:
        logger.warning(
            f"Ignoring out-of-range {_LOCK_TIMEOUT_ENV_VAR}={raw!r} "
            f"(must be a positive, finite number of seconds); "
            f"using {FALLBACK_LOCK_TIMEOUT}s"
        )
        return FALLBACK_LOCK_TIMEOUT

    return timeout


DEFAULT_LOCK_TIMEOUT: float = _parse_lock_timeout(os.environ.get(_LOCK_TIMEOUT_ENV_VAR))


class LockTimeoutError(RuntimeError):
    """Raised when a lock cannot be acquired within the timeout period.

    Deliberately *not* a TimeoutError: on Python 3.11+ asyncio.TimeoutError is
    TimeoutError, so an `except asyncio.TimeoutError` guard anywhere up the
    stack (Agent.call wraps client.execute in asyncio.wait_for, and that path
    touches the result cache) would swallow this and report a generic timeout,
    losing the lock name and wait diagnostics below.
    """

    def __init__(self, lock_name: str, timeout: float):
        self.lock_name = lock_name
        self.timeout = timeout
        super().__init__(
            f"Failed to acquire lock '{lock_name}' within {timeout}s. "
            f"This may indicate a deadlock. Set AGENTFIELD_LOCK_TIMEOUT_SECONDS "
            f"to adjust the timeout."
        )


@contextmanager
def timed_lock(
    lock: Union[threading.Lock, threading.RLock],
    name: str = "unnamed",
    timeout: float = DEFAULT_LOCK_TIMEOUT,
):
    """Context manager that acquires a lock with a timeout.

    Usage:
        with timed_lock(self._lock, "result_cache"):
            # critical section

    Replaces bare ``with self._lock:`` to prevent indefinite hangs.

    Args:
        lock: The threading.Lock or threading.RLock to acquire.
        name: Human-readable name for error messages and logging.
        timeout: Maximum seconds to wait. Defaults to DEFAULT_LOCK_TIMEOUT.

    Raises:
        LockTimeoutError: If the lock cannot be acquired within the timeout.
    """
    acquired = lock.acquire(timeout=timeout)
    if not acquired:
        raise LockTimeoutError(name, timeout)
    try:
        yield
    finally:
        lock.release()
