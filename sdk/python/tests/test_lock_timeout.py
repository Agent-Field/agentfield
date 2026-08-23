"""Tests for lock timeout utility and blocking-call offloading (#620 slice 4).

Verifies:
- timed_lock raises LockTimeoutError instead of hanging when a lock is
  already held by another thread
- timed_lock works normally (acquires/releases) when uncontended
- LockTimeoutError has useful debug info (lock name, timeout value)
- The execute_sync running-loop warning fires when called from inside a loop

The other half of #620 slice 4 — memory_events.history()'s blocking requests
fallback — has no test here: it is guarded by the ASYNC210 ruff gate, whose
per-file ignore for memory_events.py was dropped in pyproject.toml.
"""

import asyncio
import os
import subprocess
import sys
import threading
import time
import warnings

import pytest

from agentfield.lock_utils import (
    DEFAULT_LOCK_TIMEOUT,
    FALLBACK_LOCK_TIMEOUT,
    LockTimeoutError,
    timed_lock,
)


def test_timed_lock_acquires_uncontended():
    """Normal case: lock is acquired and released."""
    lock = threading.Lock()
    with timed_lock(lock, "test_lock", timeout=1.0):
        assert not lock.acquire(blocking=False)  # lock is held
    assert lock.acquire(blocking=False)  # lock is released
    lock.release()


def test_timed_lock_rlock_reentrant():
    """RLock supports reentrant acquisition."""
    lock = threading.RLock()
    with timed_lock(lock, "test_rlock", timeout=1.0):
        with timed_lock(lock, "test_rlock", timeout=1.0):
            pass  # should not raise


def test_timed_lock_raises_on_timeout():
    """When lock is held by another thread, raises after timeout."""
    lock = threading.Lock()
    lock.acquire()  # hold the lock

    try:
        with pytest.raises(LockTimeoutError) as exc_info:
            with timed_lock(lock, "blocked_lock", timeout=0.1):
                pass
        assert "blocked_lock" in str(exc_info.value)
        assert "0.1s" in str(exc_info.value)
    finally:
        lock.release()


def test_timed_lock_timeout_from_another_thread():
    """Lock held by thread A, thread B times out trying to acquire."""
    lock = threading.Lock()
    results = []

    def holder():
        with timed_lock(lock, "holder", timeout=5.0):
            time.sleep(0.5)  # hold for 500ms

    def waiter():
        time.sleep(0.05)  # let holder acquire first
        try:
            with timed_lock(lock, "waiter", timeout=0.1):
                results.append("acquired")
        except LockTimeoutError:
            results.append("timeout")

    t1 = threading.Thread(target=holder)
    t2 = threading.Thread(target=waiter)
    t1.start()
    t2.start()
    t2.join(timeout=2)
    t1.join(timeout=2)

    assert results == ["timeout"]


def test_lock_timeout_error_attributes():
    """LockTimeoutError carries lock_name and timeout for debugging."""
    err = LockTimeoutError("my_cache", 30.0)
    assert err.lock_name == "my_cache"
    assert err.timeout == 30.0
    assert "my_cache" in str(err)
    assert "30" in str(err)
    assert "AGENTFIELD_LOCK_TIMEOUT_SECONDS" in str(err)


def test_lock_timeout_error_is_not_a_timeout_error():
    """On 3.11+ `except asyncio.TimeoutError` also catches TimeoutError.

    Deriving from it would let the wait_for wrappers around client.execute()
    swallow a real deadlock and report a generic execution timeout instead.
    """
    assert issubclass(LockTimeoutError, RuntimeError)
    assert not issubclass(LockTimeoutError, TimeoutError)
    assert not issubclass(LockTimeoutError, asyncio.TimeoutError)


def test_default_timeout_is_reasonable():
    """Default timeout should be > 0 and configurable."""
    assert DEFAULT_LOCK_TIMEOUT > 0
    assert DEFAULT_LOCK_TIMEOUT <= 300  # not absurdly high


def _import_time_default_timeout(env_value):
    """Import agentfield.lock_utils in a subprocess and read DEFAULT_LOCK_TIMEOUT.

    DEFAULT_LOCK_TIMEOUT is resolved at import time, so a fresh interpreter is
    the only way to exercise the parsing that actually ships.
    """
    env = dict(os.environ)
    if env_value is None:
        env.pop("AGENTFIELD_LOCK_TIMEOUT_SECONDS", None)
    else:
        env["AGENTFIELD_LOCK_TIMEOUT_SECONDS"] = env_value

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import agentfield.lock_utils as m; "
            "print('DEFAULT_LOCK_TIMEOUT', m.DEFAULT_LOCK_TIMEOUT)",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, (
        f"import failed for AGENTFIELD_LOCK_TIMEOUT_SECONDS={env_value!r}: "
        f"{completed.stderr}"
    )
    for line in completed.stdout.splitlines():
        if line.startswith("DEFAULT_LOCK_TIMEOUT "):
            return float(line.split(" ", 1)[1])
    raise AssertionError(f"no value printed; stdout was: {completed.stdout!r}")


@pytest.mark.parametrize(
    "env_value,expected",
    [
        (None, FALLBACK_LOCK_TIMEOUT),  # unset
        ("", FALLBACK_LOCK_TIMEOUT),  # `AGENTFIELD_LOCK_TIMEOUT_SECONDS=` in compose
        ("   ", FALLBACK_LOCK_TIMEOUT),
        ("abc", FALLBACK_LOCK_TIMEOUT),  # not a number
        ("-5", FALLBACK_LOCK_TIMEOUT),  # lock.acquire() rejects negative timeouts
        ("0", FALLBACK_LOCK_TIMEOUT),
        ("inf", FALLBACK_LOCK_TIMEOUT),  # lock.acquire() rejects it as too large
        ("12.5", 12.5),  # valid override survives
    ],
)
def test_import_never_fails_on_bad_env(env_value, expected):
    """A malformed env var must not take down `import agentfield`."""
    assert _import_time_default_timeout(env_value) == expected


class _SubmissionBlocked(Exception):
    """Raised by the stubbed submit hook to prove no request was ever sent."""


def test_execute_sync_warns_in_running_loop():
    """execute_sync() emits RuntimeWarning when called from a running loop thread."""
    from agentfield.client import AgentFieldClient
    from agentfield.async_config import AsyncConfig

    client = AgentFieldClient.__new__(AgentFieldClient)
    client.api_base = "http://localhost:8080/api/v1"
    client.api_key = None
    client.caller_agent_id = None
    client.async_config = AsyncConfig()

    def _blocked(*args, **kwargs):
        raise _SubmissionBlocked("execute_sync must not reach the control plane")

    # Without this the call would POST to api_base and start polling — on a
    # dev box with a control plane up, the suite would submit a live execution.
    client._submit_execution_sync = _blocked

    # We can't call execute_sync from an async def (that would deadlock), so
    # drive it from the loop thread and let the stub stop it at submission.
    async def main():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with pytest.raises(_SubmissionBlocked):
                client.execute_sync("test.reasoner", {"input": "x"})
            return [w for w in caught if issubclass(w.category, RuntimeWarning)]

    result = asyncio.run(main())
    assert len(result) >= 1
    assert "execute_sync" in str(result[0].message)
    assert "running event loop" in str(result[0].message)
