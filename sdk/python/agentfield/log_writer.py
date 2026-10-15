"""Bounded, non-blocking stdout writer for SDK log lines."""

from __future__ import annotations

import asyncio
import atexit
import json
import math
import os
import queue
import sys
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional, TextIO, Tuple, cast


_DEFAULT_QUEUE_SIZE = 1024
_DEFAULT_FLUSH_SECONDS = 2.0
_FALSE_VALUES = ("0", "false", "no", "off")

_state_lock = threading.Lock()
_queue: "queue.Queue[Optional[Tuple[TextIO, str]]]" = queue.Queue(
    maxsize=_DEFAULT_QUEUE_SIZE
)
_worker: Optional[threading.Thread] = None
_dropped: Dict[TextIO, int] = {}
_active = False


def _queue_enabled() -> bool:
    value = os.getenv("AGENTFIELD_LOG_QUEUE", "true").strip().lower()
    return value not in _FALSE_VALUES


def _queue_size() -> int:
    raw = os.getenv("AGENTFIELD_LOG_QUEUE_SIZE", str(_DEFAULT_QUEUE_SIZE))
    try:
        return max(1, int(raw, 10))
    except ValueError:
        return _DEFAULT_QUEUE_SIZE


def _flush_seconds() -> float:
    raw = os.getenv("AGENTFIELD_LOG_QUEUE_FLUSH_SECONDS", str(_DEFAULT_FLUSH_SECONDS))
    try:
        value = float(raw)
    except ValueError:
        return _DEFAULT_FLUSH_SECONDS
    if not math.isfinite(value):
        return _DEFAULT_FLUSH_SECONDS
    return max(0.0, value)


def _has_real_fileno(stream: TextIO) -> bool:
    try:
        return isinstance(stream.fileno(), int)
    except Exception:
        return False


def _has_running_loop() -> bool:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


def _write_line(stream: TextIO, line: str) -> bool:
    try:
        stream.write(line + "\n")
        stream.flush()
    except Exception:
        return False
    return True


def _drop_marker(count: int) -> str:
    timestamp = (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )
    return json.dumps(
        {
            "ts": timestamp,
            "level": "warning",
            "source": "sdk.python.logger",
            "event_type": "log.dropped",
            "message": (f"dropped {count} structured log lines (stdout back-pressure)"),
            "attributes": {"dropped": count},
            "system_generated": True,
        },
        separators=(",", ":"),
    )


def _worker_main(
    work_queue: "queue.Queue[Optional[Tuple[TextIO, str]]]",
) -> None:
    global _active, _worker

    current = threading.current_thread()
    try:
        while True:
            item = work_queue.get()
            if item is None:
                work_queue.task_done()
                return

            stream, line = item
            dropped = 0
            with _state_lock:
                _active = True
                dropped = _dropped.pop(stream, 0)

            try:
                if dropped and not _write_line(stream, _drop_marker(dropped)):
                    with _state_lock:
                        _dropped[stream] = _dropped.get(stream, 0) + dropped
                _write_line(stream, line)
            except Exception:
                # The worker is deliberately immortal for ordinary failures.
                pass
            finally:
                with _state_lock:
                    _active = False
                work_queue.task_done()
    except Exception:
        # Queue and marker failures must not escape the daemon thread either.
        pass
    finally:
        with _state_lock:
            _active = False
            if _worker is current:
                _worker = None


def _start_worker_locked(
    work_queue: "queue.Queue[Optional[Tuple[TextIO, str]]]",
) -> bool:
    global _worker

    if _worker is not None and _worker.is_alive():
        return True

    worker = threading.Thread(
        target=_worker_main,
        args=(work_queue,),
        name="agentfield-log-writer",
        daemon=True,
    )
    _worker = worker
    try:
        worker.start()
    except Exception:
        _worker = None
        return False
    return True


def _refresh_queue_locked() -> None:
    global _queue

    configured_size = _queue_size()
    worker_alive = _worker is not None and _worker.is_alive()
    if (
        not worker_alive
        and _queue.unfinished_tasks == 0
        and _queue.maxsize != configured_size
    ):
        _queue = queue.Queue(maxsize=configured_size)


def emit_line(line: str, stream: Optional[TextIO] = None) -> None:
    """Write one line, deferring real-fd writes made from an event loop."""
    resolved_stream = stream if stream is not None else cast(TextIO, sys.stdout)
    if not _queue_enabled() or not _has_real_fileno(resolved_stream):
        _write_line(resolved_stream, line)
        return

    running_loop = _has_running_loop()
    queued = False
    with _state_lock:
        _refresh_queue_locked()
        work_queue = _queue
        backlog = work_queue.unfinished_tasks > 0
        if running_loop or backlog:
            if not _start_worker_locked(work_queue):
                if running_loop:
                    _dropped[resolved_stream] = _dropped.get(resolved_stream, 0) + 1
            else:
                try:
                    work_queue.put_nowait((resolved_stream, line))
                    queued = True
                except queue.Full:
                    try:
                        evicted = work_queue.get_nowait()
                    except queue.Empty:
                        pass
                    else:
                        work_queue.task_done()
                        if evicted is not None:
                            evicted_stream, _evicted_line = evicted
                            _dropped[evicted_stream] = (
                                _dropped.get(evicted_stream, 0) + 1
                            )
                    try:
                        work_queue.put_nowait((resolved_stream, line))
                        queued = True
                    except queue.Full:
                        _dropped[resolved_stream] = _dropped.get(resolved_stream, 0) + 1

    if queued:
        return
    if not running_loop:
        _write_line(resolved_stream, line)


def flush(timeout: Optional[float] = None) -> bool:
    """Wait until queued and in-progress lines are written."""
    with _state_lock:
        work_queue = _queue
        if work_queue.unfinished_tasks and not _start_worker_locked(work_queue):
            return False

    deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
    with work_queue.all_tasks_done:
        while work_queue.unfinished_tasks:
            if deadline is None:
                work_queue.all_tasks_done.wait()
                continue
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            work_queue.all_tasks_done.wait(remaining)
    return True


def stats() -> Dict[str, Any]:
    """Return a snapshot of the writer's queue and worker state."""
    with _state_lock:
        return {
            "pending": _queue.qsize(),
            "dropped": sum(_dropped.values()),
            "worker_alive": _worker is not None and _worker.is_alive(),
            "active": _active,
        }


def shutdown(timeout: Optional[float] = None) -> None:
    """Drain queued lines and stop the worker, primarily for test isolation."""
    if not flush(timeout):
        return

    with _state_lock:
        worker = _worker
        work_queue = _queue
        if worker is None or not worker.is_alive():
            return
        try:
            work_queue.put_nowait(None)
        except queue.Full:
            return

    if worker is not threading.current_thread():
        worker.join(timeout)


def _flush_at_exit() -> None:
    try:
        flush(_flush_seconds())
    except Exception:
        pass


def _before_fork() -> None:
    _state_lock.acquire()


def _after_fork_parent() -> None:
    _state_lock.release()


def _after_fork_child() -> None:
    global _active, _dropped, _queue, _worker

    _queue = queue.Queue(maxsize=_queue_size())
    _worker = None
    _dropped = {}
    _active = False
    _state_lock.release()


atexit.register(_flush_at_exit)
if hasattr(os, "register_at_fork"):
    os.register_at_fork(
        before=_before_fork,
        after_in_parent=_after_fork_parent,
        after_in_child=_after_fork_child,
    )
