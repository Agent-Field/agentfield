"""
Bounded in-memory process log capture (stdout/stderr) and HTTP NDJSON API for AgentField.
"""

from __future__ import annotations

import json
import os
import queue
import secrets
import sys
import threading
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Deque, Iterator, List, Optional, TextIO

# Optional: follow subscribers wake on new lines
_follow_queues: List["queue.Queue[None]"] = []
_follow_lock = threading.Lock()


def _notify_followers() -> None:
    with _follow_lock:
        for q in _follow_queues:
            try:
                q.put_nowait(None)
            except queue.Full:
                pass


def register_follow_queue(q: "queue.Queue[None]") -> None:
    with _follow_lock:
        _follow_queues.append(q)


def unregister_follow_queue(q: "queue.Queue[None]") -> None:
    with _follow_lock:
        try:
            _follow_queues.remove(q)
        except ValueError:
            pass


@dataclass
class LogEntry:
    seq: int
    ts: str
    stream: str
    line: str
    truncated: bool = False

    def to_ndjson_line(self) -> bytes:
        obj = {
            "v": 1,
            "seq": self.seq,
            "ts": self.ts,
            "stream": self.stream,
            "line": self.line,
            "source": "process",
        }
        sl = self.stream.lower()
        if sl == "stderr":
            obj["level"] = "error"
        elif sl == "stdout":
            obj["level"] = "info"
        else:
            obj["level"] = "log"
        if self.truncated:
            obj["truncated"] = True
        return (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")


class ProcessLogRing:
    """
    Thread-safe ring buffer capped by total byte size of stored line text.
    """

    def __init__(self, max_bytes: int) -> None:
        self._max_bytes = max(1024, max_bytes)
        self._lock = threading.Lock()
        self._seq = 0
        self._entries: Deque[LogEntry] = deque()
        self._approx_bytes = 0

    def append(self, stream: str, text: str, max_line_bytes: int) -> None:
        raw = text
        truncated = False
        if len(raw.encode("utf-8")) > max_line_bytes:
            raw_bytes = raw.encode("utf-8")[:max_line_bytes]
            raw = raw_bytes.decode("utf-8", errors="replace")
            truncated = True
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        with self._lock:
            self._seq += 1
            entry = LogEntry(seq=self._seq, ts=ts, stream=stream, line=raw, truncated=truncated)
            self._entries.append(entry)
            self._approx_bytes += len(entry.line.encode("utf-8")) + 64
            while self._approx_bytes > self._max_bytes and len(self._entries) > 1:
                old = self._entries.popleft()
                self._approx_bytes -= len(old.line.encode("utf-8")) + 64
        _notify_followers()

    def snapshot_after(self, since_seq: int, limit: Optional[int] = None) -> List[LogEntry]:
        with self._lock:
            items = [e for e in self._entries if e.seq > since_seq]
            if limit is not None and limit > 0:
                items = items[-limit:]
            return list(items)

    def tail(self, n: int) -> List[LogEntry]:
        with self._lock:
            if n <= 0:
                return []
            return list(self._entries)[-n:]

    def max_seq(self) -> int:
        with self._lock:
            return self._seq


class _TeeTextIO(TextIO):
    """Write-through to original stream and log ring (line-buffered by \\n)."""

    def __init__(
        self,
        stream_name: str,
        original: TextIO,
        ring: ProcessLogRing,
        max_line_bytes: int,
    ) -> None:
        self._stream_name = stream_name
        self._original = original
        self._ring = ring
        self._max_line_bytes = max_line_bytes
        self._buf = ""

    def write(self, s: str) -> int:
        if not s:
            return 0
        self._original.write(s)
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line or True:
                self._ring.append(self._stream_name, line, self._max_line_bytes)
        return len(s)

    def flush(self) -> None:
        self._original.flush()

    # Minimal TextIO protocol for print()
    @property
    def encoding(self) -> str:
        return getattr(self._original, "encoding", "utf-8") or "utf-8"

    def isatty(self) -> bool:
        return bool(self._original.isatty())


_global_ring: Optional[ProcessLogRing] = None
_tee_installed = False


def logs_enabled() -> bool:
    v = os.getenv("AGENTFIELD_LOGS_ENABLED", "true").strip().lower()
    return v not in ("0", "false", "no", "off")


def buffer_max_bytes() -> int:
    raw = os.getenv("AGENTFIELD_LOG_BUFFER_BYTES", "4194304")
    try:
        return max(1024, int(raw, 10))
    except ValueError:
        return 4194304


def max_line_bytes() -> int:
    raw = os.getenv("AGENTFIELD_LOG_MAX_LINE_BYTES", "16384")
    try:
        return max(256, int(raw, 10))
    except ValueError:
        return 16384


def get_ring() -> ProcessLogRing:
    global _global_ring
    if _global_ring is None:
        _global_ring = ProcessLogRing(buffer_max_bytes())
    return _global_ring


def install_stdio_tee() -> None:
    """Replace sys.stdout/sys.stderr with tees into the process log ring."""
    global _tee_installed
    if _tee_installed or not logs_enabled():
        return
    ring = get_ring()
    ml = max_line_bytes()
    sys.stdout = _TeeTextIO("stdout", sys.__stdout__, ring, ml)
    sys.stderr = _TeeTextIO("stderr", sys.__stderr__, ring, ml)
    _tee_installed = True


def verify_internal_bearer(authorization_header: Optional[str]) -> bool:
    token = os.getenv("AGENTFIELD_AUTHORIZATION_INTERNAL_TOKEN", "").strip()
    if not token:
        # Dev: allow when agent has no internal token configured (insecure).
        return True
    if not authorization_header or not authorization_header.startswith("Bearer "):
        return False
    got = authorization_header[7:].strip()
    return secrets.compare_digest(got, token)


def iter_tail_ndjson(
    tail_lines: int,
    since_seq: int,
    follow: bool,
) -> Iterator[bytes]:
    ring = get_ring()
    cap_tail = tail_lines
    if since_seq > 0:
        entries = ring.snapshot_after(since_seq, limit=cap_tail if cap_tail > 0 else None)
    else:
        n = cap_tail if cap_tail > 0 else 200
        entries = ring.tail(n)
    for e in entries:
        yield e.to_ndjson_line()
    if not follow:
        return
    q: "queue.Queue[None]" = queue.Queue(maxsize=8)
    register_follow_queue(q)
    last_seq = entries[-1].seq if entries else since_seq
    try:
        # Run until client disconnects (generator closed); CP also enforces max duration.
        while True:
            try:
                q.get(timeout=0.5)
            except queue.Empty:
                pass
            newer = ring.snapshot_after(last_seq, limit=None)
            for e in newer:
                yield e.to_ndjson_line()
                last_seq = e.seq
    finally:
        unregister_follow_queue(q)
