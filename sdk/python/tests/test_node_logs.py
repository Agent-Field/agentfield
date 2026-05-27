"""
Tests for agentfield.node_logs — ProcessLogRing and related helpers.
"""

from __future__ import annotations

import io
import json
import queue
import sys
import threading

import pytest

from agentfield.node_logs import (
    LogEntry,
    ProcessLogRing,
    _TeeTextIO,
    get_ring,
    install_stdio_tee,
    iter_tail_ndjson,
    verify_internal_bearer,
)


# ---------------------------------------------------------------------------
# LogEntry NDJSON serialization
# ---------------------------------------------------------------------------


class TestLogEntryNdjson:
    def test_stdout_produces_info_level(self):
        entry = LogEntry(
            seq=1, ts="2024-01-01T00:00:00.000Z", stream="stdout", line="hello"
        )
        data = json.loads(entry.to_ndjson_line().decode())
        assert data["level"] == "info"
        assert data["line"] == "hello"
        assert data["v"] == 1
        assert data["source"] == "process"

    def test_stderr_produces_error_level(self):
        entry = LogEntry(
            seq=2, ts="2024-01-01T00:00:00.000Z", stream="stderr", line="err"
        )
        data = json.loads(entry.to_ndjson_line().decode())
        assert data["level"] == "error"

    def test_other_stream_produces_log_level(self):
        entry = LogEntry(
            seq=3, ts="2024-01-01T00:00:00.000Z", stream="custom", line="msg"
        )
        data = json.loads(entry.to_ndjson_line().decode())
        assert data["level"] == "log"

    def test_truncated_flag_included_when_true(self):
        entry = LogEntry(seq=1, ts="ts", stream="stdout", line="x", truncated=True)
        data = json.loads(entry.to_ndjson_line().decode())
        assert data["truncated"] is True

    def test_truncated_not_included_when_false(self):
        entry = LogEntry(seq=1, ts="ts", stream="stdout", line="x", truncated=False)
        data = json.loads(entry.to_ndjson_line().decode())
        assert "truncated" not in data

    def test_ndjson_ends_with_newline(self):
        entry = LogEntry(seq=1, ts="ts", stream="stdout", line="x")
        assert entry.to_ndjson_line().endswith(b"\n")

    def test_seq_and_ts_preserved(self):
        entry = LogEntry(
            seq=42, ts="2024-06-15T10:00:00.000Z", stream="stdout", line="data"
        )
        data = json.loads(entry.to_ndjson_line().decode())
        assert data["seq"] == 42
        assert data["ts"] == "2024-06-15T10:00:00.000Z"


# ---------------------------------------------------------------------------
# ProcessLogRing — basic append and tail
# ---------------------------------------------------------------------------


class TestProcessLogRingBasic:
    def test_empty_ring_tail_returns_empty(self):
        ring = ProcessLogRing(max_bytes=1024 * 1024)
        assert ring.tail(10) == []

    def test_empty_ring_tail_zero_returns_empty(self):
        ring = ProcessLogRing(max_bytes=1024 * 1024)
        assert ring.tail(0) == []

    def test_append_single_entry(self):
        ring = ProcessLogRing(max_bytes=1024 * 1024)
        ring.append("stdout", "hello", max_line_bytes=1024)
        entries = ring.tail(10)
        assert len(entries) == 1
        assert entries[0].line == "hello"
        assert entries[0].stream == "stdout"
        assert entries[0].seq == 1

    def test_seq_increments_monotonically(self):
        ring = ProcessLogRing(max_bytes=1024 * 1024)
        ring.append("stdout", "a", 1024)
        ring.append("stdout", "b", 1024)
        ring.append("stdout", "c", 1024)
        entries = ring.tail(10)
        seqs = [e.seq for e in entries]
        assert seqs == [1, 2, 3]

    def test_tail_returns_last_n(self):
        ring = ProcessLogRing(max_bytes=1024 * 1024)
        for i in range(10):
            ring.append("stdout", f"line{i}", 1024)
        entries = ring.tail(3)
        assert len(entries) == 3
        assert entries[-1].line == "line9"

    def test_max_seq_reflects_appends(self):
        ring = ProcessLogRing(max_bytes=1024 * 1024)
        assert ring.max_seq() == 0
        ring.append("stdout", "a", 1024)
        assert ring.max_seq() == 1
        ring.append("stdout", "b", 1024)
        assert ring.max_seq() == 2


# ---------------------------------------------------------------------------
# ProcessLogRing — byte cap eviction
# ---------------------------------------------------------------------------


class TestProcessLogRingEviction:
    def test_ring_evicts_old_entries_when_full(self):
        # Use tiny max_bytes so we force eviction quickly
        ring = ProcessLogRing(max_bytes=1024)
        big_line = "x" * 200  # ~200 bytes per entry + 64 overhead
        for i in range(20):
            ring.append("stdout", big_line, max_line_bytes=512)
        # Ring should have fewer than 20 entries
        entries = ring.tail(100)
        assert len(entries) < 20
        assert len(entries) >= 1  # always keeps at least 1

    def test_ring_keeps_at_least_one_entry(self):
        ring = ProcessLogRing(max_bytes=1024)
        # Even a line that exceeds the ring's capacity should be kept (1 entry minimum)
        ring.append("stdout", "x" * 2000, max_line_bytes=4096)
        entries = ring.tail(10)
        assert len(entries) == 1

    def test_evicted_entries_have_higher_seq(self):
        ring = ProcessLogRing(max_bytes=1024)
        big_line = "y" * 200
        for i in range(20):
            ring.append("stdout", big_line, 512)
        entries = ring.tail(100)
        # All remaining entries should be the most recent (highest seqs)
        max_seq = ring.max_seq()
        assert entries[-1].seq == max_seq


# ---------------------------------------------------------------------------
# ProcessLogRing — line truncation
# ---------------------------------------------------------------------------


class TestProcessLogRingTruncation:
    def test_long_line_is_truncated(self):
        ring = ProcessLogRing(max_bytes=1024 * 1024)
        long_text = "a" * 500
        ring.append("stdout", long_text, max_line_bytes=10)
        entries = ring.tail(1)
        assert entries[0].truncated is True
        assert (
            len(entries[0].line.encode("utf-8")) <= 10 + 3
        )  # allow for replacement chars

    def test_short_line_is_not_truncated(self):
        ring = ProcessLogRing(max_bytes=1024 * 1024)
        ring.append("stdout", "hello", max_line_bytes=1024)
        entries = ring.tail(1)
        assert entries[0].truncated is False
        assert entries[0].line == "hello"


# ---------------------------------------------------------------------------
# ProcessLogRing — snapshot_after
# ---------------------------------------------------------------------------


class TestProcessLogRingSnapshotAfter:
    def test_snapshot_after_returns_entries_after_seq(self):
        ring = ProcessLogRing(max_bytes=1024 * 1024)
        for i in range(5):
            ring.append("stdout", f"msg{i}", 1024)
        entries = ring.snapshot_after(since_seq=2)
        seqs = [e.seq for e in entries]
        assert all(s > 2 for s in seqs)
        assert len(entries) == 3

    def test_snapshot_after_with_limit(self):
        ring = ProcessLogRing(max_bytes=1024 * 1024)
        for i in range(10):
            ring.append("stdout", f"msg{i}", 1024)
        entries = ring.snapshot_after(since_seq=0, limit=3)
        assert len(entries) == 3

    def test_snapshot_after_seq_zero_returns_all(self):
        ring = ProcessLogRing(max_bytes=1024 * 1024)
        for i in range(5):
            ring.append("stdout", f"msg{i}", 1024)
        entries = ring.snapshot_after(since_seq=0)
        assert len(entries) == 5

    def test_snapshot_after_high_seq_returns_empty(self):
        ring = ProcessLogRing(max_bytes=1024 * 1024)
        ring.append("stdout", "only", 1024)
        entries = ring.snapshot_after(since_seq=999)
        assert entries == []


# ---------------------------------------------------------------------------
# ProcessLogRing — thread safety
# ---------------------------------------------------------------------------


class TestProcessLogRingThreadSafety:
    def test_concurrent_appends_consistent(self):
        ring = ProcessLogRing(max_bytes=10 * 1024 * 1024)
        errors = []

        def writer(stream_id):
            try:
                for i in range(50):
                    ring.append(f"stream{stream_id}", f"line{i}", 1024)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"
        entries = ring.tail(1000)
        assert len(entries) <= 250  # 5 threads x 50 entries
        assert len(entries) >= 1


# ---------------------------------------------------------------------------
# iter_tail_ndjson — no-follow mode
# ---------------------------------------------------------------------------


class TestIterTailNdjson:
    def test_iter_tail_returns_last_n_as_ndjson(self, monkeypatch):
        import agentfield.node_logs as nl

        ring = ProcessLogRing(max_bytes=1024 * 1024)
        for i in range(5):
            ring.append("stdout", f"line{i}", 1024)
        monkeypatch.setattr(nl, "_global_ring", ring)

        chunks = list(iter_tail_ndjson(tail_lines=3, since_seq=0, follow=False))
        assert len(chunks) == 3
        for chunk in chunks:
            data = json.loads(chunk.decode())
            assert "line" in data

    def test_iter_tail_since_seq_filters(self, monkeypatch):
        import agentfield.node_logs as nl

        ring = ProcessLogRing(max_bytes=1024 * 1024)
        for i in range(5):
            ring.append("stdout", f"line{i}", 1024)
        monkeypatch.setattr(nl, "_global_ring", ring)

        chunks = list(iter_tail_ndjson(tail_lines=0, since_seq=3, follow=False))
        for chunk in chunks:
            data = json.loads(chunk.decode())
            assert data["seq"] > 3

    def test_iter_tail_empty_ring(self, monkeypatch):
        import agentfield.node_logs as nl

        ring = ProcessLogRing(max_bytes=1024 * 1024)
        monkeypatch.setattr(nl, "_global_ring", ring)

        chunks = list(iter_tail_ndjson(tail_lines=10, since_seq=0, follow=False))
        assert chunks == []


# ---------------------------------------------------------------------------
# _TeeTextIO and install_stdio_tee
# ---------------------------------------------------------------------------


class TestTeeTextIO:
    def test_tee_text_io_writes_to_original(self):
        original = io.StringIO()
        ring = ProcessLogRing(max_bytes=1024 * 1024)
        tee = _TeeTextIO("stdout", original, ring, max_line_bytes=1024)

        written = tee.write("hello\n")

        assert written == len("hello\n")
        assert original.getvalue() == "hello\n"

    def test_tee_text_io_appends_to_ring(self):
        original = io.StringIO()
        ring = ProcessLogRing(max_bytes=1024 * 1024)
        tee = _TeeTextIO("stdout", original, ring, max_line_bytes=1024)

        tee.write("one line\n")

        entries = ring.tail(1)
        assert len(entries) == 1
        assert entries[0].stream == "stdout"
        assert entries[0].line == "one line"

    def test_tee_text_io_buffers_until_newline(self):
        original = io.StringIO()
        ring = ProcessLogRing(max_bytes=1024 * 1024)
        tee = _TeeTextIO("stderr", original, ring, max_line_bytes=1024)

        tee.write("partial")
        assert ring.tail(1) == []

        tee.write(" line\n")
        entries = ring.tail(1)
        assert entries[0].stream == "stderr"
        assert entries[0].line == "partial line"

    def test_installed_tee_exposes_text_io_methods(self, monkeypatch):
        import agentfield.node_logs as nl

        class TextStream(io.StringIO):
            def fileno(self):
                return 42

        previous_stdout = sys.stdout
        previous_stderr = sys.stderr
        original_stdout = TextStream()
        original_stderr = TextStream()
        ring = ProcessLogRing(max_bytes=1024 * 1024)

        monkeypatch.setenv("AGENTFIELD_LOGS_ENABLED", "true")
        monkeypatch.setattr(sys, "__stdout__", original_stdout)
        monkeypatch.setattr(sys, "__stderr__", original_stderr)
        monkeypatch.setattr(nl, "_global_ring", ring)
        monkeypatch.setattr(nl, "_tee_installed", False)

        try:
            install_stdio_tee()
            assert isinstance(sys.stdout, _TeeTextIO)
            assert sys.stdout.fileno() == 42
            assert sys.stdout.readable() is True
            assert sys.stdout.writable() is True
            assert sys.stdout.seekable() is True

            sys.stdout.writelines(["first\n", "second\n"])
            assert original_stdout.getvalue() == "first\nsecond\n"
            assert [entry.line for entry in ring.tail(2)] == ["first", "second"]

            sys.stdout.write("partial")
            sys.stdout.close()
            assert original_stdout.closed is False
            assert ring.tail(1)[0].line == "partial"
            original_stdout.write(" still usable")
            assert original_stdout.getvalue().endswith("partial still usable")
        finally:
            sys.stdout = previous_stdout
            sys.stderr = previous_stderr
            nl._tee_installed = False

    def test_install_stdio_tee_replaces_sys_stdout(self, monkeypatch):
        import agentfield.node_logs as nl

        previous_stdout = sys.stdout
        previous_stderr = sys.stderr
        original_stdout = io.StringIO()
        original_stderr = io.StringIO()
        ring = ProcessLogRing(max_bytes=1024 * 1024)

        monkeypatch.setenv("AGENTFIELD_LOGS_ENABLED", "true")
        monkeypatch.setattr(sys, "__stdout__", original_stdout)
        monkeypatch.setattr(sys, "__stderr__", original_stderr)
        monkeypatch.setattr(nl, "_global_ring", ring)
        monkeypatch.setattr(nl, "_tee_installed", False)

        try:
            install_stdio_tee()
            assert isinstance(sys.stdout, _TeeTextIO)
            assert isinstance(sys.stderr, _TeeTextIO)
            first_stdout = sys.stdout
            install_stdio_tee()
            assert sys.stdout is first_stdout
            assert sys.stdout._original is original_stdout

            sys.stdout.write("captured\n")
            assert original_stdout.getvalue() == "captured\n"
            assert ring.tail(1)[0].line == "captured"
        finally:
            sys.stdout = previous_stdout
            sys.stderr = previous_stderr
            nl._tee_installed = False

    def test_install_stdio_tee_disabled_env_leaves_streams_unchanged(self, monkeypatch):
        import agentfield.node_logs as nl

        previous_stdout = sys.stdout
        previous_stderr = sys.stderr
        original_stdout = io.StringIO()
        original_stderr = io.StringIO()

        monkeypatch.setenv("AGENTFIELD_LOGS_ENABLED", "false")
        monkeypatch.setattr(sys, "__stdout__", original_stdout)
        monkeypatch.setattr(sys, "__stderr__", original_stderr)
        monkeypatch.setattr(nl, "_global_ring", ProcessLogRing(max_bytes=1024 * 1024))
        monkeypatch.setattr(nl, "_tee_installed", False)

        install_stdio_tee()

        assert sys.stdout is previous_stdout
        assert sys.stderr is previous_stderr
        assert nl._tee_installed is False


class TestIterTailNdjsonFollow:
    def test_iter_tail_ndjson_follow_mode(self, monkeypatch):
        import agentfield.node_logs as nl

        ring = ProcessLogRing(max_bytes=1024 * 1024)
        monkeypatch.setattr(nl, "_global_ring", ring)
        monkeypatch.setattr(nl, "_follow_queues", [])
        queue_registered = threading.Event()
        original_register_follow_queue = nl.register_follow_queue

        def register_follow_queue(q):
            original_register_follow_queue(q)
            queue_registered.set()

        monkeypatch.setattr(nl, "register_follow_queue", register_follow_queue)

        chunks: list[bytes] = []
        errors: list[BaseException] = []
        generator = iter_tail_ndjson(tail_lines=0, since_seq=0, follow=True)

        def read_next():
            try:
                chunks.append(next(generator))
            except Exception as exc:  # pragma: no cover - assertion reports details
                errors.append(exc)

        thread = threading.Thread(target=read_next)
        thread.start()
        assert queue_registered.wait(timeout=2)

        ring.append("stdout", "new log", max_line_bytes=1024)
        thread.join(timeout=2)
        generator.close()

        assert errors == []
        assert len(chunks) == 1
        assert json.loads(chunks[0].decode())["line"] == "new log"

    def test_iter_tail_ndjson_follow_emits_tail_then_new_entries(self, monkeypatch):
        import agentfield.node_logs as nl

        ring = ProcessLogRing(max_bytes=1024 * 1024)
        for i in range(3):
            ring.append("stdout", f"line{i}", max_line_bytes=1024)
        monkeypatch.setattr(nl, "_global_ring", ring)
        monkeypatch.setattr(nl, "_follow_queues", [])
        queue_registered = threading.Event()
        original_register_follow_queue = nl.register_follow_queue

        def register_follow_queue(q):
            original_register_follow_queue(q)
            queue_registered.set()

        monkeypatch.setattr(nl, "register_follow_queue", register_follow_queue)

        generator = iter_tail_ndjson(tail_lines=2, since_seq=0, follow=True)
        prelude = [json.loads(next(generator).decode()) for _ in range(2)]
        chunks: list[bytes] = []
        errors: list[BaseException] = []

        def read_next():
            try:
                chunks.append(next(generator))
            except Exception as exc:  # pragma: no cover - assertion reports details
                errors.append(exc)

        thread = threading.Thread(target=read_next)
        thread.start()
        assert queue_registered.wait(timeout=2)

        ring.append("stdout", "followed", max_line_bytes=1024)
        thread.join(timeout=2)
        generator.close()

        assert [entry["line"] for entry in prelude] == ["line1", "line2"]
        assert errors == []
        assert len(chunks) == 1
        assert json.loads(chunks[0].decode())["line"] == "followed"

    def test_iter_tail_ndjson_unregisters_on_close(self, monkeypatch):
        import agentfield.node_logs as nl

        class ClosingQueue:
            def __init__(self, maxsize: int) -> None:
                self.maxsize = maxsize

            def put_nowait(self, _item):
                return None

            def get(self, timeout: float):
                assert timeout == 0.5
                raise GeneratorExit

        ring = ProcessLogRing(max_bytes=1024 * 1024)
        monkeypatch.setattr(nl, "_global_ring", ring)
        monkeypatch.setattr(nl, "_follow_queues", [])
        monkeypatch.setattr(nl.queue, "Queue", ClosingQueue)

        generator = iter_tail_ndjson(tail_lines=0, since_seq=0, follow=True)
        with pytest.raises(GeneratorExit):
            next(generator)

        assert nl._follow_queues == []

    def test_iter_tail_ndjson_queue_timeout(self, monkeypatch):
        import agentfield.node_logs as nl

        ring = ProcessLogRing(max_bytes=1024 * 1024)

        class TimeoutQueue:
            def __init__(self, maxsize: int) -> None:
                self.maxsize = maxsize
                self._appended = False

            def put_nowait(self, _item):
                return None

            def get(self, timeout: float):
                assert timeout == 0.5
                if not self._appended:
                    self._appended = True
                    ring.append("stdout", "after timeout", max_line_bytes=1024)
                raise queue.Empty

        monkeypatch.setattr(nl, "_global_ring", ring)
        monkeypatch.setattr(nl, "_follow_queues", [])
        monkeypatch.setattr(nl.queue, "Queue", TimeoutQueue)

        generator = iter_tail_ndjson(tail_lines=0, since_seq=0, follow=True)
        try:
            chunk = next(generator)
        finally:
            generator.close()

        assert json.loads(chunk.decode())["line"] == "after timeout"


# ---------------------------------------------------------------------------
# verify_internal_bearer
# ---------------------------------------------------------------------------


class TestVerifyInternalBearer:
    def test_allows_when_no_token_configured(self, monkeypatch):
        monkeypatch.delenv("AGENTFIELD_AUTHORIZATION_INTERNAL_TOKEN", raising=False)
        assert verify_internal_bearer("Bearer anything") is True

    def test_allows_correct_bearer_token(self, monkeypatch):
        monkeypatch.setenv("AGENTFIELD_AUTHORIZATION_INTERNAL_TOKEN", "secret123")
        assert verify_internal_bearer("Bearer secret123") is True

    def test_rejects_wrong_bearer_token(self, monkeypatch):
        monkeypatch.setenv("AGENTFIELD_AUTHORIZATION_INTERNAL_TOKEN", "secret123")
        assert verify_internal_bearer("Bearer wrongtoken") is False

    def test_rejects_missing_bearer_prefix(self, monkeypatch):
        monkeypatch.setenv("AGENTFIELD_AUTHORIZATION_INTERNAL_TOKEN", "secret123")
        assert verify_internal_bearer("secret123") is False

    def test_rejects_none_header(self, monkeypatch):
        monkeypatch.setenv("AGENTFIELD_AUTHORIZATION_INTERNAL_TOKEN", "secret123")
        assert verify_internal_bearer(None) is False

    def test_rejects_empty_header(self, monkeypatch):
        monkeypatch.setenv("AGENTFIELD_AUTHORIZATION_INTERNAL_TOKEN", "secret123")
        assert verify_internal_bearer("") is False


# ---------------------------------------------------------------------------
# get_ring singleton
# ---------------------------------------------------------------------------


class TestGetRing:
    def test_get_ring_returns_process_log_ring(self, monkeypatch):
        import agentfield.node_logs as nl

        monkeypatch.setattr(nl, "_global_ring", None)
        ring = get_ring()
        assert isinstance(ring, ProcessLogRing)

    def test_get_ring_returns_same_instance(self, monkeypatch):
        import agentfield.node_logs as nl

        monkeypatch.setattr(nl, "_global_ring", None)
        r1 = get_ring()
        r2 = get_ring()
        assert r1 is r2
