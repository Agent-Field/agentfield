"""
Tests for agentfield.node_logs — ProcessLogRing and related helpers.
"""
from __future__ import annotations

import json
import threading


from agentfield.node_logs import (
    LogEntry,
    ProcessLogRing,
    iter_tail_ndjson,
    verify_internal_bearer,
    get_ring,
)


# ---------------------------------------------------------------------------
# LogEntry NDJSON serialization
# ---------------------------------------------------------------------------


class TestLogEntryNdjson:
    def test_stdout_produces_info_level(self):
        entry = LogEntry(seq=1, ts="2024-01-01T00:00:00.000Z", stream="stdout", line="hello")
        data = json.loads(entry.to_ndjson_line().decode())
        assert data["level"] == "info"
        assert data["line"] == "hello"
        assert data["v"] == 1
        assert data["source"] == "process"

    def test_stderr_produces_error_level(self):
        entry = LogEntry(seq=2, ts="2024-01-01T00:00:00.000Z", stream="stderr", line="err")
        data = json.loads(entry.to_ndjson_line().decode())
        assert data["level"] == "error"

    def test_other_stream_produces_log_level(self):
        entry = LogEntry(seq=3, ts="2024-01-01T00:00:00.000Z", stream="custom", line="msg")
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
        entry = LogEntry(seq=42, ts="2024-06-15T10:00:00.000Z", stream="stdout", line="data")
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
        assert len(entries[0].line.encode("utf-8")) <= 10 + 3  # allow for replacement chars

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
