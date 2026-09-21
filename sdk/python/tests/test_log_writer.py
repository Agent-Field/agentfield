import asyncio
import io
import json
import os
import tempfile
import threading
from typing import TextIO

import pytest

from agentfield import log_writer
from agentfield.logger import AgentFieldLogger


@pytest.fixture(autouse=True)
def reset_log_writer(monkeypatch):
    log_writer.shutdown(1.0)
    monkeypatch.delenv("AGENTFIELD_LOG_QUEUE", raising=False)
    monkeypatch.delenv("AGENTFIELD_LOG_QUEUE_SIZE", raising=False)
    monkeypatch.delenv("AGENTFIELD_LOG_QUEUE_FLUSH_SECONDS", raising=False)
    try:
        yield
    finally:
        log_writer.shutdown(1.0)


def _emit_deferred(*lines: str, stream: TextIO) -> None:
    async def emit() -> None:
        for line in lines:
            log_writer.emit_line(line, stream)
        await asyncio.sleep(0)

    asyncio.run(emit())


class _BlockingStream:
    def __init__(self, stream: TextIO) -> None:
        self.stream = stream
        self.started = threading.Event()
        self.release = threading.Event()

    def fileno(self) -> int:
        return self.stream.fileno()

    def write(self, value: str) -> int:
        if not self.started.is_set():
            self.started.set()
            if not self.release.wait(2.0):
                raise TimeoutError("test stream was not released")
        return self.stream.write(value)

    def flush(self) -> None:
        self.stream.flush()


class _BrokenStream:
    def __init__(self, fd: int) -> None:
        self.fd = fd

    def fileno(self) -> int:
        return self.fd

    def write(self, value: str) -> int:
        raise BrokenPipeError("closed")

    def flush(self) -> None:
        raise BrokenPipeError("closed")


class _WriteStartedStream:
    def __init__(self, stream: TextIO) -> None:
        self.stream = stream
        self.started = threading.Event()

    def fileno(self) -> int:
        return self.stream.fileno()

    def write(self, value: str) -> int:
        self.started.set()
        return self.stream.write(value)

    def flush(self) -> None:
        self.stream.flush()


class _MarkerFailingStream(_BlockingStream):
    def __init__(self, stream: TextIO) -> None:
        super().__init__(stream)
        self.marker_failed = False

    def write(self, value: str) -> int:
        if not self.started.is_set():
            self.started.set()
            if not self.release.wait(2.0):
                raise TimeoutError("test stream was not released")
        if '"event_type":"log.dropped"' in value and not self.marker_failed:
            self.marker_failed = True
            raise BrokenPipeError("first marker write fails")
        return self.stream.write(value)


class _RecordingStream:
    def __init__(self, stream: TextIO) -> None:
        self.stream = stream
        self.writes: list[str] = []
        self.flushes = 0

    def fileno(self) -> int:
        return self.stream.fileno()

    def write(self, value: str) -> int:
        self.writes.append(value)
        return self.stream.write(value)

    def flush(self) -> None:
        self.flushes += 1
        self.stream.flush()


@pytest.mark.unit
def test_full_pipe_does_not_stall_event_loop():
    # C1: a blocked real stdout fd cannot block producers or the event loop.
    read_fd, write_fd = os.pipe()
    os.set_blocking(write_fd, False)
    while True:
        try:
            os.write(write_fd, b"x" * 65536)
        except BlockingIOError:
            break
    os.set_blocking(write_fd, True)
    stream = os.fdopen(write_fd, "w", buffering=1, encoding="utf-8")

    async def emit_with_watchdog() -> tuple[float, float]:
        loop = asyncio.get_running_loop()
        started = loop.time()
        worst_lag = 0.0

        async def watchdog() -> None:
            nonlocal worst_lag
            target = loop.time()
            for _ in range(5):
                target += 0.01
                await asyncio.sleep(max(0.0, target - loop.time()))
                worst_lag = max(worst_lag, loop.time() - target)

        async def producer() -> None:
            for index in range(32):
                log_writer.emit_line(f"queued-{index}", stream)
            await asyncio.sleep(0)

        await asyncio.gather(producer(), watchdog())
        return loop.time() - started, worst_lag

    elapsed, worst_lag = asyncio.run(emit_with_watchdog())
    assert log_writer.flush(0.01) is False

    output = bytearray()

    def drain() -> None:
        while b"queued-31\n" not in output:
            chunk = os.read(read_fd, 65536)
            if not chunk:
                return
            output.extend(chunk)

    reader = threading.Thread(target=drain)
    reader.start()
    try:
        assert log_writer.flush(2.0) is True
        reader.join(2.0)
        assert not reader.is_alive()
    finally:
        stream.close()
        os.close(read_fd)

    assert elapsed < 2.0
    assert worst_lag < 0.5
    assert b"queued-0\n" in output
    assert b"queued-31\n" in output


@pytest.mark.unit
def test_plain_handler_does_not_block_event_loop_behind_sync_writer(monkeypatch):
    read_fd, write_fd = os.pipe()
    os.set_blocking(write_fd, False)
    while True:
        try:
            os.write(write_fd, b"x" * 65536)
        except BlockingIOError:
            break
    os.set_blocking(write_fd, True)
    target = os.fdopen(write_fd, "w", buffering=1, encoding="utf-8")
    stream = _WriteStartedStream(target)
    output = bytearray()

    with monkeypatch.context() as context:
        context.setattr("sys.stdout", stream)
        logger = AgentFieldLogger("log-writer-handler-lock")
        logger.set_level("INFO")
        sync_thread = threading.Thread(target=logger.info, args=("sync-blocked",))
        sync_thread.start()
        assert stream.started.wait(1.0)
        assert sync_thread.is_alive()

        async def emit_from_loop() -> float:
            loop = asyncio.get_running_loop()
            started = loop.time()
            logger.info("async-queued")
            return loop.time() - started

        elapsed = asyncio.run(emit_from_loop())
        assert elapsed < 1.0
        assert sync_thread.is_alive()

        def drain() -> None:
            while b"async-queued\n" not in output:
                chunk = os.read(read_fd, 65536)
                if not chunk:
                    return
                output.extend(chunk)

        reader = threading.Thread(target=drain)
        reader.start()
        sync_thread.join(2.0)
        assert not sync_thread.is_alive()
        assert log_writer.flush(2.0) is True
        reader.join(2.0)
        assert not reader.is_alive()

    target.close()
    os.close(read_fd)
    assert b"sync-blocked\n" in output
    assert b"async-queued\n" in output


@pytest.mark.unit
def test_structured_and_plain_lines_keep_emit_order(monkeypatch):
    # C2/C7: structured and human-readable lines share one FIFO writer.
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as stream:
        with monkeypatch.context() as context:
            context.setattr("sys.stdout", stream)
            logger = AgentFieldLogger("log-writer-order")
            logger.set_level("INFO")

            async def emit() -> None:
                logger._emit_structured_record({"sequence": 1})
                logger.info("plain-two")
                logger._emit_structured_record({"sequence": 3})
                await asyncio.sleep(0)

            asyncio.run(emit())
            assert log_writer.flush(1.0) is True
        stream.seek(0)
        lines = stream.read().splitlines()

    assert json.loads(lines[0]) == {"sequence": 1}
    assert lines[1] == "ℹ️ plain-two"
    assert json.loads(lines[2]) == {"sequence": 3}


@pytest.mark.unit
def test_sync_plain_log_stays_behind_dequeued_line(monkeypatch):
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as target:
        stream = _BlockingStream(target)
        with monkeypatch.context() as context:
            context.setattr("sys.stdout", stream)
            logger = AgentFieldLogger("log-writer-dequeued-order")
            logger.set_level("INFO")

            async def emit_first() -> None:
                logger.info("queued-first")

            asyncio.run(emit_first())
            assert stream.started.wait(1.0)

            sync_thread = threading.Thread(target=logger.info, args=("sync-second",))
            sync_thread.start()
            sync_thread.join(1.0)
            assert not sync_thread.is_alive()

            stream.release.set()
            assert log_writer.flush(1.0) is True

        target.seek(0)
        assert target.read().splitlines() == ["ℹ️ queued-first", "ℹ️ sync-second"]


@pytest.mark.unit
def test_overflow_discards_oldest_and_emits_one_marker(monkeypatch):
    # C3: overflow drops oldest pending lines and reports the complete count.
    monkeypatch.setenv("AGENTFIELD_LOG_QUEUE_SIZE", "2")
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as target:
        stream = _BlockingStream(target)
        _emit_deferred("line-0", stream=stream)  # type: ignore[arg-type]
        assert stream.started.wait(1.0)

        _emit_deferred(
            "line-1",
            "line-2",
            "line-3",
            "line-4",
            "line-5",
            stream=stream,  # type: ignore[arg-type]
        )
        assert log_writer.stats()["dropped"] == 3

        stream.release.set()
        assert log_writer.flush(1.0) is True
        target.seek(0)
        lines = target.read().splitlines()

    markers = [
        json.loads(line) for line in lines if '"event_type":"log.dropped"' in line
    ]
    assert lines[0] == "line-0"
    assert len(markers) == 1
    assert markers[0]["level"] == "warning"
    assert markers[0]["source"] == "sdk.python.logger"
    assert markers[0]["attributes"] == {"dropped": 3}
    assert lines[-2:] == ["line-4", "line-5"]


@pytest.mark.unit
def test_drop_marker_is_reported_only_on_its_destination_stream(monkeypatch):
    monkeypatch.setenv("AGENTFIELD_LOG_QUEUE_SIZE", "1")
    with (
        tempfile.TemporaryFile(mode="w+", encoding="utf-8") as target_a,
        tempfile.TemporaryFile(mode="w+", encoding="utf-8") as target_b,
    ):
        stream_a = _BlockingStream(target_a)
        _emit_deferred("a-active", stream=stream_a)  # type: ignore[arg-type]
        assert stream_a.started.wait(1.0)
        _emit_deferred("a-evicted", stream=stream_a)  # type: ignore[arg-type]
        _emit_deferred("b-kept", stream=target_b)
        assert log_writer.stats()["dropped"] == 1

        stream_a.release.set()
        assert log_writer.flush(1.0) is True
        target_b.seek(0)
        assert target_b.read().splitlines() == ["b-kept"]
        assert log_writer.stats()["dropped"] == 1

        _emit_deferred("a-later", stream=stream_a)  # type: ignore[arg-type]
        assert log_writer.flush(1.0) is True
        target_a.seek(0)
        lines_a = target_a.read().splitlines()

    assert lines_a[0] == "a-active"
    assert json.loads(lines_a[1])["attributes"] == {"dropped": 1}
    assert lines_a[2] == "a-later"
    assert log_writer.stats()["dropped"] == 0


@pytest.mark.unit
def test_failed_drop_marker_is_restored_for_next_write(monkeypatch):
    monkeypatch.setenv("AGENTFIELD_LOG_QUEUE_SIZE", "1")
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as target:
        stream = _MarkerFailingStream(target)
        _emit_deferred("active", stream=stream)  # type: ignore[arg-type]
        assert stream.started.wait(1.0)
        _emit_deferred("evicted", stream=stream)  # type: ignore[arg-type]
        _emit_deferred("kept", stream=stream)  # type: ignore[arg-type]
        assert log_writer.stats()["dropped"] == 1

        stream.release.set()
        assert log_writer.flush(1.0) is True
        assert stream.marker_failed is True
        assert log_writer.stats()["dropped"] == 1

        _emit_deferred("later", stream=stream)  # type: ignore[arg-type]
        assert log_writer.flush(1.0) is True
        target.seek(0)
        lines = target.read().splitlines()

    assert lines[0:2] == ["active", "kept"]
    assert json.loads(lines[2])["attributes"] == {"dropped": 1}
    assert lines[3] == "later"
    assert log_writer.stats()["dropped"] == 0


@pytest.mark.unit
def test_line_and_newline_use_one_stream_write():
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as target:
        stream = _RecordingStream(target)
        log_writer.emit_line("single-write", stream)  # type: ignore[arg-type]

    assert stream.writes == ["single-write\n"]
    assert stream.flushes == 1


@pytest.mark.unit
def test_worker_start_failure_drops_loop_line_and_reports_it_later(monkeypatch):
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as target:
        with monkeypatch.context() as context:
            context.setattr("sys.stdout", target)
            logger = AgentFieldLogger("log-writer-start-failure")
            logger.set_level("INFO")

            def fail_start(_thread) -> None:
                raise RuntimeError("cannot start worker")

            context.setattr(threading.Thread, "start", fail_start)

            async def emit_dropped() -> None:
                logger.info("dropped-on-start")

            asyncio.run(emit_dropped())

        target.seek(0)
        assert target.read() == ""
        assert log_writer.stats()["dropped"] == 1

        with monkeypatch.context() as context:
            context.setattr("sys.stdout", target)
            logger = AgentFieldLogger("log-writer-start-recovered")
            logger.set_level("INFO")

            async def emit_recovered() -> None:
                logger.info("written-after-recovery")

            asyncio.run(emit_recovered())
            assert log_writer.flush(1.0) is True

        target.seek(0)
        lines = target.read().splitlines()

    assert json.loads(lines[0])["attributes"] == {"dropped": 1}
    assert lines[1] == "ℹ️ written-after-recovery"


@pytest.mark.unit
def test_stdout_disabled_never_reaches_writer(monkeypatch):
    # C4: disabling the structured mirror creates no line and no queue work.
    monkeypatch.setenv("AGENTFIELD_LOG_STDOUT", "false")
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as stream:
        with monkeypatch.context() as context:
            context.setattr("sys.stdout", stream)
            logger = AgentFieldLogger("log-writer-stdout-disabled")

            async def emit() -> None:
                logger._emit_structured_record({"event_type": "test"})
                await asyncio.sleep(0)

            asyncio.run(emit())
        stream.seek(0)
        assert stream.read() == ""
    assert log_writer.stats()["pending"] == 0
    assert log_writer.stats()["worker_alive"] is False


@pytest.mark.unit
@pytest.mark.parametrize("value", ["false", "FALSE", "  off  ", "0", "no"])
def test_queue_opt_out_writes_inline_inside_event_loop(monkeypatch, value):
    # C5: every supported false spelling restores synchronous writes.
    monkeypatch.setenv("AGENTFIELD_LOG_QUEUE", value)
    read_fd, write_fd = os.pipe()
    stream = os.fdopen(write_fd, "w", encoding="utf-8")
    try:

        async def emit() -> None:
            log_writer.emit_line("inline", stream)

        asyncio.run(emit())
        assert os.read(read_fd, 7) == b"inline\n"
        assert log_writer.stats()["worker_alive"] is False
    finally:
        stream.close()
        os.close(read_fd)


@pytest.mark.unit
def test_capture_and_no_loop_writes_are_immediate():
    # C6: captures and synchronous callers retain immediate visibility.
    capture = io.StringIO()

    async def emit_to_capture() -> None:
        log_writer.emit_line("captured", capture)
        assert capture.getvalue() == "captured\n"

    asyncio.run(emit_to_capture())

    with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as stream:
        log_writer.emit_line("synchronous", stream)
        stream.seek(0)
        assert stream.read() == "synchronous\n"
    assert log_writer.stats()["worker_alive"] is False


@pytest.mark.unit
def test_exit_flush_drains_backlog(monkeypatch):
    # C8: the registered exit callback drains queued lines within its timeout.
    monkeypatch.setenv("AGENTFIELD_LOG_QUEUE_FLUSH_SECONDS", "1")
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as target:
        stream = _BlockingStream(target)
        _emit_deferred("first", stream=stream)  # type: ignore[arg-type]
        assert stream.started.wait(1.0)
        _emit_deferred("second", stream=stream)  # type: ignore[arg-type]

        stream.release.set()
        log_writer._flush_at_exit()
        target.seek(0)
        assert target.read().splitlines() == ["first", "second"]


@pytest.mark.unit
def test_at_fork_handlers_reset_child_state(monkeypatch):
    # C8: a child receives fresh queue state instead of an inherited worker.
    monkeypatch.setenv("AGENTFIELD_LOG_QUEUE_SIZE", "7")
    log_writer._before_fork()
    log_writer._after_fork_child()
    assert log_writer.stats() == {
        "pending": 0,
        "dropped": 0,
        "worker_alive": False,
        "active": False,
    }
    assert log_writer._queue.maxsize == 7

    log_writer._before_fork()
    log_writer._after_fork_parent()


@pytest.mark.unit
def test_broken_stream_is_swallowed_and_worker_survives():
    # C9: a failed destination does not escape or poison later writes.
    read_fd, write_fd = os.pipe()
    broken = _BrokenStream(write_fd)
    try:
        _emit_deferred("broken", stream=broken)  # type: ignore[arg-type]
        assert log_writer.flush(1.0) is True
        assert log_writer.stats()["worker_alive"] is True

        with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as healthy:
            _emit_deferred("healthy", stream=healthy)
            assert log_writer.flush(1.0) is True
            healthy.seek(0)
            assert healthy.read() == "healthy\n"
    finally:
        os.close(write_fd)
        os.close(read_fd)


@pytest.mark.unit
def test_shutdown_allows_worker_restart():
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as stream:
        _emit_deferred("before", stream=stream)
        assert log_writer.flush(1.0) is True
        log_writer.shutdown(1.0)
        assert log_writer.stats()["worker_alive"] is False

        _emit_deferred("after", stream=stream)
        assert log_writer.flush(1.0) is True
        stream.seek(0)
        assert stream.read().splitlines() == ["before", "after"]


@pytest.mark.unit
def test_invalid_queue_configuration_falls_back_and_clamps(monkeypatch):
    monkeypatch.setenv("AGENTFIELD_LOG_QUEUE_SIZE", "invalid")
    assert log_writer._queue_size() == 1024
    monkeypatch.setenv("AGENTFIELD_LOG_QUEUE_SIZE", "0")
    assert log_writer._queue_size() == 1

    monkeypatch.setenv("AGENTFIELD_LOG_QUEUE_FLUSH_SECONDS", "invalid")
    assert log_writer._flush_seconds() == 2.0
    monkeypatch.setenv("AGENTFIELD_LOG_QUEUE_FLUSH_SECONDS", "-1")
    assert log_writer._flush_seconds() == 0.0
