import io
import json
import logging
import threading
import time
from unittest.mock import Mock

import pytest

import agentfield.logger as logger_module
from agentfield.logger import (
    AgentFieldLogger,
    get_logger,
    log_info,
    set_cp_client,
    set_log_level,
)


@pytest.mark.unit
def test_structured_stdout_can_be_disabled(monkeypatch, capsys):
    monkeypatch.setenv("AGENTFIELD_LOG_STDOUT", "false")
    logger = AgentFieldLogger("structured.stdout.disabled")

    logger._emit_structured_record({"event_type": "test"})

    assert capsys.readouterr().out == ""


@pytest.mark.unit
def test_structured_stdout_disabled_skips_serialization(monkeypatch, capsys):
    class Unserializable:
        def __str__(self):
            raise AssertionError(
                "structured stdout should not serialize disabled records"
            )

    monkeypatch.setenv("AGENTFIELD_LOG_STDOUT", "false")
    logger = AgentFieldLogger("structured.stdout.no-serialization")

    logger._emit_structured_record({"event_type": "test", "payload": Unserializable()})

    assert capsys.readouterr().out == ""


@pytest.mark.unit
def test_structured_stdout_is_enabled_by_default(capsys):
    logger = AgentFieldLogger("structured.stdout.default")

    logger._emit_structured_record({"event_type": "test"})

    assert '"event_type":"test"' in capsys.readouterr().out


@pytest.mark.unit
@pytest.mark.parametrize("value", ["false", "FALSE", "  False  ", "0", "no", "off"])
def test_structured_stdout_disabled_for_every_falsy_spelling(
    monkeypatch, capsys, value
):
    monkeypatch.setenv("AGENTFIELD_LOG_STDOUT", value)
    logger = AgentFieldLogger(f"structured.stdout.off.{value.strip().lower()}")

    logger._emit_structured_record({"event_type": "test"})

    assert capsys.readouterr().out == ""


@pytest.mark.unit
@pytest.mark.parametrize(
    "value", ["true", "TRUE", "  True  ", "1", "yes", "on", "", "ture"]
)
def test_structured_stdout_stays_on_unless_explicitly_disabled(
    monkeypatch, capsys, value
):
    """Only the documented falsy values silence the mirror.

    ``1``/``yes``/``on`` are truthy everywhere else in the SDK, and a
    set-but-empty or misspelt value must not silently drop log output -- the
    default has to fail towards keeping records visible.
    """
    monkeypatch.setenv("AGENTFIELD_LOG_STDOUT", value)
    logger = AgentFieldLogger(
        f"structured.stdout.on.{value.strip().lower() or 'empty'}"
    )

    logger._emit_structured_record({"event_type": "test"})

    assert '"event_type":"test"' in capsys.readouterr().out


@pytest.fixture(autouse=True)
def reset_logger_state():
    """Isolate each test from global logger state.

    Two distinct kinds of global state have to be reset:

    1. The SDK-level globals (``_logger_cache`` / ``_global_log_level`` /
       ``_global_cp_client``). These are referenced via ``logger_module`` so the
       reset hits the *real* module globals — assigning a bare module-level name
       here would only rebind this test module's copy and silently fail to reset
       anything.
    2. The stdlib ``logging`` registry. ``AgentFieldLogger`` attaches a handler
       and sets ``propagate = False`` on the underlying ``logging.Logger``.
       Those mutations outlive the test and leak across the whole session — e.g.
       creating the ``"agentfield"`` logger here would stop ``"agentfield.cancel"``
       records from reaching root handlers (pytest's ``caplog``), breaking
       unrelated tests like ``test_cancel.py``. Snapshot and restore so this
       test file is order-independent.
    """
    manager = logging.root.manager
    saved = {
        name: (lgr.propagate, list(lgr.handlers))
        for name, lgr in manager.loggerDict.items()
        if isinstance(lgr, logging.Logger)
    }

    logger_module._logger_cache.clear()
    logger_module._global_log_level = None
    logger_module._global_cp_client = None
    try:
        yield
    finally:
        logger_module._logger_cache.clear()
        logger_module._global_log_level = None
        logger_module._global_cp_client = None
        for name, lgr in list(manager.loggerDict.items()):
            if not isinstance(lgr, logging.Logger):
                continue
            if name in saved:
                propagate, handlers = saved[name]
                lgr.propagate = propagate
                lgr.handlers[:] = handlers
            else:
                # Logger created during the test — return it to stdlib defaults.
                lgr.propagate = True
                lgr.handlers.clear()


@pytest.mark.unit
def test_get_logger_returns_correct_name():
    """Test that get_logger returns logger with correct name."""
    logger = get_logger("agentfield.client")
    assert logger.logger.name == "agentfield.client"


@pytest.mark.unit
def test_different_names_produce_different_loggers():
    """Test that different names produce different logger instances."""
    a = get_logger("module_a")
    b = get_logger("module_b")
    assert a.logger.name == "module_a"
    assert b.logger.name == "module_b"
    assert a is not b


@pytest.mark.unit
def test_same_name_returns_same_logger():
    """Test that requesting the same name returns the cached logger."""
    a = get_logger("module_a")
    b = get_logger("module_a")
    assert a is b


@pytest.mark.unit
def test_set_log_level_affects_all_loggers():
    """Test that set_log_level affects all existing logger instances."""
    a = get_logger("a")
    b = get_logger("b")

    set_log_level("DEBUG")

    assert a.log_level == "DEBUG"
    assert b.log_level == "DEBUG"


@pytest.mark.unit
def test_set_log_level_applies_to_new_loggers():
    """Test that set_log_level also applies to loggers created after it's called."""
    set_log_level("DEBUG")
    a = get_logger("a")
    b = get_logger("b")

    assert a.log_level == "DEBUG"
    assert b.log_level == "DEBUG"


@pytest.mark.unit
def test_log_info_works_without_arguments():
    """Test that log_info works without explicit logger arguments."""
    log_info("test message")


@pytest.mark.unit
def test_default_name_returns_agentfield():
    """Test that get_logger() without arguments returns logger with name 'agentfield'."""
    logger = get_logger()
    assert logger.logger.name == "agentfield"


@pytest.mark.unit
def test_backward_compatibility_default_logger():
    """Test backward compatibility: get_logger() returns proper default logger."""
    logger1 = get_logger()
    logger2 = get_logger("agentfield")
    assert logger1 is logger2
    assert logger1.logger.name == "agentfield"


@pytest.mark.unit
def test_concurrent_access_is_threadsafe():
    """Concurrent get_logger()/set_log_level() must not raise or corrupt the cache.

    Without locking, iterating the cache in set_log_level() while get_logger()
    inserts into it can raise ``RuntimeError: dictionary changed size during
    iteration``. Hammer both paths from several threads and assert clean results.
    """
    errors: list[BaseException] = []
    barrier = threading.Barrier(8)

    def worker(i: int) -> None:
        try:
            barrier.wait()
            for j in range(50):
                get_logger(f"concurrent.{i}.{j}")
                set_log_level("DEBUG" if j % 2 else "INFO")
        except BaseException as exc:  # noqa: BLE001 - surface any thread failure
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"thread-safety violation: {errors[:3]}"
    # Every distinct name must have produced its own correctly-named logger.
    assert get_logger("concurrent.0.0").logger.name == "concurrent.0.0"


class _FakeCpClient:
    """Stand-in control-plane client; identity is all these tests check."""


@pytest.mark.unit
def test_logger_created_after_set_cp_client_forwards_to_cp():
    """Regression: a logger created *after* set_cp_client() must still be wired
    to the control-plane client.

    Before the fix, set_cp_client() only mutated loggers already in the cache,
    so a logger created later (e.g. the lazily-imported agentfield.verification
    logger, created inside Agent.__init__ *after* set_cp_client runs) kept the
    class default _cp_client=None and silently dropped all structured telemetry
    in _dispatch_to_cp().
    """
    client = _FakeCpClient()
    set_cp_client(client)

    late_logger = get_logger("agentfield.created.after")

    assert late_logger._cp_client is client


@pytest.mark.unit
def test_set_cp_client_applies_to_already_cached_loggers():
    """set_cp_client() must still reach loggers created before it ran."""
    early_logger = get_logger("agentfield.created.before")
    assert early_logger._cp_client is None

    client = _FakeCpClient()
    set_cp_client(client)

    assert early_logger._cp_client is client


@pytest.mark.unit
def test_set_cp_client_none_clears_forwarding_for_future_loggers():
    """Passing None detaches the client globally so later loggers don't forward."""
    set_cp_client(_FakeCpClient())
    set_cp_client(None)

    assert get_logger("agentfield.created.after.reset")._cp_client is None


@pytest.mark.unit
def test_structured_mirror_is_bounded_valid_json_and_cp_receives_full_record(
    monkeypatch,
):
    monkeypatch.setenv("AGENTFIELD_LOG_MAX_LINE_BYTES", "512")
    stream = io.StringIO()
    monkeypatch.setattr(logger_module.sys, "stdout", stream)
    logger = AgentFieldLogger("bounded-structured")
    dispatch = Mock()
    monkeypatch.setattr(logger, "_dispatch_to_cp", dispatch)
    record = logger._build_execution_record(
        message="kept",
        level="INFO",
        event_type="reasoner.completed",
        source="test.source",
        attributes={"result": "x" * 4000},
    )
    record["execution_id"] = "exec-1"
    record["run_id"] = "run-1"

    logger._emit_structured_record(record)

    line = stream.getvalue().rstrip("\n")
    mirrored = json.loads(line)
    assert len(line.encode()) <= 512
    for key in ("execution_id", "run_id", "level", "event_type", "message", "source"):
        assert mirrored[key] == record[key]
    assert mirrored["attributes"]["result"] == "<4002 bytes elided>"
    dispatch.assert_called_once_with(record)
    assert record["attributes"]["result"] == "x" * 4000


@pytest.mark.unit
def test_structured_mirror_elides_oversized_message(monkeypatch):
    monkeypatch.setenv("AGENTFIELD_LOG_MAX_LINE_BYTES", "512")
    stream = io.StringIO()
    monkeypatch.setattr(logger_module.sys, "stdout", stream)

    AgentFieldLogger("oversized-message").log_execution(
        "λ" * 4000, event_type="test.oversized-message"
    )

    line = stream.getvalue().rstrip("\n")
    mirrored = json.loads(line)
    assert len(line.encode("utf-8")) <= 512
    assert "bytes elided]" in mirrored["message"]


@pytest.mark.unit
def test_structured_mirror_elides_oversized_non_dict_attributes(monkeypatch):
    monkeypatch.setenv("AGENTFIELD_LOG_MAX_LINE_BYTES", "512")
    stream = io.StringIO()
    monkeypatch.setattr(logger_module.sys, "stdout", stream)
    logger = AgentFieldLogger("non-dict-attributes")
    record = logger._build_execution_record(
        message="kept", level="INFO", event_type="test.non-dict"
    )
    record["attributes"] = ["x" * 4000]

    logger._emit_structured_record(record)

    line = stream.getvalue().rstrip("\n")
    mirrored = json.loads(line)
    assert len(line.encode("utf-8")) <= 512
    assert mirrored["attributes"] == "<4004 bytes elided>"


@pytest.mark.unit
def test_structured_mirror_many_large_attributes_stays_fast(monkeypatch):
    monkeypatch.setenv("AGENTFIELD_LOG_MAX_LINE_BYTES", "4000")
    monkeypatch.setattr(logger_module.sys, "stdout", io.StringIO())
    attributes = {f"key-{index}": "x" * 50_000 for index in range(200)}

    start = time.perf_counter()
    AgentFieldLogger("many-attributes").log_execution(
        "bounded", event_type="test.many-attributes", attributes=attributes
    )
    elapsed = time.perf_counter() - start

    assert elapsed < 0.2


@pytest.mark.unit
def test_public_structured_logger_emits_only_bounded_json_lines(monkeypatch):
    cap = 512
    monkeypatch.setenv("AGENTFIELD_LOG_MAX_LINE_BYTES", str(cap))
    monkeypatch.setenv("AGENTFIELD_LOG_STDOUT", "true")
    stream = io.StringIO()
    monkeypatch.setattr(logger_module.sys, "stdout", stream)
    logger = get_logger("agentfield")
    set_log_level("INFO")

    logger.log_execution("short", event_type="test.short", attributes={"ok": True})
    logger.log_execution(
        "x" * 4000,
        event_type="test.large",
        attributes={"payload": "y" * 4000},
    )

    lines = stream.getvalue().splitlines()
    assert len(lines) == 2
    for line in lines:
        json.loads(line)
        assert len(line.encode("utf-8")) <= cap


@pytest.mark.unit
def test_structured_stdout_false_skips_serialization_but_dispatches(monkeypatch):
    monkeypatch.setenv("AGENTFIELD_LOG_STDOUT", "false")
    logger = AgentFieldLogger("no-structured-stdout")
    dispatch = Mock()
    monkeypatch.setattr(logger, "_dispatch_to_cp", dispatch)
    dumps = Mock(side_effect=AssertionError("mirror serialization must be skipped"))
    monkeypatch.setattr(logger_module.json, "dumps", dumps)
    record = {"execution_id": "exec-1", "attributes": {"large": "x" * 1000}}

    logger._emit_structured_record(record)

    dumps.assert_not_called()
    dispatch.assert_called_once_with(record)


@pytest.mark.unit
@pytest.mark.parametrize("error", [BrokenPipeError(), OSError("closed")])
def test_structured_stdout_errors_never_escape(monkeypatch, error):
    class BrokenStream:
        def write(self, _value):
            raise error

        def flush(self):
            raise error

    monkeypatch.setattr(logger_module.sys, "stdout", BrokenStream())
    logger = AgentFieldLogger("broken-structured-stdout")
    monkeypatch.setattr(logger, "_dispatch_to_cp", Mock())

    logger.log_execution("still safe", event_type="test.event", execution_id="exec-1")


@pytest.mark.unit
def test_logger_created_before_tee_uses_current_stdout(monkeypatch):
    before = io.StringIO()
    after = io.StringIO()
    monkeypatch.setattr(logger_module.sys, "stdout", before)
    logger = AgentFieldLogger("lazy-stdout")
    monkeypatch.setattr(logger_module.sys, "stdout", after)

    logger.logger.error("captured after install")

    assert before.getvalue() == ""
    assert after.getvalue() == "captured after install\n"
