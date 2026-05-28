import json
import pytest
from unittest.mock import MagicMock

from agentfield.execution_context import (
    ExecutionContext,
    set_execution_context,
    reset_execution_context,
)
from agentfield.logger import (
    AgentFieldLogger,
    get_logger,
    set_log_level,
    set_cp_client,
)


class TestLoggerGlobals:
    def test_get_logger_returns_singleton(self):
        logger = get_logger()
        assert isinstance(logger, AgentFieldLogger)
        assert get_logger() is logger

    def test_set_log_level_accepts_debug(self):
        set_log_level("DEBUG")
        assert get_logger().logger.level == 10
        set_log_level("WARNING")

    def test_set_log_level_accepts_warn(self):
        set_log_level("WARN")
        assert get_logger().logger.level == 30
        set_log_level("WARNING")

    def test_set_cp_client_attaches_and_clears(self):
        fake = MagicMock()
        set_cp_client(fake)
        assert get_logger()._cp_client is fake
        set_cp_client(None)
        assert get_logger()._cp_client is None


@pytest.fixture
def debug_logger(monkeypatch):
    monkeypatch.setenv("AGENTFIELD_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("AGENTFIELD_LOG_TRACKING", "true")
    monkeypatch.setenv("AGENTFIELD_LOG_FIRE", "true")
    logger = AgentFieldLogger("level-test")
    logger.logger.propagate = True
    return logger


class TestLoggerLevelGating:
    def test_heartbeat_suppressed_at_info_level(self, caplog):
        logger = AgentFieldLogger("suppress-test")
        logger.logger.propagate = True
        logger.heartbeat("should not appear")
        assert "should not appear" not in caplog.text

    def test_heartbeat_shows_at_debug_level(self, debug_logger, caplog):
        debug_logger.heartbeat("should appear now")
        assert "should appear now" in caplog.text

    def test_track_suppressed_by_default(self, caplog, monkeypatch):
        monkeypatch.setenv("AGENTFIELD_LOG_LEVEL", "DEBUG")
        monkeypatch.setenv("AGENTFIELD_LOG_TRACKING", "false")
        logger = AgentFieldLogger("track-off")
        logger.logger.propagate = True
        logger.track("track-me")
        assert "track-me" not in caplog.text

    def test_track_shows_when_enabled(self, debug_logger, caplog):
        debug_logger.track("track-me-now")
        assert "track-me-now" in caplog.text

    def test_fire_suppressed_by_default(self, caplog, monkeypatch):
        monkeypatch.setenv("AGENTFIELD_LOG_LEVEL", "DEBUG")
        monkeypatch.setenv("AGENTFIELD_LOG_FIRE", "false")
        logger = AgentFieldLogger("fire-off")
        logger.logger.propagate = True
        logger.fire("fire-me")
        assert "fire-me" not in caplog.text

    def test_fire_shows_when_enabled(self, debug_logger, caplog):
        debug_logger.fire("fire-me-now", payload={"id": 1})
        assert "fire-me-now" in caplog.text


class TestLoggerStructuredOutput:
    def test_log_execution_emits_via_global_logger(self, capsys):
        ctx = ExecutionContext(
            workflow_id="wf-global",
            execution_id="exec-global",
            run_id="run-global",
            agent_instance=None,
            reasoner_name="r-global",
            agent_node_id="n-global",
            parent_execution_id="p-global",
            root_workflow_id="root-global",
            registered=True,
        )
        from agentfield.logger import log_execution

        log_execution(
            "global exec log",
            event_type="test.global",
            level="INFO",
            attributes={"src": "test"},
            execution_context=ctx,
        )

        out = capsys.readouterr().out.strip().splitlines()
        assert out
        record = json.loads(out[-1])
        assert record["execution_id"] == "exec-global"
        assert record["workflow_id"] == "wf-global"
        assert record["event_type"] == "test.global"
        assert record["message"] == "global exec log"
        assert record["attributes"]["src"] == "test"

    def test_info_enriches_current_context(self, capsys):
        ctx = ExecutionContext(
            workflow_id="wf-enrich",
            execution_id="exec-enrich",
            run_id="run-enrich",
            agent_instance=None,
            reasoner_name="r-enrich",
            agent_node_id="n-enrich",
            parent_execution_id=None,
            root_workflow_id="root-enrich",
            registered=True,
        )
        token = set_execution_context(ctx)
        try:
            get_logger().info("enriched info", stage="infer")
        finally:
            reset_execution_context(token)

        out = capsys.readouterr().out.strip().splitlines()
        assert out
        record = json.loads(out[-1])
        assert record["execution_id"] == "exec-enrich"
        assert record["workflow_id"] == "wf-enrich"
        assert record["level"] == "info"
        assert record["attributes"]["stage"] == "infer"


class TestLoggerInternals:
    def test_normalize_level_uppercases(self):
        assert AgentFieldLogger._normalize_level("info") == "INFO"
        assert AgentFieldLogger._normalize_level("INFO") == "INFO"

    def test_now_iso_ends_with_z(self):
        iso = AgentFieldLogger._now_iso()
        assert iso.endswith("Z")
        assert "T" in iso

    def test_merge_attributes_both_none(self):
        assert AgentFieldLogger._merge_attributes(None, None) == {}

    def test_merge_attributes_combines(self):
        assert AgentFieldLogger._merge_attributes({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}

    def test_merge_attributes_extra_overrides(self):
        assert AgentFieldLogger._merge_attributes({"a": 1}, {"a": 2}) == {"a": 2}

    def test_warning_alias(self, capsys):
        logger = AgentFieldLogger("alias-test")
        logger.logger.propagate = True
        logger.warning("alias msg")
        out = capsys.readouterr().out.strip()
        assert "alias msg" in out

    def test_build_record_no_context(self):
        logger = AgentFieldLogger("rec-test")
        record = logger._build_execution_record(
            message="test msg",
            level="ERROR",
            event_type="test.event",
        )
        assert record["message"] == "test msg"
        assert record["level"] == "error"
        assert record["event_type"] == "test.event"
        assert record["execution_id"] is None

    def test_build_record_with_context(self):
        ctx = ExecutionContext(
            workflow_id="wf-rec",
            execution_id="exec-rec",
            run_id="run-rec",
            agent_instance=None,
            reasoner_name="r-rec",
            agent_node_id="n-rec",
            parent_execution_id=None,
            root_workflow_id="root-rec",
            registered=True,
        )
        logger = AgentFieldLogger("rec-ctx")
        record = logger._build_execution_record(
            message="ctx msg",
            level="CRITICAL",
            event_type="ctx.event",
            execution_context=ctx,
            system_generated=True,
            source="test.src",
        )
        assert record["execution_id"] == "exec-rec"
        assert record["workflow_id"] == "wf-rec"
        assert record["level"] == "critical"
        assert record["system_generated"] is True
        assert record["source"] == "test.src"

    def test_set_level_runtime(self):
        logger = AgentFieldLogger("setlev")
        logger.set_level("CRITICAL")
        assert logger.logger.level == 50
