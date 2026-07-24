import json
import pytest

from agentfield.execution_context import (
    ExecutionContext,
    reset_execution_context,
    set_execution_context,
)
from agentfield.logger import log_execution, log_info, AgentFieldLogger


@pytest.mark.unit
def test_log_info_auto_enriches_current_execution_context(capsys):
    ctx = ExecutionContext(
        workflow_id="wf-1",
        execution_id="exec-1",
        run_id="run-1",
        agent_instance=None,
        reasoner_name="sample_reasoner",
        agent_node_id="node-1",
        parent_execution_id="parent-1",
        parent_workflow_id="wf-parent",
        root_workflow_id="wf-root",
        registered=True,
    )
    token = set_execution_context(ctx)

    try:
        log_info("Execution checkpoint", stage="loading")
    finally:
        reset_execution_context(token)

    out = capsys.readouterr().out.strip().splitlines()
    assert out, "expected a structured execution log line"
    record = json.loads(out[-1])

    assert record["execution_id"] == "exec-1"
    assert record["workflow_id"] == "wf-1"
    assert record["run_id"] == "run-1"
    assert record["root_workflow_id"] == "wf-root"
    assert record["parent_execution_id"] == "parent-1"
    assert record["agent_node_id"] == "node-1"
    assert record["reasoner_id"] == "sample_reasoner"
    assert record["level"] == "info"
    assert record["event_type"] == "log.info"
    assert record["message"] == "Execution checkpoint"
    assert record["system_generated"] is False
    assert record["attributes"]["stage"] == "loading"
    assert record["attributes"]["depth"] == 0


@pytest.mark.unit
def test_log_execution_emits_structured_record(capsys):
    ctx = ExecutionContext(
        workflow_id="wf-2",
        execution_id="exec-2",
        run_id="run-2",
        agent_instance=None,
        reasoner_name="workflow_reasoner",
        agent_node_id="node-2",
        parent_execution_id=None,
        root_workflow_id="wf-root-2",
        registered=True,
    )

    log_execution(
        "Reasoner started",
        event_type="reasoner.started",
        level="INFO",
        attributes={"status": "running"},
        execution_context=ctx,
        system_generated=True,
        source="sdk.python.agent_workflow",
    )

    out = capsys.readouterr().out.strip().splitlines()
    assert out, "expected a structured execution log line"
    record = json.loads(out[-1])

    assert record["execution_id"] == "exec-2"
    assert record["workflow_id"] == "wf-2"
    assert record["run_id"] == "run-2"
    assert record["root_workflow_id"] == "wf-root-2"
    assert record["agent_node_id"] == "node-2"
    assert record["reasoner_id"] == "workflow_reasoner"
    assert record["level"] == "info"
    assert record["event_type"] == "reasoner.started"
    assert record["message"] == "Reasoner started"
    assert record["system_generated"] is True
    assert record["source"] == "sdk.python.agent_workflow"
    assert record["attributes"]["status"] == "running"


@pytest.mark.asyncio
async def test_workflow_lifecycle_logs_emit_execution_events(monkeypatch):
    from agentfield.agent_workflow import AgentWorkflow
    from tests.helpers import StubAgent

    agent = StubAgent()
    workflow = AgentWorkflow(agent)
    captured = []

    async def noop_update(payload):
        return None

    def capture(*args, **kwargs):
        captured.append(
            {
                "message": args[0],
                "event_type": kwargs["event_type"],
                "level": kwargs["level"],
                "attributes": kwargs["attributes"],
                "system_generated": kwargs["system_generated"],
                "source": kwargs["source"],
            }
        )
        return {}

    monkeypatch.setattr(workflow, "fire_and_forget_update", noop_update)
    monkeypatch.setattr("agentfield.agent_workflow.log_execution", capture)

    context = ExecutionContext.create_new(agent.node_id, "root")
    context.reasoner_name = "sample_reasoner"

    await workflow.notify_call_start(
        context.execution_id,
        context,
        "sample_reasoner",
        {"value": 1},
    )
    await workflow.notify_call_complete(
        context.execution_id,
        context.workflow_id,
        {"ok": True},
        15,
        context,
        input_data={"value": 1},
    )
    await workflow.notify_call_error(
        context.execution_id,
        context.workflow_id,
        "boom",
        16,
        context,
        input_data={"value": 1},
    )

    assert [entry["event_type"] for entry in captured] == [
        "reasoner.started",
        "reasoner.completed",
        "reasoner.failed",
    ]
    assert captured[0]["system_generated"] is True
    assert captured[1]["attributes"]["duration_ms"] == 15
    assert captured[1]["attributes"]["result"] == {"ok": True}
    assert captured[2]["attributes"]["error"] == "boom"

@pytest.fixture
def base_logger(monkeypatch):
    """
    Initializes an AgentFieldLogger with environment overrides to validate 
    core observability and data-handling logic.
    """
    monkeypatch.setenv("AGENTFIELD_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("AGENTFIELD_LOG_PAYLOADS", "true")
    monkeypatch.setenv("AGENTFIELD_LOG_TRUNCATE", "50")
    monkeypatch.setenv("AGENTFIELD_LOG_TRACKING", "true")
    monkeypatch.setenv("AGENTFIELD_LOG_FIRE", "true")
    
    logger = AgentFieldLogger(name="telemetry")
    logger.logger.propagate = True
    return logger

@pytest.mark.unit
def test_heartbeat_event(base_logger, caplog):
    base_logger.heartbeat("Agent pulsing", status="nominal")
    assert "Agent pulsing" in caplog.text

@pytest.mark.unit
def test_logger_track_output(base_logger, caplog):
    base_logger.track("token_usage", count=500)
    assert "token_usage" in caplog.text

@pytest.mark.unit
def test_logger_fire_output(base_logger, caplog):
    base_logger.fire("node_transition", target="reasoning_node")
    assert "node_transition" in caplog.text

@pytest.mark.unit
def test_logger_debug_output(base_logger, caplog):
    base_logger.debug("Debugging logic")
    assert "Debugging" in caplog.text

@pytest.mark.unit
def test_logger_security_output(base_logger, caplog):
    base_logger.security("Sanitized keys", level="HIGH")
    assert "Sanitized" in caplog.text

@pytest.mark.unit
def test_logger_network_output(base_logger, caplog):
    base_logger.network("GET https://api.openai.com", method="GET")
    assert "api.openai.com" in caplog.text

@pytest.mark.unit
def test_logger_severity_levels(base_logger, caplog):
    base_logger.warn("Warning msg")
    base_logger.error("Error msg")
    base_logger.critical("Failure msg")
    out = caplog.text
    assert "Warning" in out
    assert "Error" in out
    assert "Failure" in out 

@pytest.mark.unit
def test_logger_success_and_setup(base_logger, caplog):
    base_logger.success("Operation complete")
    base_logger.setup("Environment ready")
    out = caplog.text
    assert "Operation" in out and "Environment" in out

# --- FORMATTING & PAYLOAD EDGE CASES ---

@pytest.mark.unit
def test_truncate_message_at_limit(base_logger):
    """Verifies message longer than truncate_length -> ends with '...'"""
    long_msg = "A" * 100
    truncated = base_logger._truncate_message(long_msg)
    assert len(truncated) == 53
    assert truncated.endswith("...")

@pytest.mark.unit
def test_logger_short_message_handling(base_logger):
    """Verifies that messages under the truncate limit are untouched"""
    msg = "Short"
    assert base_logger._truncate_message(msg) == "Short"

@pytest.mark.unit
def test_format_payload_hides_by_default(monkeypatch):
    """Verifies dict -> '[payload hidden]' when flag is false"""
    monkeypatch.setenv("AGENTFIELD_LOG_PAYLOADS", "false")
    logger = AgentFieldLogger(name="privacy-test")
    test_data = {"secret": "key"}
    formatted = logger._format_payload(test_data)
    assert formatted == "[payload hidden - set AGENTFIELD_LOG_PAYLOADS=true to show]"

@pytest.mark.unit
def test_format_payload_shows_when_enabled(base_logger):
    """Verifies AGENTFIELD_LOG_PAYLOADS=true -> JSON string"""
    test_data = {"id": "123", "meta": "data"}
    formatted = base_logger._format_payload(test_data)
    decoded = json.loads(formatted)
    assert decoded["id"] == "123"

@pytest.mark.unit
def test_format_payload_handles_non_serializable(base_logger):
    """Verifies fallback to str() for objects with no __dict__ or JSON support"""
    test_data = {1, 2, 3}
    formatted = base_logger._format_payload(test_data)
    assert "{1, 2, 3}" in formatted