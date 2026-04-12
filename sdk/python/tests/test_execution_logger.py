import json

import pytest

from agentfield.execution_context import (
    ExecutionContext,
    reset_execution_context,
    set_execution_context,
)
from agentfield.logger import log_execution, log_info


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
