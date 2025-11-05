import pytest

from agentfield.execution_context import (
    ExecutionContext,
    generate_execution_id,
)


@pytest.mark.unit
def test_to_headers_includes_optional_fields():
    ctx = ExecutionContext(
        workflow_id="wf-1",
        execution_id="exec-1",
        agent_instance=None,
        reasoner_name="reasoner",
        parent_execution_id="parent-1",
        parent_workflow_id="wf-parent",
        session_id="sess-1",
        caller_did="did:caller",
        target_did="did:target",
        agent_node_did="did:agent",
        run_id="run-1",
    )

    headers = ctx.to_headers()

    assert headers["X-Workflow-ID"] == "wf-1"
    assert headers["X-Execution-ID"] == "exec-1"
    assert headers["X-Parent-Execution-ID"] == "parent-1"
    assert headers["X-Parent-Workflow-ID"] == "wf-parent"
    assert headers["X-Session-ID"] == "sess-1"
    assert headers["X-Caller-DID"] == "did:caller"
    assert headers["X-Target-DID"] == "did:target"
    assert headers["X-Agent-Node-DID"] == "did:agent"
    assert headers["X-Workflow-Run-ID"] == "run-1"


@pytest.mark.unit
def test_child_context_derives_from_parent():
    root = ExecutionContext(
        workflow_id="wf-1",
        execution_id="exec-1",
        agent_instance=None,
        reasoner_name="root",
        depth=0,
        run_id="run-1",
    )

    child = root.create_child_context()

    assert child.workflow_id == root.workflow_id
    assert child.parent_execution_id == root.execution_id
    assert child.parent_workflow_id == root.workflow_id
    assert child.depth == root.depth + 1
    assert child.execution_id.startswith("exec_")
    assert child.execution_id != root.execution_id
    assert child.run_id == root.run_id
    assert not child.registered


@pytest.mark.unit
def test_generate_execution_id_has_unique_prefix():
    first = generate_execution_id()
    second = generate_execution_id()

    assert first.startswith("exec_")
    assert second.startswith("exec_")
    assert first != second
