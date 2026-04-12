import pytest

from agentfield.execution_context import (
    ExecutionContext,
    generate_execution_id,
    set_execution_context,
    reset_execution_context,
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
def test_execution_context_log_fields_default_root_workflow():
    ctx = ExecutionContext(
        workflow_id="wf-1",
        execution_id="exec-1",
        agent_instance=None,
        reasoner_name="reasoner",
        parent_execution_id="parent-1",
        agent_node_id="node-1",
        run_id="run-1",
        parent_workflow_id="wf-parent",
    )

    identity = ctx.to_log_identity()
    attributes = ctx.to_log_attributes()

    assert ctx.root_workflow_id == "wf-1"
    assert identity == {
        "execution_id": "exec-1",
        "workflow_id": "wf-1",
        "run_id": "run-1",
        "root_workflow_id": "wf-1",
        "parent_execution_id": "parent-1",
        "agent_node_id": "node-1",
        "reasoner_id": "reasoner",
    }
    assert attributes["depth"] == 0
    assert attributes["parent_workflow_id"] == "wf-parent"


@pytest.mark.unit
def test_generate_execution_id_has_unique_prefix():
    first = generate_execution_id()
    second = generate_execution_id()

    assert first.startswith("exec_")
    assert second.startswith("exec_")
    assert first != second


@pytest.mark.unit
def test_agent_ctx_property_returns_none_outside_execution():
    """Verify app.ctx returns None when not inside a reasoner/skill execution."""
    from agentfield import Agent

    agent = Agent(node_id="test-ctx-agent")

    # Outside of any execution, ctx should be None
    assert agent.ctx is None


@pytest.mark.unit
def test_agent_ctx_property_returns_context_during_execution():
    """Verify app.ctx returns the execution context when set via thread-local."""
    from agentfield import Agent

    agent = Agent(node_id="test-ctx-agent")

    # Create a registered execution context (simulating incoming request)
    ctx = ExecutionContext(
        workflow_id="wf-test",
        execution_id="exec-test",
        run_id="run-test",
        agent_instance=agent,
        reasoner_name="test_reasoner",
        registered=True,
    )

    # Set the context (simulating what happens during request handling)
    token = set_execution_context(ctx)

    try:
        # Now ctx should be available
        assert agent.ctx is not None
        assert agent.ctx.workflow_id == "wf-test"
        assert agent.ctx.execution_id == "exec-test"
        assert agent.ctx.run_id == "run-test"
        assert agent.ctx.registered is True
    finally:
        # Clean up
        reset_execution_context(token)

    # After reset, ctx should be None again
    assert agent.ctx is None


@pytest.mark.unit
def test_agent_ctx_property_ignores_unregistered_context():
    """Verify app.ctx returns None for unregistered contexts (created at init time)."""
    from agentfield import Agent

    agent = Agent(node_id="test-ctx-agent")

    # The agent creates an unregistered context internally, but ctx should still be None
    # because we only expose registered contexts (from actual executions)
    assert agent.ctx is None

    # Even if we manually set an unregistered context on the agent, ctx should return None
    unregistered_ctx = ExecutionContext(
        workflow_id="wf-unregistered",
        execution_id="exec-unregistered",
        run_id="run-unregistered",
        agent_instance=agent,
        reasoner_name="test",
        registered=False,  # Not from a real request
    )
    agent._current_execution_context = unregistered_ctx

    # Should still return None because it's not registered
    assert agent.ctx is None
