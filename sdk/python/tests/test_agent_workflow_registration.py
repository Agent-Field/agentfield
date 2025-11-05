import pytest

from agentfield.agent_workflow import AgentWorkflow
from agentfield.agent_registry import set_current_agent, clear_current_agent
from agentfield.decorators import _execute_with_tracking
from agentfield.execution_context import (
    ExecutionContext,
    set_execution_context,
    reset_execution_context,
)


class DummyResponse:
    def __init__(self, payload=None, status=200):
        self._payload = payload or {}
        self.status_code = status
        self.status = status

    def json(self):
        return self._payload


class DummyClient:
    def __init__(self):
        self.calls = []

    async def _async_request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        return DummyResponse(
            {
                "execution_id": "exec-registered",
                "workflow_id": "wf-1",
                "run_id": "run-1",
            }
        )


class DummyAgent:
    def __init__(self):
        self.node_id = "agent-node"
        self.agentfield_server = "http://agentfield.local"
        self.client = DummyClient()
        self.dev_mode = False
        self._current_execution_context = None
        self.workflow_handler = None


class DummyWorkflowHandler:
    def __init__(self):
        self.ensure_calls = []
        self.updates = []

    async def _ensure_execution_registered(
        self, context, reasoner_name, parent_context
    ):
        self.ensure_calls.append((context, reasoner_name, parent_context))
        context.execution_id = "exec-child"
        context.registered = True
        return context

    async def fire_and_forget_update(self, payload):
        self.updates.append(payload)


@pytest.mark.asyncio
async def test_child_execution_registration():
    agent = DummyAgent()
    workflow = AgentWorkflow(agent)

    parent = ExecutionContext(
        workflow_id="wf-1",
        execution_id="exec-parent",
        agent_instance=agent,
        reasoner_name="parent",
        run_id="run-1",
        registered=True,
    )

    child = parent.create_child_context()
    assert not child.registered

    await workflow._ensure_execution_registered(child, "child_reasoner", parent)

    assert child.registered is True
    assert child.execution_id == "exec-registered"
    assert child.run_id == "run-1"

    assert agent.client.calls, "registration request not sent"
    method, url, kwargs = agent.client.calls[0]
    assert method == "POST"
    assert url.endswith("/api/v1/workflow/executions")
    assert kwargs["json"]["parent_execution_id"] == parent.execution_id
    assert kwargs["json"]["run_id"] == "run-1"


@pytest.mark.asyncio
async def test_execute_with_tracking_registers_child_context():
    agent = DummyAgent()
    handler = DummyWorkflowHandler()
    agent.workflow_handler = handler

    set_current_agent(agent)

    parent_context = ExecutionContext(
        workflow_id="wf-parent",
        execution_id="exec-parent",
        agent_instance=agent,
        reasoner_name="parent",
        run_id="run-123",
        registered=True,
    )
    parent_context.session_id = "session-abc"
    parent_context.caller_did = "caller"
    parent_context.target_did = "target"
    parent_context.agent_node_did = "agent-did"

    agent._current_execution_context = parent_context
    token = set_execution_context(parent_context)

    async def child_reasoner(value: int) -> int:
        return value * 2

    result = await _execute_with_tracking(child_reasoner, 21)

    assert result == 42
    assert len(handler.ensure_calls) == 1

    registered_context, reasoner_name, parent = handler.ensure_calls[0]
    assert reasoner_name == "child_reasoner"
    assert parent is parent_context
    assert registered_context.run_id == "run-123"
    assert registered_context.parent_execution_id == parent_context.execution_id
    assert registered_context.parent_workflow_id == parent_context.workflow_id
    assert registered_context.registered is True
    assert registered_context.execution_id == "exec-child"

    # Fire-and-forget updates should have been emitted for start and completion
    assert len(handler.updates) == 2
    statuses = {payload["status"] for payload in handler.updates}
    assert statuses == {"running", "succeeded"}
    for payload in handler.updates:
        assert payload["execution_id"] == "exec-child"
        assert payload["parent_execution_id"] == parent_context.execution_id
        assert payload["run_id"] == "run-123"

    # Ensure the agent context is restored after execution
    assert agent._current_execution_context is parent_context

    reset_execution_context(token)
    agent._current_execution_context = None
    clear_current_agent()
