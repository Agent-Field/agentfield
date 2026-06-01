"""Extended tests for AgentWorkflow.

Focuses on paths not covered by test_agent_workflow.py:
- replace_function_references updates the agent attribute
- _build_input_payload with various signature shapes
- fire_and_forget_update skips when no client is present
- fire_and_forget_update skips when no agentfield_server is set
- fire_and_forget_update calls client when properly configured
- _ensure_execution_registered marks context registered when no client
- _ensure_execution_registered calls server and updates context IDs
- notify_call_complete and notify_call_error build correct payloads
- Sync (non-async) wrapped functions are executed correctly
- execution_context is NOT injected when signature has no such parameter
- Parent-child workflow context propagation
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from agentfield.agent_workflow import AgentWorkflow
from agentfield.execution_context import ExecutionContext
from agentfield.agent_registry import clear_current_agent, set_current_agent
from tests.helpers import StubAgent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_stub_with_client(*, base_url="http://agentfield", dev_mode=False):
    """StubAgent that has a minimal _async_request client."""
    agent = StubAgent(agentfield_server=base_url, dev_mode=dev_mode)

    class _Client:
        def __init__(self):
            self.calls: List[Dict[str, Any]] = []
            self._current_workflow_context = None

        async def _async_request(self, method: str, url: str, **kwargs):
            self.calls.append({"method": method, "url": url, "kwargs": kwargs})
            # Return a fake response with .json()
            return SimpleNamespace(json=lambda: {"execution_id": "srv-exec-1", "workflow_id": "srv-wf-1", "run_id": "srv-run-1"})

    agent.client = _Client()
    return agent


# ---------------------------------------------------------------------------
# replace_function_references
# ---------------------------------------------------------------------------


def test_replace_function_references_sets_agent_attribute():
    agent = StubAgent()
    workflow = AgentWorkflow(agent)

    def original():
        return "original"

    def tracked():
        return "tracked"

    agent.my_func = original
    workflow.replace_function_references(original, tracked, "my_func")

    assert agent.my_func is tracked


# ---------------------------------------------------------------------------
# _build_input_payload
# ---------------------------------------------------------------------------


def test_build_input_payload_positional_args_mapped_by_param_name():
    def func(a: int, b: str):
        pass

    sig = inspect.signature(func)
    payload = AgentWorkflow._build_input_payload(sig, (1, "hello"), {})

    assert payload == {"a": 1, "b": "hello"}


def test_build_input_payload_kwargs_only():
    def func(x: int, y: int):
        pass

    sig = inspect.signature(func)
    payload = AgentWorkflow._build_input_payload(sig, (), {"x": 10, "y": 20})

    assert payload == {"x": 10, "y": 20}


def test_build_input_payload_skips_self_parameter():
    class _Obj:
        def method(self, value: int):
            pass

    sig = inspect.signature(_Obj.method)
    payload = AgentWorkflow._build_input_payload(sig, (_Obj(), 42), {})

    assert "self" not in payload
    assert payload.get("value") == 42


def test_build_input_payload_empty_signature_returns_kwargs():
    sig = inspect.Signature()
    payload = AgentWorkflow._build_input_payload(sig, (), {"key": "val"})

    assert payload == {"key": "val"}


def test_build_input_payload_with_defaults():
    def func(a: int, b: str = "default"):
        pass

    sig = inspect.signature(func)
    payload = AgentWorkflow._build_input_payload(sig, (5,), {})

    assert payload["a"] == 5
    assert payload["b"] == "default"


# ---------------------------------------------------------------------------
# fire_and_forget_update — no client / no base_url
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fire_and_forget_update_no_op_when_no_client():
    agent = StubAgent()
    agent.client = None
    workflow = AgentWorkflow(agent)

    # Should not raise
    await workflow.fire_and_forget_update({"event": "test"})


@pytest.mark.asyncio
async def test_fire_and_forget_update_no_op_when_no_agentfield_server():
    agent = StubAgent()
    agent.agentfield_server = None

    class _ClientNoRequest:
        _current_workflow_context = None

    agent.client = _ClientNoRequest()
    workflow = AgentWorkflow(agent)

    await workflow.fire_and_forget_update({"event": "test"})
    # No exception means the guard worked correctly


@pytest.mark.asyncio
async def test_fire_and_forget_update_calls_client_when_configured():
    agent = _make_stub_with_client()
    workflow = AgentWorkflow(agent)

    payload = {"execution_id": "abc", "status": "running"}
    await workflow.fire_and_forget_update(payload)

    assert len(agent.client.calls) == 1
    call = agent.client.calls[0]
    assert call["method"] == "POST"
    assert "/api/v1/workflow/executions/events" in call["url"]


# ---------------------------------------------------------------------------
# _ensure_execution_registered — no client skips registration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ensure_execution_registered_marks_registered_when_no_client():
    agent = StubAgent()
    agent.client = None
    workflow = AgentWorkflow(agent)

    ctx = ExecutionContext.create_new(agent.node_id, "my_reasoner")
    assert not ctx.registered

    result = await workflow._ensure_execution_registered(ctx, "my_reasoner", None)

    assert result.registered is True


@pytest.mark.asyncio
async def test_ensure_execution_registered_skips_if_already_registered():
    agent = _make_stub_with_client()
    workflow = AgentWorkflow(agent)

    ctx = ExecutionContext.create_new(agent.node_id, "test")
    ctx.registered = True

    result = await workflow._ensure_execution_registered(ctx, "test", None)

    # Client should NOT be called since already registered
    assert len(agent.client.calls) == 0
    assert result is ctx


@pytest.mark.asyncio
async def test_ensure_execution_registered_calls_server_and_updates_ids():
    agent = _make_stub_with_client()
    workflow = AgentWorkflow(agent)

    ctx = ExecutionContext.create_new(agent.node_id, "my_reasoner")
    assert not ctx.registered

    result = await workflow._ensure_execution_registered(ctx, "my_reasoner", None)

    assert result.registered is True
    # Server should have been called once (POST /api/v1/workflow/executions)
    assert len(agent.client.calls) == 1
    assert "/api/v1/workflow/executions" in agent.client.calls[0]["url"]
    # IDs should be updated from the server response
    assert result.execution_id == "srv-exec-1"
    assert result.workflow_id == "srv-wf-1"


# ---------------------------------------------------------------------------
# notify_call_complete — payload structure
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_notify_call_complete_payload_has_result_and_duration(monkeypatch):
    agent = StubAgent()
    workflow = AgentWorkflow(agent)

    ctx = ExecutionContext.create_new(agent.node_id, "worker")
    ctx.reasoner_name = "worker"

    payloads: List[Dict[str, Any]] = []

    async def capture(payload):
        payloads.append(payload)

    monkeypatch.setattr(workflow, "fire_and_forget_update", capture)

    await workflow.notify_call_complete(
        ctx.execution_id,
        ctx.workflow_id,
        {"output": 42},
        150,
        ctx,
        input_data={"x": 1},
        parent_execution_id="parent-id",
    )

    assert len(payloads) == 1
    p = payloads[0]
    assert p["result"] == {"output": 42}
    assert p["duration_ms"] == 150
    assert p["status"] == "succeeded"
    assert p["input_data"] == {"x": 1}
    assert p["parent_execution_id"] == "parent-id"


# ---------------------------------------------------------------------------
# notify_call_error — payload structure
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_notify_call_error_payload_has_error_and_duration(monkeypatch):
    agent = StubAgent()
    workflow = AgentWorkflow(agent)

    ctx = ExecutionContext.create_new(agent.node_id, "worker")
    ctx.reasoner_name = "worker"

    payloads: List[Dict[str, Any]] = []

    async def capture(payload):
        payloads.append(payload)

    monkeypatch.setattr(workflow, "fire_and_forget_update", capture)

    await workflow.notify_call_error(
        ctx.execution_id,
        ctx.workflow_id,
        "something went wrong",
        200,
        ctx,
        input_data={"x": 1},
        parent_execution_id=None,
    )

    assert len(payloads) == 1
    p = payloads[0]
    assert p["error"] == "something went wrong"
    assert p["duration_ms"] == 200
    assert p["status"] == "failed"


# ---------------------------------------------------------------------------
# execute_with_tracking — sync (non-async) function
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_with_tracking_sync_function(monkeypatch):
    agent = StubAgent()
    workflow = AgentWorkflow(agent)

    async def noop_start(*args, **kwargs):
        pass

    async def noop_complete(*args, **kwargs):
        pass

    monkeypatch.setattr(workflow, "notify_call_start", noop_start)
    monkeypatch.setattr(workflow, "notify_call_complete", noop_complete)

    def sync_func(x: int) -> int:
        return x * 3

    result = await workflow.execute_with_tracking(sync_func, (7,), {})

    assert result == 21


# ---------------------------------------------------------------------------
# execute_with_tracking — no execution_context param in signature
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_with_tracking_no_context_param_not_injected(monkeypatch):
    agent = StubAgent()
    workflow = AgentWorkflow(agent)

    async def noop(*args, **kwargs):
        pass

    monkeypatch.setattr(workflow, "notify_call_start", noop)
    monkeypatch.setattr(workflow, "notify_call_complete", noop)

    received_kwargs: Dict[str, Any] = {}

    async def my_func(value: int):
        received_kwargs["value"] = value
        assert "execution_context" not in received_kwargs
        return value

    result = await workflow.execute_with_tracking(my_func, (99,), {})

    assert result == 99
    assert "execution_context" not in received_kwargs


# ---------------------------------------------------------------------------
# Parent-child context propagation via agent._current_execution_context
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_child_execution_inherits_parent_workflow_id(monkeypatch):
    agent = StubAgent()
    workflow = AgentWorkflow(agent)

    parent_ctx = ExecutionContext.create_new(agent.node_id, "parent")
    agent._current_execution_context = parent_ctx

    child_contexts: List[ExecutionContext] = []

    async def noop_start(execution_id, context, reasoner_name, input_data, parent_execution_id=None):
        child_contexts.append(context)

    async def noop_complete(*args, **kwargs):
        pass

    monkeypatch.setattr(workflow, "notify_call_start", noop_start)
    monkeypatch.setattr(workflow, "notify_call_complete", noop_complete)

    async def child_func(execution_context: ExecutionContext = None):
        return "ok"

    set_current_agent(agent)
    try:
        await workflow.execute_with_tracking(child_func, (), {})
    finally:
        clear_current_agent()
        agent._current_execution_context = None

    assert len(child_contexts) == 1
    child_ctx = child_contexts[0]
    assert child_ctx.workflow_id == parent_ctx.workflow_id
    assert child_ctx.parent_execution_id == parent_ctx.execution_id


# ---------------------------------------------------------------------------
# Context cleanup — agent._current_execution_context restored after error
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execution_context_restored_after_exception(monkeypatch):
    agent = StubAgent()
    workflow = AgentWorkflow(agent)

    previous_ctx = ExecutionContext.create_new(agent.node_id, "previous")
    agent._current_execution_context = previous_ctx

    async def noop(*args, **kwargs):
        pass

    monkeypatch.setattr(workflow, "notify_call_start", noop)
    monkeypatch.setattr(workflow, "notify_call_error", noop)

    async def raises():
        raise ValueError("test error")

    with pytest.raises(ValueError):
        await workflow.execute_with_tracking(raises, (), {})

    # The context should be restored to the previous one
    assert agent._current_execution_context is previous_ctx
