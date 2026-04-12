from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from agentfield.agent import Agent
from agentfield.client import ApprovalResult
from agentfield.execution_context import ExecutionContext
from agentfield.exceptions import AgentFieldClientError


def make_agent():
    agent = object.__new__(Agent)
    agent.node_id = "node-1"
    agent.version = "1.2.3"
    agent._reasoner_registry = {"summarize": SimpleNamespace(id="summarize")}
    agent._skill_registry = {"search": SimpleNamespace(id="search")}
    agent._entry_to_metadata = lambda entry, kind: (
        {"id": entry.id, "input_schema": {"type": "object"}, "output_schema": {}, "memory_config": {}, "tags": ["nlp"]}
        if kind == "reasoner"
        else {"id": entry.id, "input_schema": {}, "tags": ["tools"]}
    )
    agent.base_url = "http://agent.test"
    agent.client = SimpleNamespace(
        request_approval=AsyncMock(),
        get_approval_status=AsyncMock(),
    )
    agent._pause_manager = SimpleNamespace(
        register=AsyncMock(),
        resolve=AsyncMock(),
    )
    agent.note = Mock()
    agent.agentfield_server = "http://agentfield.test"
    agent.api_key = "key"
    agent.memory_event_client = None
    agent._current_execution_context = ExecutionContext(
        run_id="run-1",
        execution_id="exec-1",
        agent_instance=None,
        reasoner_name="reasoner",
        workflow_id="wf-1",
        registered=True,
    )
    return agent


def test_handle_discovery_returns_serverless_metadata():
    agent = make_agent()

    payload = agent._handle_discovery()

    assert payload["node_id"] == "node-1"
    assert payload["deployment_type"] == "serverless"
    assert payload["reasoners"][0]["id"] == "summarize"
    assert payload["skills"][0]["id"] == "search"


def test_memory_property_builds_interface_and_reuses_event_client():
    agent = make_agent()

    memory = agent.memory
    memory_again = agent.memory

    assert memory is not None
    assert memory_again is not None
    assert memory.memory_client.agent_node_id == "node-1"
    assert agent.memory_event_client is not None


@pytest.mark.asyncio
async def test_pause_returns_expired_result_on_timeout(monkeypatch):
    agent = make_agent()
    future = asyncio.Future()
    agent._pause_manager.register.return_value = future

    async def raise_timeout(awaitable, timeout):
        raise asyncio.TimeoutError

    monkeypatch.setattr("agentfield.agent.asyncio.wait_for", raise_timeout)

    result = await agent.pause("approval-1", timeout=0.01)

    assert result.decision == "expired"
    assert result.execution_id == "exec-1"
    agent.client.request_approval.assert_awaited_once()
    agent._pause_manager.resolve.assert_awaited_once()
    agent.note.assert_called_once()


@pytest.mark.asyncio
async def test_pause_resolves_error_when_control_plane_request_fails():
    agent = make_agent()
    future = asyncio.Future()
    agent._pause_manager.register.return_value = future
    agent.client.request_approval.side_effect = RuntimeError("cp down")

    with pytest.raises(RuntimeError, match="cp down"):
        await agent.pause("approval-2", timeout=0.01)

    resolved = agent._pause_manager.resolve.await_args.args[1]
    assert resolved.decision == "error"
    assert resolved.approval_request_id == "approval-2"


@pytest.mark.asyncio
async def test_wait_for_resume_falls_back_to_status_poll(monkeypatch):
    agent = make_agent()
    agent._pause_manager.register.return_value = asyncio.Future()
    agent.client.get_approval_status.return_value = SimpleNamespace(
        status="approved",
        response={"decision": "approved"},
    )

    async def raise_timeout(awaitable, timeout):
        raise asyncio.TimeoutError

    monkeypatch.setattr("agentfield.agent.asyncio.wait_for", raise_timeout)

    result = await agent.wait_for_resume("approval-3", timeout=0.01)

    assert isinstance(result, ApprovalResult)
    assert result.decision == "approved"
    assert result.raw_response == {"decision": "approved"}


@pytest.mark.asyncio
async def test_wait_for_resume_returns_expired_when_status_check_fails(monkeypatch):
    agent = make_agent()
    agent._pause_manager.register.return_value = asyncio.Future()
    agent.client.get_approval_status.side_effect = AgentFieldClientError("boom")

    async def raise_timeout(awaitable, timeout):
        raise asyncio.TimeoutError

    monkeypatch.setattr("agentfield.agent.asyncio.wait_for", raise_timeout)

    result = await agent.wait_for_resume("approval-4", timeout=0.01)

    assert result.decision == "expired"
    assert result.approval_request_id == "approval-4"
