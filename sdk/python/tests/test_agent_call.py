from types import MethodType, SimpleNamespace

import pytest

from haxen_sdk.agent import Agent
from haxen_sdk.agent_registry import set_current_agent, clear_current_agent


@pytest.mark.asyncio
async def test_call_local_reasoner_argument_mapping():
    agent = object.__new__(Agent)
    agent.node_id = "node"
    agent.haxen_connected = True
    agent.dev_mode = False
    agent.async_config = SimpleNamespace(
        enable_async_execution=False, fallback_to_sync=False
    )
    agent._async_execution_manager = None
    agent._current_execution_context = None

    recorded = {}

    async def fake_execute(target, input_data, headers):
        recorded["target"] = target
        recorded["input_data"] = input_data
        recorded["headers"] = headers
        return {"result": {"ok": True}}

    agent.client = SimpleNamespace(execute=fake_execute)

    async def local_reasoner(self, a, b, execution_context=None, extra=None):
        return a + b

    agent.local_reasoner = MethodType(local_reasoner, agent)

    set_current_agent(agent)
    try:
        result = await agent.call("node.local_reasoner", 2, 3, extra=4)
    finally:
        clear_current_agent()

    assert result == {"ok": True}
    assert recorded["target"] == "node.local_reasoner"
    assert recorded["input_data"] == {"a": 2, "b": 3, "extra": 4}
    assert "X-Execution-ID" in recorded["headers"]


@pytest.mark.asyncio
async def test_call_remote_target_uses_generic_arg_names():
    agent = object.__new__(Agent)
    agent.node_id = "node"
    agent.haxen_connected = True
    agent.dev_mode = False
    agent.async_config = SimpleNamespace(
        enable_async_execution=False, fallback_to_sync=False
    )
    agent._async_execution_manager = None
    agent._current_execution_context = None

    recorded = {}

    async def fake_execute(target, input_data, headers):
        recorded["target"] = target
        recorded["input_data"] = input_data
        return {"result": {"value": 10}}

    agent.client = SimpleNamespace(execute=fake_execute)

    set_current_agent(agent)
    try:
        result = await agent.call("other.remote_reasoner", 5, 6)
    finally:
        clear_current_agent()

    assert result == {"value": 10}
    assert recorded["target"] == "other.remote_reasoner"
    assert recorded["input_data"] == {"arg_0": 5, "arg_1": 6}


@pytest.mark.asyncio
async def test_call_raises_when_haxen_disconnected():
    agent = object.__new__(Agent)
    agent.node_id = "node"
    agent.haxen_connected = False
    agent.dev_mode = False
    agent.async_config = SimpleNamespace(
        enable_async_execution=False, fallback_to_sync=False
    )
    agent._async_execution_manager = None
    agent._current_execution_context = None
    agent.client = SimpleNamespace()

    set_current_agent(agent)
    try:
        with pytest.raises(Exception):
            await agent.call("other.reasoner", 1)
    finally:
        clear_current_agent()
