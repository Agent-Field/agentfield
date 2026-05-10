from types import MethodType, SimpleNamespace

import pytest

from agentfield.agent import Agent
from agentfield.agent_registry import set_current_agent, clear_current_agent


@pytest.mark.asyncio
async def test_call_local_reasoner_argument_mapping():
    agent = object.__new__(Agent)
    agent.node_id = "node"
    agent.agentfield_connected = True
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
    agent.agentfield_connected = True
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
async def test_call_raises_when_agentfield_disconnected():
    agent = object.__new__(Agent)
    agent.node_id = "node"
    agent.agentfield_connected = False
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


# ---------------------------------------------------------------------------
# fallback_to_sync — must NOT fire on ExecutionFailedError / ExecutionTimeoutError.
#
# Background: a remote reasoner that returns an explicit failed status
# means the work already ran. Re-running it through the sync fallback path
# burns the same per-call budget for the same deterministic outcome — and
# can show up in production as 2× cost on every failed cross-agent call
# (see github-buddy → pr-af.review where pr-af raises BudgetExhaustedError
# and the SDK silently retries it). The fix is to skip the sync fallback
# for these two specific terminal-failure exceptions while keeping it
# enabled for transient transport errors.
# ---------------------------------------------------------------------------


def _make_agent_with_async_path():
    """Build a minimal Agent wired for the async-then-fallback path.

    The async submission succeeds, but the configured async manager raises
    on `wait_for_execution_result`. Whether app.call retries via sync depends
    on the exception class — that's what these tests pin.
    """
    agent = object.__new__(Agent)
    agent.node_id = "node"
    agent.agentfield_connected = True
    agent.dev_mode = False
    agent.async_config = SimpleNamespace(
        enable_async_execution=True,
        fallback_to_sync=True,  # ON — but the new code must skip it for these errors
    )
    agent._async_execution_manager = None
    agent._current_execution_context = None
    return agent


@pytest.mark.asyncio
async def test_call_skips_sync_fallback_on_execution_failed_error():
    """ExecutionFailedError = the reasoner ran and returned an error; the SDK
    must NOT retry via sync (which would re-run the same reasoner and burn
    the same budget for the same outcome)."""
    from agentfield.exceptions import ExecutionFailedError

    agent = _make_agent_with_async_path()

    sync_calls = 0

    async def fake_execute_async(target, input_data, headers, timeout=None):
        return "exec_xyz"

    async def fake_wait_for_execution_result(execution_id, timeout=None):
        # Reasoner ran and returned failed.
        raise ExecutionFailedError("Execution failed: budget exhausted")

    async def fake_execute(target, input_data, headers):
        # If this fires, the SDK incorrectly retried via sync — counts the failure.
        nonlocal sync_calls
        sync_calls += 1
        return {"result": {"never_reached": True}}

    agent.client = SimpleNamespace(
        execute=fake_execute,
        execute_async=fake_execute_async,
        wait_for_execution_result=fake_wait_for_execution_result,
    )

    set_current_agent(agent)
    try:
        with pytest.raises(ExecutionFailedError):
            await agent.call("other.reasoner", 1)
    finally:
        clear_current_agent()

    assert sync_calls == 0, (
        "ExecutionFailedError must NOT trigger sync fallback — "
        "the reasoner already ran and failed deterministically; "
        "retrying just doubles the cost."
    )


@pytest.mark.asyncio
async def test_call_skips_sync_fallback_on_execution_timeout_error():
    """ExecutionTimeoutError = the wait deadline hit; the work either is
    still running on the agent side or already burned its budget. Either
    way, retrying via sync just stacks another full-budget invocation."""
    from agentfield.exceptions import ExecutionTimeoutError

    agent = _make_agent_with_async_path()

    sync_calls = 0

    async def fake_execute_async(target, input_data, headers, timeout=None):
        return "exec_xyz"

    async def fake_wait_for_execution_result(execution_id, timeout=None):
        raise ExecutionTimeoutError("Execution exec_xyz exceeded timeout")

    async def fake_execute(target, input_data, headers):
        nonlocal sync_calls
        sync_calls += 1
        return {"result": {"never_reached": True}}

    agent.client = SimpleNamespace(
        execute=fake_execute,
        execute_async=fake_execute_async,
        wait_for_execution_result=fake_wait_for_execution_result,
    )

    set_current_agent(agent)
    try:
        with pytest.raises(ExecutionTimeoutError):
            await agent.call("other.reasoner", 1)
    finally:
        clear_current_agent()

    assert sync_calls == 0


@pytest.mark.asyncio
async def test_call_still_falls_back_on_transport_errors():
    """Plain AgentFieldClientError (transport / submission / network) MUST
    still trigger the sync fallback. Only post-execution errors are skipped.
    Pin so the fix doesn't accidentally disable retry-on-transport-error."""
    from agentfield.exceptions import AgentFieldClientError

    agent = _make_agent_with_async_path()

    sync_calls = 0

    async def fake_execute_async(target, input_data, headers, timeout=None):
        return "exec_xyz"

    async def fake_wait_for_execution_result(execution_id, timeout=None):
        # Generic transport failure — the kind retry was designed for.
        raise AgentFieldClientError("connection reset by peer")

    async def fake_execute(target, input_data, headers):
        nonlocal sync_calls
        sync_calls += 1
        return {"result": {"recovered": True}}

    agent.client = SimpleNamespace(
        execute=fake_execute,
        execute_async=fake_execute_async,
        wait_for_execution_result=fake_wait_for_execution_result,
    )

    set_current_agent(agent)
    try:
        result = await agent.call("other.reasoner", 1)
    finally:
        clear_current_agent()

    assert sync_calls == 1, (
        "AgentFieldClientError without an execution-side cause must STILL "
        "trigger the sync fallback — that's the recovery path for transport "
        "blips and 502/503s."
    )
    assert result == {"recovered": True}
