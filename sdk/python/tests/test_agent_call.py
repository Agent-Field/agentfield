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
async def test_call_skips_sync_fallback_on_execution_cancelled_error():
    """ExecutionCancelledError = the user explicitly cancelled the awaited
    child (typically via the control plane's cancel-tree endpoint). The SDK
    must NOT retry via sync — silently re-issuing a cancelled call defeats
    the cancellation and re-runs work the user told the system to abandon.
    Repro: github-buddy → pr-af.review run cancelled mid-flight; pr-af got
    invoked again seconds later because the cancellation surfaced as a
    plain AgentFieldClientError and slipped past the skip-list."""
    from agentfield.exceptions import ExecutionCancelledError

    agent = _make_agent_with_async_path()

    sync_calls = 0

    async def fake_execute_async(target, input_data, headers, timeout=None):
        return "exec_xyz"

    async def fake_wait_for_execution_result(execution_id, timeout=None):
        raise ExecutionCancelledError(
            "Execution was cancelled: user clicked cancel"
        )

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
        with pytest.raises(ExecutionCancelledError):
            await agent.call("other.reasoner", 1)
    finally:
        clear_current_agent()

    assert sync_calls == 0, (
        "ExecutionCancelledError must NOT trigger sync fallback — "
        "the user explicitly told the system to stop; silently re-issuing "
        "the call defeats the cancellation."
    )


@pytest.mark.asyncio
async def test_call_wires_awaiter_status_callbacks_for_multihop_pause():
    """Agent.call must pass on_child_waiting / on_child_running callbacks to
    wait_for_execution_result, and those callbacks must hit
    client.notify_awaiter_status(execution_id=<MY exec_id>, status=...).

    Without this wiring, multi-hop pause propagation breaks: a grandparent
    awaiting a middle reasoner only ever sees middle as RUNNING (because
    middle's status doesn't transition while it's blocked on its own
    awaited child), so grandparent's pause_clock never pauses and it times
    out at wallclock. This pins the contract end-to-end through Agent.call.
    """
    from agentfield.agent_pause import PauseClock
    from agentfield.execution_context import (
        ExecutionContext,
        set_execution_context,
        reset_execution_context,
    )

    agent = object.__new__(Agent)
    agent.node_id = "middle"
    agent.agentfield_connected = True
    agent.dev_mode = False
    agent.async_config = SimpleNamespace(
        enable_async_execution=True,
        fallback_to_sync=True,
    )
    agent._async_execution_manager = None
    agent._current_execution_context = None
    # PauseClock registry — this is how Agent.call recovers the parent's clock.
    agent._pause_clocks = {"middle-exec-id": PauseClock()}

    notify_calls: list[dict] = []

    async def fake_notify_awaiter_status(execution_id, status, reason=""):
        notify_calls.append(
            {"execution_id": execution_id, "status": status, "reason": reason}
        )

    captured_wait_kwargs: dict = {}

    async def fake_execute_async(target, input_data, headers, timeout=None):
        return "exec_child_xyz"

    async def fake_wait_for_execution_result(**kwargs):
        # Capture so we can verify the kwargs Agent.call constructed.
        captured_wait_kwargs.update(kwargs)
        # Simulate the wait loop firing the on_child_waiting / on_child_running
        # callbacks the way wait_for_result would when the awaited child enters
        # and leaves WAITING.
        if "on_child_waiting" in kwargs:
            await kwargs["on_child_waiting"]()
        if "on_child_running" in kwargs:
            await kwargs["on_child_running"]()
        return {"result": {"ok": True}}

    agent.client = SimpleNamespace(
        execute_async=fake_execute_async,
        wait_for_execution_result=fake_wait_for_execution_result,
        notify_awaiter_status=fake_notify_awaiter_status,
    )

    ctx = ExecutionContext(
        run_id="run-123",
        execution_id="middle-exec-id",
        agent_instance=agent,
        reasoner_name="middle.process",
        agent_node_id="middle",
    )
    ctx_token = set_execution_context(ctx)
    set_current_agent(agent)
    try:
        result = await agent.call("other.reasoner", 1)
    finally:
        clear_current_agent()
        reset_execution_context(ctx_token)

    assert result == {"ok": True}

    # The callbacks must have been passed.
    assert "on_child_waiting" in captured_wait_kwargs, (
        "Agent.call must pass on_child_waiting to wait_for_execution_result "
        "when it has a parent pause_clock — otherwise multi-hop propagation "
        "is impossible"
    )
    assert "on_child_running" in captured_wait_kwargs

    # The callbacks must target THIS reasoner's own execution_id, not the
    # child's — pushing the child's status would be a no-op (child already
    # handles its own status).
    waiting_calls = [c for c in notify_calls if c["status"] == "waiting"]
    running_calls = [c for c in notify_calls if c["status"] == "running"]
    assert len(waiting_calls) == 1, (
        f"on_child_waiting must call notify_awaiter_status once; got {notify_calls}"
    )
    assert len(running_calls) == 1
    assert waiting_calls[0]["execution_id"] == "middle-exec-id", (
        f"awaiter-status update must target the awaiter's own execution_id, "
        f"not the child's; got {waiting_calls[0]['execution_id']}"
    )
    assert running_calls[0]["execution_id"] == "middle-exec-id"
    # Reason should reference the awaited child for observability.
    assert "exec_child_xyz" in waiting_calls[0]["reason"]


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
