import asyncio
import time

import httpx
import pytest

from agentfield.agent import Agent
from agentfield.agent_pause import PauseClock
from agentfield.client import AgentFieldClient, ApprovalResult


@pytest.mark.asyncio
async def test_reasoner_async_mode_sends_status(monkeypatch):
    agent = Agent(
        node_id="test-agent", agentfield_server="http://control", auto_register=False
    )

    @agent.reasoner()
    async def echo(value: int) -> dict:
        await asyncio.sleep(0)
        return {"value": value}

    recorded = []

    class DummyResponse:
        def __init__(self, status_code: int = 200):
            self.status_code = status_code

        def json(self):
            return {}

    async def fake_request(self, method, url, **kwargs):
        recorded.append({"method": method, "url": url, "json": kwargs.get("json")})
        return DummyResponse(200)

    monkeypatch.setattr(AgentFieldClient, "_async_request", fake_request)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=agent), base_url="http://agent"
    ) as client:
        response = await client.post(
            "/reasoners/echo",
            json={"value": 7},
            headers={"X-Execution-ID": "exec-123"},
        )

    assert response.status_code == 202
    await asyncio.sleep(0.1)

    status_calls = [entry for entry in recorded if "/executions/" in entry["url"]]
    assert status_calls, "expected async status callback"
    payload = status_calls[-1]["json"]
    assert payload["status"] == "succeeded"
    assert payload["result"]["value"] == 7


@pytest.mark.asyncio
async def test_post_execution_status_retries(monkeypatch):
    agent = Agent(
        node_id="test-agent", agentfield_server="http://control", auto_register=False
    )

    attempts = {"count": 0}

    class DummyResponse:
        def __init__(self, status_code: int):
            self.status_code = status_code

    async def fake_request(self, method, url, **kwargs):
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise RuntimeError("transient error")
        return DummyResponse(200)

    monkeypatch.setattr(AgentFieldClient, "_async_request", fake_request)

    sleeps = []

    async def fake_sleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    await agent._post_execution_status(
        "http://control/api/v1/executions/exec-1/status",
        {"status": "running"},
        "exec-1",
        max_retries=5,
    )

    assert attempts["count"] == 3
    assert sleeps == [1, 2]


@pytest.mark.asyncio
async def test_pause_does_not_consume_active_timeout_budget(monkeypatch):
    """A reasoner paused in ``app.pause()`` for longer than the wall-clock
    timeout should still succeed once the approval webhook resolves it.

    The reasoner-level timeout is supposed to bound *active* time (so a hung
    reasoner can't run forever) — not human-response time, which is governed
    by ``expires_in_hours``. Without the pause-clock subtraction, the outer
    timeout silently caps every approval at the reasoner timeout.
    """
    agent = Agent(
        node_id="test-agent",
        agentfield_server="http://control",
        auto_register=False,
    )
    agent.base_url = "http://agent"
    agent.async_config.default_execution_timeout = 1.0

    pause_duration = 2.0  # > default_execution_timeout

    @agent.reasoner()
    async def needs_approval(prompt: str) -> dict:
        result = await agent.pause(
            approval_request_id="req-1",
            approval_request_url="http://hax/approvals/req-1",
            expires_in_hours=24,
        )
        return {"decision": result.decision}

    recorded: list[dict] = []

    class DummyResponse:
        def __init__(self, status_code: int = 200):
            self.status_code = status_code

        def json(self):
            return {}

    async def fake_request(self, method, url, **kwargs):
        recorded.append({"method": method, "url": url, "json": kwargs.get("json")})
        return DummyResponse(200)

    monkeypatch.setattr(AgentFieldClient, "_async_request", fake_request)

    async def fake_request_approval(*args, **kwargs):
        return None

    monkeypatch.setattr(agent.client, "request_approval", fake_request_approval)

    async def resolve_after_delay():
        await asyncio.sleep(pause_duration)
        await agent._pause_manager.resolve(
            "req-1",
            ApprovalResult(
                decision="approved",
                execution_id="exec-pause-1",
                approval_request_id="req-1",
            ),
        )

    resolver = asyncio.create_task(resolve_after_delay())

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=agent), base_url="http://agent"
    ) as client:
        response = await client.post(
            "/reasoners/needs_approval",
            json={"prompt": "ship it?"},
            headers={"X-Execution-ID": "exec-pause-1"},
        )

    assert response.status_code == 202

    # Wait for the resolver to fire and the reasoner to post its terminal
    # status callback (the running-event broadcasts are not what we want).
    await resolver

    def terminal_calls():
        out = []
        for e in recorded:
            body = e.get("json") or {}
            if body.get("status") in {"succeeded", "failed", "cancelled"}:
                out.append(e)
        return out

    for _ in range(30):
        await asyncio.sleep(0.1)
        if terminal_calls():
            break

    status_calls = terminal_calls()
    assert status_calls, "expected terminal async status callback after pause resolved"
    payload = status_calls[-1]["json"]
    assert payload["status"] == "succeeded", (
        f"reasoner timed out while paused; payload={payload}"
    )
    assert payload["result"]["decision"] == "approved"


@pytest.mark.asyncio
async def test_active_work_past_timeout_still_times_out(monkeypatch):
    """A reasoner doing real CPU/IO work past the active budget must still
    time out — the pause-clock subtraction must not disable the watchdog.
    """
    agent = Agent(
        node_id="test-agent",
        agentfield_server="http://control",
        auto_register=False,
    )
    agent.async_config.default_execution_timeout = 0.5

    @agent.reasoner()
    async def slow_work(value: int) -> dict:
        await asyncio.sleep(2.0)
        return {"value": value}

    recorded: list[dict] = []

    class DummyResponse:
        def __init__(self, status_code: int = 200):
            self.status_code = status_code

        def json(self):
            return {}

    async def fake_request(self, method, url, **kwargs):
        recorded.append({"method": method, "url": url, "json": kwargs.get("json")})
        return DummyResponse(200)

    monkeypatch.setattr(AgentFieldClient, "_async_request", fake_request)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=agent), base_url="http://agent"
    ) as client:
        response = await client.post(
            "/reasoners/slow_work",
            json={"value": 1},
            headers={"X-Execution-ID": "exec-timeout-1"},
        )

    assert response.status_code == 202

    def terminal_calls():
        out = []
        for e in recorded:
            body = e.get("json") or {}
            if body.get("status") in {"succeeded", "failed", "cancelled"}:
                out.append(e)
        return out

    for _ in range(40):
        await asyncio.sleep(0.1)
        if terminal_calls():
            break

    status_calls = terminal_calls()
    assert status_calls, "expected terminal async status callback after timeout"
    payload = status_calls[-1]["json"]
    assert payload["status"] == "failed"
    assert payload["error_details"]["reason"] == "reasoner_timeout"


# ---------------------------------------------------------------------------
# Direct unit tests for the PauseClock primitive.  The Agent-level tests above
# exercise the same machinery end-to-end, but pinning behaviour at this layer
# protects the primitive from accidental refactors and gives a faster failure
# signal when the contract changes.


def test_pause_clock_starts_with_zero_paused():
    clock = PauseClock()
    assert clock.total_paused() == 0.0
    assert clock.timed_out is False


def test_pause_clock_accumulates_completed_intervals():
    clock = PauseClock()

    clock.start_pause()
    time.sleep(0.05)
    clock.end_pause()
    first = clock.total_paused()
    assert first >= 0.05

    clock.start_pause()
    time.sleep(0.05)
    clock.end_pause()
    second = clock.total_paused()
    assert second >= first + 0.05


def test_pause_clock_includes_in_progress_pause():
    clock = PauseClock()
    clock.start_pause()
    time.sleep(0.05)
    # Without ending the pause we should still see the elapsed time so the
    # watchdog doesn't trip while a long pause is mid-flight.
    mid = clock.total_paused()
    assert mid >= 0.05
    clock.end_pause()
    assert clock.total_paused() >= mid


def test_pause_clock_double_start_is_idempotent():
    clock = PauseClock()
    clock.start_pause()
    time.sleep(0.05)
    # A second start_pause must not reset the in-progress interval — otherwise
    # nested awaits inside pause() could silently zero the paused duration.
    clock.start_pause()
    clock.end_pause()
    assert clock.total_paused() >= 0.05


def test_pause_clock_end_without_start_is_safe():
    clock = PauseClock()
    clock.end_pause()
    assert clock.total_paused() == 0.0


@pytest.mark.asyncio
async def test_external_cancel_during_pause_reports_cancelled_not_timeout(monkeypatch):
    """An external cooperative cancel that arrives while the reasoner is
    inside ``app.pause()`` must surface as ``cancelled`` — not as a phantom
    timeout. The watchdog distinguishes its own timeout-cancel from external
    cancels by reading ``PauseClock.timed_out``.
    """
    agent = Agent(
        node_id="test-agent",
        agentfield_server="http://control",
        auto_register=False,
    )
    agent.base_url = "http://agent"
    # Generous active budget so the watchdog cannot fire while we cancel.
    agent.async_config.default_execution_timeout = 60.0

    @agent.reasoner()
    async def needs_approval(prompt: str) -> dict:
        result = await agent.pause(
            approval_request_id="req-cancel",
            approval_request_url="http://hax/approvals/req-cancel",
            expires_in_hours=24,
        )
        return {"decision": result.decision}

    recorded: list[dict] = []

    class DummyResponse:
        def __init__(self, status_code: int = 200):
            self.status_code = status_code

        def json(self):
            return {}

    async def fake_request(self, method, url, **kwargs):
        recorded.append({"method": method, "url": url, "json": kwargs.get("json")})
        return DummyResponse(200)

    monkeypatch.setattr(AgentFieldClient, "_async_request", fake_request)

    async def fake_request_approval(*args, **kwargs):
        return None

    monkeypatch.setattr(agent.client, "request_approval", fake_request_approval)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=agent), base_url="http://agent"
    ) as client:
        response = await client.post(
            "/reasoners/needs_approval",
            json={"prompt": "ship it?"},
            headers={"X-Execution-ID": "exec-cancel-1"},
        )

    assert response.status_code == 202

    # Give the reasoner task a tick to enter pause(), then cooperatively
    # cancel via the same path the control plane uses.
    await asyncio.sleep(0.1)
    from agentfield.cancel import cancel_execution

    cancelled = await cancel_execution(agent, "exec-cancel-1")
    assert cancelled is True

    def terminal_calls():
        out = []
        for e in recorded:
            body = e.get("json") or {}
            if body.get("status") in {"succeeded", "failed", "cancelled"}:
                out.append(e)
        return out

    for _ in range(30):
        await asyncio.sleep(0.1)
        if terminal_calls():
            break

    status_calls = terminal_calls()
    assert status_calls, "expected terminal callback after external cancel"
    payload = status_calls[-1]["json"]
    assert payload["status"] == "cancelled", (
        f"external cancel during pause should not be reported as timeout; "
        f"payload={payload}"
    )


# ---------------------------------------------------------------------------
# Cross-reasoner pause propagation
#
# When a parent reasoner calls a child via ``app.call`` and the child enters
# an ``app.pause`` waiting state, the parent's pause-clock must be paused too
# — otherwise the parent's wall-clock budget keeps ticking through the
# child's human-approval delay and the parent times out at 7200s while the
# child is correctly waiting. These tests pin the propagation logic.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_wait_for_result_pauses_parent_clock_while_child_waits():
    """End-to-end check on the production path: the polling task updates
    ``_executions[child].status`` from RUNNING -> WAITING -> RUNNING -> SUCCEEDED,
    and ``wait_for_result`` (with a pause_clock) toggles the parent's clock
    accordingly. No SSE listener, no internal hooks — just the data path the
    deployed services actually use.

    Regression for run_1778268481826_8c9dd544 where the parent watchdog tripped
    at exactly the wallclock budget despite a long approval wait, because the
    listener-based mechanism was gated off behind ``enable_event_stream`` and
    never fired in production. Polling is unconditional, so this version of
    the propagation works regardless of the SSE flag.
    """
    from agentfield.async_execution_manager import AsyncExecutionManager
    from agentfield.async_config import AsyncConfig
    from agentfield.execution_state import ExecutionState, ExecutionStatus

    manager = AsyncExecutionManager(base_url="http://control", config=AsyncConfig())
    exec_id = "exec-child"
    state = ExecutionState(
        execution_id=exec_id,
        target="agent.reasoner",
        input_data={},
        timeout=5.0,
    )
    state.update_status(ExecutionStatus.RUNNING)
    manager._executions[exec_id] = state

    parent_clock = PauseClock()

    async def _drive_polling():
        # t=0.05: child enters WAITING (e.g. hax-sdk approval gate fires).
        await asyncio.sleep(0.05)
        async with manager._execution_lock:
            state.update_status(ExecutionStatus.WAITING)
        # t=0.25: child resumes after ~0.2s of waiting.
        await asyncio.sleep(0.20)
        async with manager._execution_lock:
            state.update_status(ExecutionStatus.RUNNING)
        # t=0.30: child finishes shortly after resuming.
        await asyncio.sleep(0.05)
        async with manager._execution_lock:
            state.set_result({"ok": True})

    poller = asyncio.create_task(_drive_polling())
    result = await manager.wait_for_result(
        exec_id, timeout=5.0, pause_clock=parent_clock
    )
    await poller

    assert result == {"ok": True}
    paused = parent_clock.total_paused()
    # The WAITING window was ~0.2s. Allow generous slack for scheduler jitter.
    assert paused >= 0.15, (
        f"parent clock should reflect the child's WAITING window, got {paused:.3f}s"
    )
    assert paused < 0.5, (
        f"parent clock should not include time outside WAITING, got {paused:.3f}s"
    )


@pytest.mark.asyncio
async def test_wait_for_result_preserves_budget_across_long_wait():
    """The headline scenario: the WAITING window is longer than the entire
    wait_timeout, but ``wait_for_result`` must still complete because
    waiting time is excluded from active-elapsed. Without the pause toggle
    in the wait loop this would (and previously did) trip the timeout."""
    from agentfield.async_execution_manager import AsyncExecutionManager
    from agentfield.async_config import AsyncConfig
    from agentfield.execution_state import ExecutionState, ExecutionStatus

    manager = AsyncExecutionManager(base_url="http://control", config=AsyncConfig())
    exec_id = "exec-long-wait"
    state = ExecutionState(
        execution_id=exec_id,
        target="agent.reasoner",
        input_data={},
        timeout=0.5,
    )
    state.update_status(ExecutionStatus.RUNNING)
    manager._executions[exec_id] = state

    parent_clock = PauseClock()

    async def _drive_polling():
        await asyncio.sleep(0.05)
        async with manager._execution_lock:
            state.update_status(ExecutionStatus.WAITING)
        # WAITING for 0.7s — longer than the 0.5s wait_timeout. With the
        # pause_clock active this whole window is excluded from active time,
        # so the wait must NOT raise ExecutionTimeoutError here.
        await asyncio.sleep(0.7)
        async with manager._execution_lock:
            state.update_status(ExecutionStatus.RUNNING)
        await asyncio.sleep(0.05)
        async with manager._execution_lock:
            state.set_result({"value": 42})

    poller = asyncio.create_task(_drive_polling())
    result = await manager.wait_for_result(
        exec_id, timeout=0.5, pause_clock=parent_clock
    )
    await poller

    assert result == {"value": 42}


# ---------------------------------------------------------------------------
# Multi-hop pause propagation
#
# A reasoner R that's awaiting a child via app.call() blocks for the duration
# the child sits in WAITING. Today only R's *local* pause_clock pauses — R's
# *status* in the control plane stays RUNNING. If R has its own parent G,
# G's wait loop polls R's status, sees RUNNING, and never pauses G's clock —
# producing the 7200s timeout on the great-grandparent observed in run
# run_1778429268006_76e417b7 (implement_from_issue → build → plan → run_X
# where only run_X explicitly called app.pause()).
#
# The fix: wait_for_result accepts on_child_waiting / on_child_running async
# callbacks, fired in lockstep with start_pause / end_pause on the local
# clock. Agent.call wires these to push the awaiter's OWN status as WAITING
# to the control plane, so the next hop up sees WAITING and the chain
# propagates transparently to arbitrary depth.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_wait_for_result_invokes_callbacks_on_child_waiting_transitions():
    """When the awaited child enters WAITING, ``wait_for_result`` must fire
    the ``on_child_waiting`` callback alongside ``pause_clock.start_pause()``.
    When the child exits WAITING, it must fire ``on_child_running`` alongside
    ``end_pause``. This is the hook ``Agent.call`` uses to push its own
    execution's status upstream so multi-hop pause propagation works.

    Pins the new SDK contract — without these callbacks the SDK has no way
    to tell the control plane "I am also waiting now" while it's blocked in
    its own wait loop, and any ancestor 2+ hops up times out at wallclock."""
    from agentfield.async_execution_manager import AsyncExecutionManager
    from agentfield.async_config import AsyncConfig
    from agentfield.execution_state import ExecutionState, ExecutionStatus

    manager = AsyncExecutionManager(base_url="http://control", config=AsyncConfig())
    exec_id = "exec-multihop-child"
    state = ExecutionState(
        execution_id=exec_id,
        target="agent.reasoner",
        input_data={},
        timeout=5.0,
    )
    state.update_status(ExecutionStatus.RUNNING)
    manager._executions[exec_id] = state

    parent_clock = PauseClock()
    waiting_events: list[float] = []
    running_events: list[float] = []

    async def on_child_waiting() -> None:
        waiting_events.append(time.time())

    async def on_child_running() -> None:
        running_events.append(time.time())

    async def _drive_polling():
        await asyncio.sleep(0.05)
        async with manager._execution_lock:
            state.update_status(ExecutionStatus.WAITING)
        # Keep the WAITING and RUNNING windows comfortably above the wait
        # loop's 0.1s poll interval so the loop is guaranteed to observe
        # each transition before the next one happens. The original 0.05s
        # post-RUNNING gap raced with the loop's poll cycle and any added
        # work inside the toggle block (logging, callback overhead) tipped
        # the race the wrong way on 3.10/3.11 — the loop would see
        # WAITING then jump straight to SUCCEEDED, never firing
        # on_child_running.
        await asyncio.sleep(0.30)
        async with manager._execution_lock:
            state.update_status(ExecutionStatus.RUNNING)
        await asyncio.sleep(0.30)
        async with manager._execution_lock:
            state.set_result({"ok": True})

    poller = asyncio.create_task(_drive_polling())
    result = await manager.wait_for_result(
        exec_id,
        timeout=5.0,
        pause_clock=parent_clock,
        on_child_waiting=on_child_waiting,
        on_child_running=on_child_running,
    )
    await poller

    assert result == {"ok": True}
    assert len(waiting_events) == 1, (
        f"on_child_waiting should fire exactly once when child enters WAITING; "
        f"fired {len(waiting_events)} times"
    )
    assert len(running_events) == 1, (
        f"on_child_running should fire exactly once when child exits WAITING; "
        f"fired {len(running_events)} times"
    )
    assert running_events[0] > waiting_events[0], (
        "on_child_running must fire AFTER on_child_waiting"
    )
    # The callbacks must fire in lockstep with the local pause-clock toggles.
    assert parent_clock.total_paused() >= 0.15


@pytest.mark.asyncio
async def test_wait_for_result_callbacks_optional_for_backward_compat():
    """A caller that doesn't pass the new callbacks must see no change in
    behavior. The pause_clock toggle and overall wait semantics stay identical
    to the existing tests."""
    from agentfield.async_execution_manager import AsyncExecutionManager
    from agentfield.async_config import AsyncConfig
    from agentfield.execution_state import ExecutionState, ExecutionStatus

    manager = AsyncExecutionManager(base_url="http://control", config=AsyncConfig())
    exec_id = "exec-multihop-nocallbacks"
    state = ExecutionState(
        execution_id=exec_id,
        target="agent.reasoner",
        input_data={},
        timeout=5.0,
    )
    state.update_status(ExecutionStatus.RUNNING)
    manager._executions[exec_id] = state

    parent_clock = PauseClock()

    async def _drive_polling():
        await asyncio.sleep(0.05)
        async with manager._execution_lock:
            state.update_status(ExecutionStatus.WAITING)
        await asyncio.sleep(0.20)
        async with manager._execution_lock:
            state.set_result({"ok": True})

    poller = asyncio.create_task(_drive_polling())
    result = await manager.wait_for_result(
        exec_id, timeout=5.0, pause_clock=parent_clock
    )
    await poller

    assert result == {"ok": True}
    assert parent_clock.total_paused() >= 0.15


@pytest.mark.asyncio
async def test_wait_for_result_callback_exceptions_dont_break_wait_loop():
    """If on_child_waiting / on_child_running raises (e.g. control plane is
    temporarily unreachable), the wait loop MUST continue — pause-state
    propagation is best-effort and an HTTP blip up the chain must not crash
    the awaiter or leave its pause_clock in a stuck state. Pinned because the
    callbacks fire HTTP calls in production and we never want a transient
    failure there to break the call graph."""
    from agentfield.async_execution_manager import AsyncExecutionManager
    from agentfield.async_config import AsyncConfig
    from agentfield.execution_state import ExecutionState, ExecutionStatus

    manager = AsyncExecutionManager(base_url="http://control", config=AsyncConfig())
    exec_id = "exec-multihop-flaky"
    state = ExecutionState(
        execution_id=exec_id,
        target="agent.reasoner",
        input_data={},
        timeout=5.0,
    )
    state.update_status(ExecutionStatus.RUNNING)
    manager._executions[exec_id] = state

    parent_clock = PauseClock()

    async def boom_waiting() -> None:
        raise RuntimeError("control plane unreachable")

    async def boom_running() -> None:
        raise RuntimeError("control plane unreachable")

    async def _drive_polling():
        await asyncio.sleep(0.05)
        async with manager._execution_lock:
            state.update_status(ExecutionStatus.WAITING)
        await asyncio.sleep(0.20)
        async with manager._execution_lock:
            state.update_status(ExecutionStatus.RUNNING)
        await asyncio.sleep(0.05)
        async with manager._execution_lock:
            state.set_result({"ok": True})

    poller = asyncio.create_task(_drive_polling())
    result = await manager.wait_for_result(
        exec_id,
        timeout=5.0,
        pause_clock=parent_clock,
        on_child_waiting=boom_waiting,
        on_child_running=boom_running,
    )
    await poller

    # Result must still come through; clock must still have paused.
    assert result == {"ok": True}
    assert parent_clock.total_paused() >= 0.15


@pytest.mark.asyncio
async def test_wait_for_result_without_pause_clock_unaffected():
    """No pause_clock kwarg = no toggling, behaves as a plain wallclock wait.
    Regression check that the new toggle code doesn't change pre-existing
    semantics for callers that don't pass a clock."""
    from agentfield.async_execution_manager import AsyncExecutionManager
    from agentfield.async_config import AsyncConfig
    from agentfield.execution_state import ExecutionState, ExecutionStatus
    from agentfield.exceptions import ExecutionTimeoutError

    manager = AsyncExecutionManager(base_url="http://control", config=AsyncConfig())
    exec_id = "exec-plain"
    state = ExecutionState(
        execution_id=exec_id,
        target="agent.reasoner",
        input_data={},
        timeout=0.2,
    )
    state.update_status(ExecutionStatus.WAITING)
    manager._executions[exec_id] = state

    # No pause_clock provided. Even though the child sits in WAITING, the
    # wait must time out at wallclock since there's no clock to subtract from.
    with pytest.raises(ExecutionTimeoutError):
        await manager.wait_for_result(exec_id, timeout=0.2)


@pytest.mark.asyncio
async def test_wait_for_result_ends_pause_on_terminal_while_waiting():
    """If the child transitions straight from WAITING to a terminal state
    (e.g. cancelled mid-approval), the wait's finally block must still close
    the in-flight pause so the parent's clock isn't stuck paused forever."""
    from agentfield.async_execution_manager import AsyncExecutionManager
    from agentfield.async_config import AsyncConfig
    from agentfield.execution_state import ExecutionState, ExecutionStatus
    from agentfield.exceptions import ExecutionCancelledError

    manager = AsyncExecutionManager(base_url="http://control", config=AsyncConfig())
    exec_id = "exec-cancelled-while-waiting"
    state = ExecutionState(
        execution_id=exec_id,
        target="agent.reasoner",
        input_data={},
        timeout=5.0,
    )
    state.update_status(ExecutionStatus.RUNNING)
    manager._executions[exec_id] = state

    parent_clock = PauseClock()

    async def _drive_polling():
        await asyncio.sleep(0.05)
        async with manager._execution_lock:
            state.update_status(ExecutionStatus.WAITING)
        await asyncio.sleep(0.10)
        async with manager._execution_lock:
            state.cancel("user_cancelled_during_approval")

    poller = asyncio.create_task(_drive_polling())
    with pytest.raises(ExecutionCancelledError):
        await manager.wait_for_result(
            exec_id, timeout=5.0, pause_clock=parent_clock
        )
    await poller

    # Pause must be closed: total_paused stops accumulating once the wait
    # returns. Sleep and verify the value is stable.
    paused_at_exit = parent_clock.total_paused()
    await asyncio.sleep(0.1)
    assert abs(parent_clock.total_paused() - paused_at_exit) < 1e-3, (
        "pause must be closed when wait_for_result exits via terminal child status"
    )


@pytest.mark.asyncio
async def test_wait_for_result_subtracts_pause_clock_from_elapsed():
    """Direct check on the active-elapsed math: a pause_clock that reports
    10s of paused time keeps a 0.5s wait alive long enough to settle."""
    from agentfield.async_execution_manager import AsyncExecutionManager
    from agentfield.async_config import AsyncConfig
    from agentfield.execution_state import ExecutionState
    from agentfield.exceptions import ExecutionTimeoutError

    manager = AsyncExecutionManager(base_url="http://control", config=AsyncConfig())

    exec_id = "exec-fake-clock"
    state = ExecutionState(
        execution_id=exec_id,
        target="agent.reasoner",
        input_data={},
        timeout=0.5,
    )
    manager._executions[exec_id] = state

    class _FakeClock:
        def total_paused(self) -> float:
            return 10.0

        def start_pause(self) -> None:
            pass

        def end_pause(self) -> None:
            pass

    async def _settle_after_short_delay():
        await asyncio.sleep(0.2)
        async with manager._execution_lock:
            state.set_result({"ok": True})

    asyncio.create_task(_settle_after_short_delay())
    result = await manager.wait_for_result(
        exec_id, timeout=0.5, pause_clock=_FakeClock()
    )
    assert result == {"ok": True}

    # Sanity: without any pause_clock, the same kind of setup with a tighter
    # timeout actually trips ExecutionTimeoutError.
    state2 = ExecutionState(
        execution_id="exec-strict",
        target="agent.reasoner",
        input_data={},
        timeout=0.1,
    )
    manager._executions["exec-strict"] = state2
    with pytest.raises(ExecutionTimeoutError):
        await manager.wait_for_result("exec-strict", timeout=0.1)


@pytest.mark.asyncio
async def test_wait_for_result_attaches_pause_clock_to_execution_state():
    """The polling task's ``is_overdue`` check must respect the same pause
    clock that ``wait_for_result`` is using to keep the loop alive. Without
    this attachment, ``_poll_active_executions`` flips the execution's
    status to TIMEOUT at exactly wallclock ``timeout`` — and the next iter
    of the wait loop surfaces it as ``ExecutionTimeoutError(\"Execution
    timed out after N seconds\")`` even though the child was paused for
    most of that time.

    Reproduces the production failure mode observed on Railway run
    run_1778346573033_dafddc40 where github-buddy's ``app.call`` to
    swe-af.build timed out at exactly 21600s wallclock despite v0.1.81's
    pause-aware wait loop, because the polling task's overdue check still
    used pure wallclock age."""
    from agentfield.async_execution_manager import AsyncExecutionManager
    from agentfield.async_config import AsyncConfig
    from agentfield.execution_state import ExecutionState, ExecutionStatus

    manager = AsyncExecutionManager(base_url="http://control", config=AsyncConfig())
    exec_id = "exec-attach"
    state = ExecutionState(
        execution_id=exec_id,
        target="agent.reasoner",
        input_data={},
        timeout=0.5,
    )
    state.update_status(ExecutionStatus.WAITING)
    manager._executions[exec_id] = state

    parent_clock = PauseClock()
    saw_attached = asyncio.Event()
    saw_detached = asyncio.Event()

    async def _observe_then_settle():
        # Wait until wait_for_result has started and attached the clock,
        # then verify and settle the child so the wait can return cleanly.
        for _ in range(50):
            if state._pause_clock is parent_clock:
                saw_attached.set()
                break
            await asyncio.sleep(0.01)
        assert saw_attached.is_set(), (
            "wait_for_result did not attach pause_clock to ExecutionState"
        )
        async with manager._execution_lock:
            state.set_result({"ok": True})

    asyncio.create_task(_observe_then_settle())
    result = await manager.wait_for_result(
        exec_id, timeout=0.5, pause_clock=parent_clock
    )
    assert result == {"ok": True}

    # After wait_for_result returns, the clock must be detached so future
    # polling passes don't keep subtracting paused time from a stale clock.
    assert state._pause_clock is None, (
        "wait_for_result must restore previous _pause_clock on exit"
    )
    saw_detached.set()  # marker for readability; kept to mirror saw_attached


@pytest.mark.asyncio
async def test_poll_active_executions_respects_attached_pause_clock():
    """End-to-end check on the bug: with an attached pause_clock that
    reports more paused time than the wallclock budget, the polling task
    must NOT mark the execution TIMEOUT. Without the fix, this test fails
    immediately because ``is_overdue`` would return True purely from
    wallclock age."""
    from agentfield.async_execution_manager import AsyncExecutionManager
    from agentfield.async_config import AsyncConfig
    from agentfield.execution_state import ExecutionState, ExecutionStatus

    manager = AsyncExecutionManager(base_url="http://control", config=AsyncConfig())
    exec_id = "exec-poll-pause"
    state = ExecutionState(
        execution_id=exec_id,
        target="agent.reasoner",
        input_data={},
        timeout=0.05,
    )
    state.update_status(ExecutionStatus.WAITING)
    state.next_poll_time = 0.0  # eligible for polling immediately

    class _SteadyClock:
        """Reports a constant 10s of paused time — much more than the
        0.05s timeout. With the fix, ``is_overdue`` reads age - 10s
        which is overwhelmingly negative, so the execution stays alive."""

        def total_paused(self) -> float:
            return 10.0

    state._pause_clock = _SteadyClock()
    manager._executions[exec_id] = state

    # Wait long enough that wallclock age exceeds timeout (0.05s).
    await asyncio.sleep(0.1)
    assert state.age > state.timeout, "test setup: wallclock age must exceed timeout"

    # Polling pass: would mark TIMEOUT pre-fix; with fix, leaves it alone.
    await manager._poll_active_executions()

    assert state.status == ExecutionStatus.WAITING, (
        f"polling task must not flip a paused execution to TIMEOUT, "
        f"got {state.status}"
    )


@pytest.mark.asyncio
async def test_poll_active_executions_still_times_out_without_pause_clock():
    """Backward-compat: without a pause_clock attached, the polling task's
    overdue check still fires at wallclock — same behaviour as before."""
    from agentfield.async_execution_manager import AsyncExecutionManager
    from agentfield.async_config import AsyncConfig
    from agentfield.execution_state import ExecutionState, ExecutionStatus

    manager = AsyncExecutionManager(base_url="http://control", config=AsyncConfig())
    exec_id = "exec-no-clock"
    state = ExecutionState(
        execution_id=exec_id,
        target="agent.reasoner",
        input_data={},
        timeout=0.05,
    )
    state.update_status(ExecutionStatus.RUNNING)
    state.next_poll_time = 0.0
    manager._executions[exec_id] = state

    await asyncio.sleep(0.1)
    await manager._poll_active_executions()

    assert state.status == ExecutionStatus.TIMEOUT, (
        "no pause_clock attached: wallclock overdue check must still fire"
    )


@pytest.mark.asyncio
async def test_reasoner_failed_reports_failed_status_and_preserves_result(monkeypatch):
    """A reasoner that raises ReasonerFailed must surface as status=failed
    while still carrying its structured result (so the control plane, which
    stores the result payload regardless of terminal status, keeps the rich
    outcome rather than just a bare error string)."""
    from agentfield.exceptions import ReasonerFailed

    agent = Agent(
        node_id="test-agent", agentfield_server="http://control", auto_register=False
    )

    @agent.reasoner()
    async def build() -> dict:
        await asyncio.sleep(0)
        raise ReasonerFailed(
            "Build failed: 0/3 issues completed, no branches merged",
            result={"success": False, "completed_issues": 0, "merged_branches": 0},
            error_details={"reason": "empty_build"},
        )

    recorded = []

    class DummyResponse:
        status_code = 200

        def json(self):
            return {}

    async def fake_request(self, method, url, **kwargs):
        recorded.append({"url": url, "json": kwargs.get("json")})
        return DummyResponse()

    monkeypatch.setattr(AgentFieldClient, "_async_request", fake_request)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=agent), base_url="http://agent"
    ) as client:
        response = await client.post(
            "/reasoners/build",
            json={},
            headers={"X-Execution-ID": "exec-fail-1"},
        )

    assert response.status_code == 202
    await asyncio.sleep(0.1)

    status_calls = [e for e in recorded if "/executions/" in e["url"]]
    assert status_calls, "expected async status callback"
    payload = status_calls[-1]["json"]
    assert payload["status"] == "failed"
    assert "0/3 issues" in payload["error"]
    assert payload["error_details"] == {"reason": "empty_build"}
    # Structured result preserved alongside the failed status.
    assert payload["result"]["success"] is False
    assert payload["result"]["completed_issues"] == 0


@pytest.mark.asyncio
async def test_plain_exception_failed_status_has_no_result(monkeypatch):
    """Regression guard: a generic exception still maps to status=failed with
    no result key — ReasonerFailed's result-preservation must not leak into
    the ordinary failure path."""
    agent = Agent(
        node_id="test-agent", agentfield_server="http://control", auto_register=False
    )

    @agent.reasoner()
    async def boom() -> dict:
        await asyncio.sleep(0)
        raise RuntimeError("kaboom")

    recorded = []

    class DummyResponse:
        status_code = 200

        def json(self):
            return {}

    async def fake_request(self, method, url, **kwargs):
        recorded.append({"url": url, "json": kwargs.get("json")})
        return DummyResponse()

    monkeypatch.setattr(AgentFieldClient, "_async_request", fake_request)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=agent), base_url="http://agent"
    ) as client:
        response = await client.post(
            "/reasoners/boom",
            json={},
            headers={"X-Execution-ID": "exec-fail-2"},
        )

    assert response.status_code == 202
    await asyncio.sleep(0.1)

    status_calls = [e for e in recorded if "/executions/" in e["url"]]
    payload = status_calls[-1]["json"]
    assert payload["status"] == "failed"
    assert payload["error"] == "kaboom"
    assert "result" not in payload
