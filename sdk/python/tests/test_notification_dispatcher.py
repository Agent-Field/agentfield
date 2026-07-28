"""Behavior tests for the non-blocking notification path (issue #622).

Each test is derived from the validation contract for the fix, not from the
implementation:

1. Telemetry POSTs must not block the reasoner critical path.
2. Events for a given execution must reach the control plane in order
   (``running`` before the terminal status), including under concurrent
   executions with failures.
3. Shutdown must deliver already-queued events before the HTTP client closes.
4. Submitting on a dispatcher that was never explicitly started must still
   deliver (CLI ``call`` mode and direct ASGI mounts never run the server
   lifespan that starts the dispatcher).
"""

import asyncio
import time
from types import SimpleNamespace

from agentfield.agent import Agent
from agentfield.agent import _NotificationDispatcher
from agentfield.agent_workflow import AgentWorkflow

EVENT_POST_DELAY = 0.5


class RecordingClient:
    """Fake control-plane client.

    Registration POSTs return immediately; event POSTs sleep for
    ``event_post_delay`` and record ``(execution_id, status, closed)`` in
    arrival order, where ``closed`` captures whether ``aclose()`` had already
    run when the event was delivered.
    """

    _current_workflow_context = None

    def __init__(self, event_post_delay: float = 0.0):
        self.event_post_delay = event_post_delay
        self.events = []
        self.closed = False

    async def _async_request(self, method, url, json=None):
        if url.endswith("/executions/events"):
            await asyncio.sleep(self.event_post_delay)
            self.events.append((json.get("execution_id"), json.get("status"), self.closed))
        return SimpleNamespace(json=lambda: {})

    async def aclose(self):
        self.closed = True

    def statuses(self):
        return [status for _, status, _ in self.events]


def make_workflow_agent(client):
    """Agent stub (init bypassed) with a real dispatcher, plus its workflow."""
    agent = object.__new__(Agent)
    agent.node_id = "node"
    agent.agentfield_server = "http://agentfield"
    agent.dev_mode = False
    agent.client = client
    agent._current_execution_context = None
    agent._async_execution_manager = None
    agent._background_tasks = set()
    agent._notification_dispatcher = _NotificationDispatcher(dev_mode=False)
    return agent, AgentWorkflow(agent)


async def quick_reasoner():
    await asyncio.sleep(0.01)
    return "ok"


async def failing_reasoner():
    await asyncio.sleep(0.01)
    raise ValueError("boom")


async def test_reasoner_path_does_not_wait_for_event_posts():
    """Contract 1: the reasoner returns without awaiting telemetry POSTs."""
    client = RecordingClient(event_post_delay=EVENT_POST_DELAY)
    agent, workflow = make_workflow_agent(client)
    agent._notification_dispatcher.start()

    started = time.perf_counter()
    result = await workflow.execute_with_tracking(quick_reasoner, (), {})
    elapsed = time.perf_counter() - started

    assert result == "ok"
    # The pre-fix behavior awaited the start and complete POSTs inline, so it
    # could never return in less than 2 * EVENT_POST_DELAY. Allow generous
    # slack for slow CI: anything under a single POST delay proves the
    # notifications were detached from the critical path.
    assert elapsed < EVENT_POST_DELAY, (
        f"reasoner path blocked for {elapsed:.2f}s; telemetry POSTs are "
        "back on the critical path"
    )

    # The events must still be delivered afterwards, in order.
    await agent._notification_dispatcher.shutdown(timeout=30)
    assert client.statuses() == ["running", "succeeded"]


async def test_per_execution_event_order_preserved_under_concurrency():
    """Contract 2: running precedes the terminal event for every execution."""
    client = RecordingClient(event_post_delay=0.02)
    agent, workflow = make_workflow_agent(client)
    agent._notification_dispatcher.start()

    results = await asyncio.gather(
        workflow.execute_with_tracking(quick_reasoner, (), {}),
        workflow.execute_with_tracking(failing_reasoner, (), {}),
        workflow.execute_with_tracking(quick_reasoner, (), {}),
        return_exceptions=True,
    )
    assert results[0] == "ok"
    assert isinstance(results[1], ValueError)
    assert results[2] == "ok"

    await agent._notification_dispatcher.shutdown(timeout=30)

    per_execution = {}
    for execution_id, status, _ in client.events:
        per_execution.setdefault(execution_id, []).append(status)

    assert len(per_execution) == 3
    assert sorted(seq[-1] for seq in per_execution.values()) == [
        "failed",
        "succeeded",
        "succeeded",
    ]
    for execution_id, seq in per_execution.items():
        assert seq[0] == "running", (
            f"{execution_id}: expected 'running' first, saw {seq}"
        )
        assert len(seq) == 2, f"{execution_id}: unexpected events {seq}"


async def test_cleanup_delivers_queued_events_before_client_closes():
    """Contract 3: shutdown drains pending events, then closes the client."""
    client = RecordingClient(event_post_delay=0.05)
    agent, workflow = make_workflow_agent(client)
    agent._notification_dispatcher.start()

    for _ in range(3):
        await workflow.execute_with_tracking(quick_reasoner, (), {})

    # Several events are still queued or in flight here; the real shutdown
    # entrypoint must deliver all of them before the HTTP client closes.
    await agent._cleanup_async_resources()

    assert client.closed is True
    assert len(client.events) == 6
    assert all(
        closed_at_delivery is False for _, _, closed_at_delivery in client.events
    ), "an event was delivered after the HTTP client was closed"


async def test_never_started_dispatcher_still_delivers():
    """Contract 4: paths that skip the server lifespan still get telemetry.

    The CLI ``call`` path runs a tracked function via ``asyncio.run`` without
    ever starting the dispatcher; submissions must lazily start it instead of
    silently dropping events.
    """
    client = RecordingClient()
    agent, workflow = make_workflow_agent(client)
    # Deliberately no dispatcher.start() here.

    result = await workflow.execute_with_tracking(quick_reasoner, (), {})
    assert result == "ok"

    await agent._notification_dispatcher.shutdown(timeout=30)
    assert client.statuses() == ["running", "succeeded"]
