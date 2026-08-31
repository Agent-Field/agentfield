import asyncio
import signal
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from agentfield import Agent, AgentMesh, MeshTargetNotFound
from agentfield.agent_registry import get_current_agent_instance, set_current_agent
from agentfield.agent_server import DEFAULT_SHUTDOWN_TIMEOUT
from agentfield.cost_tracker import (
    CostTracker,
    get_current_cost_tracker,
    reset_current_cost_tracker,
    set_current_cost_tracker,
)
from agentfield.exceptions import AgentFieldClientError, ExecutionFailedError
from agentfield.execution_context import (
    ExecutionContext,
    get_current_context,
    set_execution_context,
)
from agentfield.execution_state import ExecuteError


def make_agent(node_id):
    return Agent(node_id=node_id, auto_register=False)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_mesh_offline_two_agents_call_each_other(monkeypatch):
    monkeypatch.delenv("AGENTFIELD_SERVER", raising=False)
    monkeypatch.delenv("AGENTFIELD_SERVER_URL", raising=False)
    a, b = make_agent("agent-a"), make_agent("agent-b")

    @b.reasoner()
    async def greet(name: str) -> dict:
        return {"hello": name}

    a.client.execute = AsyncMock(side_effect=AssertionError("HTTP used"))
    a.client.execute_async = AsyncMock(side_effect=AssertionError("HTTP used"))
    AgentMesh([a, b])
    assert await a.call("agent-b.greet", name="x") == {"hello": "x"}
    a.client.execute.assert_not_called()
    a.client.execute_async.assert_not_called()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_mesh_caller_context_survives():
    a, b = make_agent("agent-a"), make_agent("agent-b")

    @b.reasoner()
    async def greet() -> dict:
        return {"ok": True}

    AgentMesh([a, b])
    context = ExecutionContext.create_new(
        agent_node_id=a.node_id, workflow_name="agent-a_workflow"
    )
    set_execution_context(context)
    set_current_agent(a)
    Agent._current_agent = a
    await a.call("agent-b.greet")
    assert get_current_agent_instance() is a
    assert get_current_context() is context
    assert get_current_context().execution_id == context.execution_id


@pytest.mark.unit
@pytest.mark.asyncio
async def test_mesh_nested_a_b_a_current_agent_at_each_hop():
    a, b = make_agent("agent-a"), make_agent("agent-b")
    seen = []

    def record():
        seen.append(
            (get_current_agent_instance().node_id, Agent.get_current().node_id)
        )

    @a.reasoner()
    async def pong() -> dict:
        record()
        return {"ok": True}

    @b.reasoner()
    async def echo() -> dict:
        record()
        return await b.call("agent-a.pong")

    @a.reasoner()
    async def greet() -> dict:
        record()
        return await a.call("agent-b.echo")

    AgentMesh([a, b])
    set_current_agent(a)
    Agent._current_agent = a
    assert await a.call("agent-a.greet") == {"ok": True}
    assert seen == [("agent-a", "agent-a"), ("agent-b", "agent-b"), ("agent-a", "agent-a")]
    assert get_current_agent_instance() is a


@pytest.mark.unit
@pytest.mark.asyncio
async def test_mesh_child_context_lineage():
    a, b = make_agent("agent-a"), make_agent("agent-b")

    @b.reasoner()
    async def lineage(execution_context=None) -> dict:
        return {
            "run_id": execution_context.run_id,
            "parent_execution_id": execution_context.parent_execution_id,
            "depth": execution_context.depth,
            "agent_instance": execution_context.agent_instance.node_id,
        }

    AgentMesh([a, b])
    parent = ExecutionContext.create_new(
        agent_node_id=a.node_id, workflow_name="agent-a_workflow"
    )
    set_execution_context(parent)
    set_current_agent(a)
    Agent._current_agent = a
    result = await a.call("agent-b.lineage")
    assert result == {
        "run_id": parent.run_id,
        "parent_execution_id": parent.execution_id,
        "depth": 0,
        "agent_instance": "agent-b",
    }


@pytest.mark.unit
@pytest.mark.asyncio
async def test_mesh_same_node_short_circuit():
    a = make_agent("agent-a")

    @a.reasoner()
    async def hello() -> dict:
        return {"hello": True}

    AgentMesh([a])
    assert not a.agentfield_connected
    assert await a.call("agent-a.hello") == {"hello": True}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_mesh_and_call_local_obey_outbound_call_limit():
    async def exercise(caller, invoke):
        active = 0
        max_active = 0

        @caller.reasoner(name="limited")
        async def limited() -> dict:
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0.01)
            active -= 1
            return {"ok": True}

        caller._max_concurrent_calls = 1
        await asyncio.gather(invoke(), invoke())
        assert max_active == 1

    mesh_caller = make_agent("mesh-caller")
    mesh_target = make_agent("mesh-target")

    # Register the exercising handler on the target while keeping the limiter
    # on the caller, which is the owner of outbound-call capacity.
    active = 0
    max_active = 0

    @mesh_target.reasoner(name="concurrent")
    async def concurrent() -> dict:
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.01)
        active -= 1
        return {"ok": True}

    mesh_caller._max_concurrent_calls = 1
    AgentMesh([mesh_caller, mesh_target])
    await asyncio.gather(
        mesh_caller.call("mesh-target.concurrent"),
        mesh_caller.call("mesh-target.concurrent"),
    )
    assert max_active == 1

    local = make_agent("local")
    await exercise(local, lambda: local.call_local("limited"))


@pytest.mark.unit
@pytest.mark.asyncio
async def test_mesh_unknown_node_raises_named_error():
    a = make_agent("agent-a")
    AgentMesh([a])
    with pytest.raises(MeshTargetNotFound) as exc:
        await a.call("agent-z.x")
    assert "agent-a" in str(exc.value)
    assert "server unavailable" not in str(exc.value)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_mesh_unknown_reasoner_on_known_node_raises_named_error():
    a, b = make_agent("agent-a"), make_agent("agent-b")
    AgentMesh([a, b])
    with pytest.raises(MeshTargetNotFound) as exc:
        await a.call("agent-b.nope")
    assert "agent-b.nope" in str(exc.value)
    assert not isinstance(exc.value, ExecuteError)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_mesh_validation_error_parity():
    a, b = make_agent("agent-a"), make_agent("agent-b")

    @b.reasoner()
    async def greet(name: str) -> dict:
        return {"name": name}

    direct = TestClient(b).post("/reasoners/greet", json={})
    AgentMesh([a, b])
    with pytest.raises(ExecuteError) as exc:
        await a.call("agent-b.greet")
    assert direct.status_code == exc.value.status_code == 422
    assert str(direct.json()["detail"]) in str(exc.value)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_mesh_result_has_no_usage_envelope():
    a, b = make_agent("agent-a"), make_agent("agent-b")

    @b.reasoner()
    async def use() -> dict:
        get_current_cost_tracker().record("model-x", 2, 3, 5)
        return {"ok": True}

    AgentMesh([a, b])
    tracker = CostTracker()
    token = set_current_cost_tracker(tracker)
    try:
        result = await a.call("agent-b.use")
    finally:
        reset_current_cost_tracker(token)
    assert "__agentfield_usage__" not in result
    assert tracker.serialize()["entries"][0]["model"] == "model-x"
    assert tracker.total_tokens == 5


@pytest.mark.unit
@pytest.mark.asyncio
async def test_mesh_emits_workflow_events():
    a, b = make_agent("agent-a"), make_agent("agent-b")
    events = []
    b.workflow_handler.fire_and_forget_update = lambda payload: events.append(payload)

    @b.reasoner()
    async def greet() -> dict:
        return {"ok": True}

    AgentMesh([a, b])
    await a.call("agent-b.greet")
    await b._notification_dispatcher._queue.join()
    statuses = [event["status"] for event in events]
    assert statuses.count("running") == 1
    assert statuses.count("succeeded") == 1
    common = {
        "agent_node_id", "reasoner_id", "execution_id", "run_id", "status"
    }
    assert common <= set(events[0]) and common <= set(events[1])


@pytest.mark.unit
@pytest.mark.asyncio
async def test_mesh_workflow_event_failure_does_not_break_the_call():
    a, b = make_agent("agent-a"), make_agent("agent-b")
    b.workflow_handler.fire_and_forget_update = lambda payload: (_ for _ in ()).throw(
        RuntimeError("offline")
    )

    @b.reasoner()
    async def greet() -> dict:
        return {"ok": True}

    AgentMesh([a, b])
    assert await a.call("agent-b.greet") == {"ok": True}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_mesh_does_not_forward_execution_id_header(monkeypatch):
    import agentfield.mesh as mesh_module

    a, b = make_agent("agent-a"), make_agent("agent-b")

    @b.reasoner()
    async def greet() -> dict:
        return {"ok": True}

    original = mesh_module._call_asgi
    captured = {}

    async def capture(app, path, body, headers):
        captured.update({key.lower(): value for key, value in headers.items()})
        return await original(app, path, body, headers)

    monkeypatch.setattr(mesh_module, "_call_asgi", capture)
    AgentMesh([a, b])
    result = await a.call("agent-b.greet")
    assert "x-execution-id" not in captured
    assert "x-run-id" in captured and "x-parent-execution-id" in captured
    assert result == {"ok": True} and result != {"status": "processing"}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_call_semantics_unchanged_without_mesh(monkeypatch):
    monkeypatch.delenv("AGENTFIELD_SERVER", raising=False)
    monkeypatch.delenv("AGENTFIELD_SERVER_URL", raising=False)
    a = make_agent("agent-a")
    target = "agent-b.greet"
    with pytest.raises(AgentFieldClientError) as exc:
        await a.call(target)
    assert str(exc.value) == (
        f"Cross-agent call to {target} failed: AgentField server unavailable. "
        "Agent is running in local mode."
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_mesh_positional_binding_matches_control_plane():
    a, b = make_agent("agent-a"), make_agent("agent-b")

    @b.reasoner()
    async def greet(name: str) -> dict:
        return {"name": name}

    AgentMesh([a, b])
    with pytest.raises(ExecuteError) as exc:
        await a.call("agent-b.greet", "x")
    assert exc.value.status_code == 422
    bare = make_agent("caller")
    bare.agentfield_connected = True
    bare.client.execute_async = AsyncMock(return_value="execution-id")
    bare.client.wait_for_execution_result = AsyncMock(return_value={"result": {}})
    assert await bare.call("agent-b.greet", "x") == {}
    assert bare.client.execute_async.await_args.kwargs["input_data"] == {"arg_0": "x"}

    b._mesh = None
    b.agentfield_connected = True
    b.client.execute_async = AsyncMock(return_value="execution-id")
    b.client.wait_for_execution_result = AsyncMock(return_value={"result": {}})
    assert await b.call("agent-b.greet", "x") == {}
    assert b.client.execute_async.await_args.kwargs["input_data"] == {"name": "x"}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_call_local_without_mesh_or_cp():
    agent = make_agent("local")
    events = []
    agent.workflow_handler.fire_and_forget_update = lambda payload: events.append(payload)

    @agent.reasoner()
    async def hello(name: str) -> dict:
        return {"hello": name}

    assert await agent.call_local("hello", "world") == {"hello": "world"}
    await agent._notification_dispatcher._queue.join()
    assert {event["status"] for event in events} == {"running", "succeeded"}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_call_local_rejects_foreign_node():
    agent = make_agent("local")
    with pytest.raises(MeshTargetNotFound):
        await agent.call_local("other-node.hello")


@pytest.mark.unit
def test_mesh_mounts_all_agents_on_one_app():
    a, b = make_agent("agent-a"), make_agent("agent-b")

    @a.reasoner()
    async def hello() -> dict:
        return {"a": True}

    @b.reasoner()
    async def greet() -> dict:
        return {"b": True}

    client = TestClient(AgentMesh([a, b]).app)
    assert client.post("/agent-a/reasoners/hello", json={}).status_code == 200
    assert client.post("/agent-b/reasoners/greet", json={}).status_code == 200
    assert client.get("/health").json() == {
        "status": "healthy", "nodes": ["agent-a", "agent-b"], "count": 2
    }


@pytest.mark.unit
def test_mesh_register_true_is_refused():
    with pytest.raises(NotImplementedError, match="callback URL path"):
        AgentMesh([make_agent("a")], register=True)


@pytest.mark.unit
def test_mesh_sets_auto_register_false_on_members():
    agent = make_agent("a")
    agent.auto_register = True
    AgentMesh([agent])
    assert agent.auto_register is False


@pytest.mark.unit
def test_mesh_rejects_duplicate_node_ids():
    with pytest.raises(ValueError):
        AgentMesh([make_agent("a"), make_agent("a")])


@pytest.mark.unit
def test_mesh_rejects_empty_member_list():
    with pytest.raises(ValueError):
        AgentMesh([])


@pytest.mark.unit
@pytest.mark.asyncio
async def test_mesh_shutdown_drains_every_member_under_one_budget(monkeypatch):
    members = [make_agent(name) for name in ("a", "b", "c")]
    calls = []

    async def drain_a(budget):
        calls.append(("a", budget))
        raise RuntimeError("one failed")

    monkeypatch.setattr(members[0].server_handler, "_drain_reasoner_tasks", drain_a)
    for member in members[1:]:
        async def drain(budget, name=member.node_id):
            calls.append((name, budget))
        monkeypatch.setattr(member.server_handler, "_drain_reasoner_tasks", drain)
    for member in members:
        monkeypatch.setattr(member, "_cleanup_async_resources", AsyncMock())
    await AgentMesh(members)._shutdown()
    assert sorted(name for name, _ in calls) == ["a", "b", "c"]
    assert {budget for _, budget in calls} == {DEFAULT_SHUTDOWN_TIMEOUT}


@pytest.mark.unit
def test_mesh_shutdown_timeout_comes_from_env(monkeypatch):
    members = [make_agent("a"), make_agent("b")]
    mesh = AgentMesh(members)
    monkeypatch.setenv("AGENTFIELD_SHUTDOWN_TIMEOUT", "2s")
    assert mesh._resolve_shutdown_timeout() == 2.0
    assert all(member.server_handler._shutdown_timeout == 2.0 for member in members)
    monkeypatch.setenv("AGENTFIELD_SHUTDOWN_TIMEOUT", "invalid")
    assert mesh._resolve_shutdown_timeout() == DEFAULT_SHUTDOWN_TIMEOUT


@pytest.mark.unit
def test_mesh_installs_one_signal_handler_pair():
    mesh = AgentMesh([make_agent(name) for name in ("a", "b", "c")])

    class Loop:
        def __init__(self, fail=False):
            self.calls = []
            self.fail = fail

        def add_signal_handler(self, *args):
            if self.fail:
                raise NotImplementedError
            self.calls.append(args)

    loop = Loop()
    mesh._install_signal_handlers(loop, object())
    assert [call[0] for call in loop.calls] == [signal.SIGTERM, signal.SIGINT]
    mesh._install_signal_handlers(Loop(fail=True), object())


@pytest.mark.unit
@pytest.mark.asyncio
async def test_mesh_reasoner_exception_maps_to_execution_failed_error():
    a, b = make_agent("a"), make_agent("b")

    @b.reasoner()
    async def explode() -> dict:
        raise RuntimeError("boom")

    AgentMesh([a, b])
    with pytest.raises(ExecutionFailedError) as exc:
        await a.call("b.explode")
    assert isinstance(exc.value.__cause__, RuntimeError)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_mesh_application_404_remains_execute_error():
    a, b = make_agent("a"), make_agent("b")

    @b.reasoner()
    async def missing_record() -> dict:
        raise HTTPException(status_code=404, detail="record absent")

    AgentMesh([a, b])
    with pytest.raises(ExecuteError) as exc:
        await a.call("b.missing_record")
    assert exc.value.status_code == 404
    assert "record absent" in str(exc.value)
    assert not isinstance(exc.value, MeshTargetNotFound)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_mesh_custom_reasoner_path_resolves():
    a, b = make_agent("a"), make_agent("b")

    @b.reasoner(path="custom")
    async def greet() -> dict:
        return {"custom": True}

    AgentMesh([a, b])
    assert await a.call("b.greet") == {"custom": True}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_mesh_raw_asgi_and_defensive_dispatch_branches(monkeypatch):
    import agentfield.mesh as mesh_module

    seen = {}

    async def app(scope, receive, send):
        seen["scope"] = scope
        seen["request"] = await receive()
        seen["disconnect"] = await receive()
        await send({"type": "http.response.start", "status": 201})
        await send({"type": "http.response.body", "body": b'{"ok":', "more_body": True})
        await send({"type": "http.response.body", "body": b"true}"})

    assert await mesh_module._call_asgi(app, "/x", b"{}", {}) == (
        201,
        b'{"ok":true}',
    )
    assert seen["disconnect"] == {"type": "http.disconnect"}
    assert (b"content-length", b"2") in seen["scope"]["headers"]

    agent = make_agent("a")
    with pytest.raises(MeshTargetNotFound):
        await mesh_module.dispatch_in_process(agent, "undotted", {}, {})

    @agent.skill()
    async def local_skill(value: str) -> dict:
        return {"value": value}

    context = ExecutionContext.create_new(
        agent_node_id="a", workflow_name="a_workflow"
    )
    headers = mesh_module.build_child_headers(context)
    assert await mesh_module.dispatch_in_process(
        agent, "a.local_skill", {"value": "x"}, headers
    ) == {"value": "x"}

    async def response_404(*args, **kwargs):
        return 404, b'{"detail":"gone"}'

    monkeypatch.setattr(mesh_module, "_call_asgi", response_404)
    with pytest.raises(ExecuteError) as exc:
        await mesh_module.dispatch_in_process(
            agent, "a.local_skill", {"value": "x"}, {"x-execution-id": "drop"}
        )
    assert exc.value.status_code == 404


@pytest.mark.unit
@pytest.mark.asyncio
async def test_mesh_serve_lifespan_and_cleanup_are_single_owner(monkeypatch):
    import agentfield.mesh as mesh_module

    members = [make_agent("a"), make_agent("b")]
    mesh = AgentMesh(members)
    removed = []

    class Loop:
        def add_signal_handler(self, *args):
            pass

        def remove_signal_handler(self, sig):
            removed.append(sig)

    class Server:
        def __init__(self, config):
            self.config = config

        def run(self):
            pass

    monkeypatch.setattr(mesh_module.uvicorn, "Server", Server)
    monkeypatch.setattr(mesh_module.asyncio, "get_running_loop", lambda: Loop())
    for member in members:
        monkeypatch.setattr(
            member.server_handler, "_drain_reasoner_tasks", AsyncMock()
        )
    monkeypatch.setattr(members[0], "_cleanup_async_resources", AsyncMock())
    monkeypatch.setattr(
        members[1],
        "_cleanup_async_resources",
        AsyncMock(side_effect=RuntimeError("cleanup")),
    )
    mesh.run(host="127.0.0.1", port=9123)
    assert mesh.resolve("a") is members[0] and mesh.resolve("missing") is None
    async with mesh.app.router.lifespan_context(mesh.app):
        pass
    assert removed == [signal.SIGTERM, signal.SIGINT]


@pytest.mark.unit
def test_mesh_signal_callback_marks_all_members():
    members = [make_agent("a"), make_agent("b")]
    mesh = AgentMesh(members)
    handlers = {}

    class Loop:
        def add_signal_handler(self, sig, callback, value):
            handlers[sig] = (callback, value)

    class Server:
        def __init__(self):
            self.exits = []

        def handle_exit(self, sig, frame):
            self.exits.append((sig, frame))

    server = Server()
    mesh._install_signal_handlers(Loop(), server)
    callback, value = handlers[signal.SIGTERM]
    callback(value)
    assert all(member._shutdown_requested for member in members)
    assert server.exits == [(signal.SIGTERM, None)]

    class RuntimeErrorLoop:
        def add_signal_handler(self, *args):
            raise RuntimeError

    mesh._install_signal_handlers(RuntimeErrorLoop(), server)


@pytest.mark.unit
def test_mesh_serve_applies_budget_and_handles_keyboard_interrupt(monkeypatch):
    import agentfield.mesh as mesh_module

    member = make_agent("a")
    mesh = AgentMesh([member])
    captured = {}

    class Server:
        def __init__(self, config):
            captured["config"] = config

        def run(self):
            raise KeyboardInterrupt

    monkeypatch.setenv("AGENTFIELD_SHUTDOWN_TIMEOUT", "0.25s")
    monkeypatch.setattr(mesh_module.uvicorn, "Server", Server)
    mesh.serve(host="127.0.0.1", port=9125)

    assert captured["config"].timeout_graceful_shutdown == 0.25
    assert member.server_handler._shutdown_timeout == 0.25


@pytest.mark.unit
@pytest.mark.asyncio
async def test_call_local_forwards_parent_vc_id(monkeypatch):
    """build_child_headers propagates parent_vc_id, like Agent.call's inline build."""
    import agentfield.mesh as mesh_module

    agent = make_agent("local")

    @agent.reasoner()
    async def hello() -> dict:
        return {"ok": True}

    original = mesh_module._call_asgi
    captured = {}

    async def capture(app, path, body, headers):
        captured.update({key.lower(): value for key, value in headers.items()})
        return await original(app, path, body, headers)

    monkeypatch.setattr(mesh_module, "_call_asgi", capture)
    context = ExecutionContext.create_new(
        agent_node_id=agent.node_id, workflow_name="local_workflow"
    )
    context.parent_vc_id = "vc-123"
    set_execution_context(context)

    assert await agent.call_local("hello") == {"ok": True}
    assert captured["x-parent-vc-id"] == "vc-123"
    assert captured["x-parent-execution-id"] == context.execution_id


@pytest.mark.unit
@pytest.mark.asyncio
async def test_mesh_dispatch_leaves_no_ambient_agent_when_none_was_set(monkeypatch):
    """A dispatch must not leave an ambient current agent behind.

    `Agent._current_agent` is a CLASS attribute, so a child context does not
    protect it. When none was set before the call, dispatch_in_process must
    delete whatever the target left behind rather than restoring `None` — a
    restored `None` would make `Agent.get_current()` look "set to nothing".
    """
    import agentfield.mesh as mesh_module

    agent = make_agent("local")

    @agent.reasoner()
    async def hello() -> dict:
        return {"ok": True}

    async def leaky(app, path, body, headers):
        Agent._current_agent = agent
        return 200, b'{"ok": true}'

    monkeypatch.setattr(mesh_module, "_call_asgi", leaky)
    if hasattr(Agent, "_current_agent"):
        delattr(Agent, "_current_agent")

    context = ExecutionContext.create_new(
        agent_node_id=agent.node_id, workflow_name="local_workflow"
    )
    headers = mesh_module.build_child_headers(context)
    assert await mesh_module.dispatch_in_process(
        agent, "local.hello", {}, headers
    ) == {"ok": True}
    assert not hasattr(Agent, "_current_agent")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_mesh_lifespan_tolerates_loop_without_signal_removal(monkeypatch):
    """A loop that cannot remove signal handlers must not break shutdown."""
    import agentfield.mesh as mesh_module

    members = [make_agent("a")]
    mesh = AgentMesh(members)

    class Loop:
        def add_signal_handler(self, *args):
            pass

        def remove_signal_handler(self, sig):
            raise NotImplementedError

    class Server:
        def __init__(self, config):
            self.config = config

        def run(self):
            pass

    monkeypatch.setattr(mesh_module.uvicorn, "Server", Server)
    monkeypatch.setattr(mesh_module.asyncio, "get_running_loop", lambda: Loop())
    monkeypatch.setattr(members[0].server_handler, "_drain_reasoner_tasks", AsyncMock())
    monkeypatch.setattr(members[0], "_cleanup_async_resources", AsyncMock())
    mesh.serve(host="127.0.0.1", port=9124)
    async with mesh.app.router.lifespan_context(mesh.app):
        pass
    members[0].server_handler._drain_reasoner_tasks.assert_awaited_once()
