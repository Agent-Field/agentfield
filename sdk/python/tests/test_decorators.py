import asyncio

import pytest

from haxen_sdk.decorators import reasoner, _execute_with_tracking
from haxen_sdk.execution_context import ExecutionContext
from haxen_sdk.agent_registry import set_current_agent, clear_current_agent
from tests.helpers import StubAgent


def test_reasoner_metadata_and_plain_call():
    @reasoner(tags=["t1"], description="desc")
    def add(a: int, b: int):
        return a + b

    # metadata
    assert getattr(add, "_is_reasoner", False) is True
    assert add._reasoner_tags == ["t1"]
    assert add._reasoner_description == "desc"
    # executes without agent context (falls back to plain call)
    assert asyncio.run(add(2, 3)) == 5


def test_reasoner_no_parentheses_syntax():
    @reasoner
    def echo(x):
        return x

    assert getattr(echo, "_is_reasoner", False) is True
    assert asyncio.run(echo("hi")) == "hi"


def test_reasoner_disable_tracking():
    @reasoner(track_workflow=False)
    def mul(a, b):
        return a * b

    assert asyncio.run(mul(3, 4)) == 12


@pytest.mark.asyncio
async def test_execute_with_tracking_success(monkeypatch):
    captured = {}

    async def record_start(agent, ctx, payload):
        captured.setdefault("start", []).append((ctx, payload))

    async def record_complete(agent, ctx, result, duration_ms, payload):
        captured.setdefault("complete", []).append((ctx, result))

    monkeypatch.setattr("haxen_sdk.decorators._send_workflow_start", record_start)
    monkeypatch.setattr(
        "haxen_sdk.decorators._send_workflow_completion", record_complete
    )

    agent = StubAgent()
    set_current_agent(agent)

    tasks = []

    def capture_task(coro):
        task = asyncio.ensure_future(coro)
        tasks.append(task)
        return task

    monkeypatch.setattr(asyncio, "create_task", capture_task)

    async def sample(value: int, execution_context: ExecutionContext = None) -> int:
        assert isinstance(execution_context, ExecutionContext)
        return value * 2

    try:
        result = await _execute_with_tracking(sample, 5)
    finally:
        clear_current_agent()
        if tasks:
            await asyncio.gather(*tasks)

    assert result == 10
    assert "start" in captured
    ctx, payload = captured["start"][0]
    assert ctx.reasoner_name == "sample"
    assert payload["args"][0] == 5
    assert "complete" in captured


@pytest.mark.asyncio
async def test_execute_with_tracking_error(monkeypatch):
    calls = {}

    async def record_error(agent, ctx, message, duration_ms, payload):
        calls.setdefault("error", []).append((ctx, message))

    monkeypatch.setattr(
        "haxen_sdk.decorators._send_workflow_start", lambda *a, **k: asyncio.sleep(0)
    )
    monkeypatch.setattr("haxen_sdk.decorators._send_workflow_error", record_error)

    agent = StubAgent()
    set_current_agent(agent)

    tasks = []

    def capture_task(coro):
        task = asyncio.ensure_future(coro)
        tasks.append(task)
        return task

    monkeypatch.setattr(asyncio, "create_task", capture_task)

    async def boom():
        raise ValueError("fail")

    with pytest.raises(ValueError):
        try:
            await _execute_with_tracking(boom)
        finally:
            clear_current_agent()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

    assert "error" in calls
