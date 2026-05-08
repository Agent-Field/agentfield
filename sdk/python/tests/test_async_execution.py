import asyncio
import httpx
import pytest

from agentfield.agent import Agent
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
