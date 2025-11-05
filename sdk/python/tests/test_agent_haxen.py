import threading

import pytest
import requests

from haxen_sdk.agent_haxen import AgentHaxen
from tests.helpers import StubAgent, DummyHaxenClient


@pytest.mark.asyncio
async def test_register_with_haxen_server_sets_base_url(monkeypatch):
    agent = StubAgent(callback_url="agent.local", base_url=None)
    agent.client = DummyHaxenClient()
    agent.haxen_connected = False

    monkeypatch.setattr(
        "haxen_sdk.agent._resolve_callback_url",
        lambda url, port: f"http://resolved:{port}",
    )
    monkeypatch.setattr(
        "haxen_sdk.agent._build_callback_candidates",
        lambda value, port, include_defaults=True: [f"http://resolved:{port}"],
    )
    monkeypatch.setattr("haxen_sdk.agent._is_running_in_container", lambda: False)

    haxen = AgentHaxen(agent)
    await haxen.register_with_haxen_server(port=8080)

    assert agent.base_url == "http://resolved:8080"
    assert agent.haxen_connected is True
    assert agent.client.register_calls[0]["base_url"] == "http://resolved:8080"


@pytest.mark.asyncio
async def test_register_with_haxen_server_handles_failure(monkeypatch):
    async def failing_register(*args, **kwargs):
        raise RuntimeError("boom")

    agent = StubAgent(callback_url=None, base_url="http://already", dev_mode=True)
    agent.client = DummyHaxenClient()
    monkeypatch.setattr(agent.client, "register_agent", failing_register)
    monkeypatch.setattr(
        "haxen_sdk.agent._build_callback_candidates",
        lambda value, port, include_defaults=True: [],
    )
    monkeypatch.setattr("haxen_sdk.agent._is_running_in_container", lambda: False)

    haxen = AgentHaxen(agent)
    agent.haxen_connected = True

    await haxen.register_with_haxen_server(port=9000)
    assert agent.haxen_connected is False


@pytest.mark.asyncio
async def test_register_with_haxen_updates_existing_port(monkeypatch):
    agent = StubAgent(callback_url=None, base_url="http://host:5000")
    agent.client = DummyHaxenClient()

    monkeypatch.setattr(
        "haxen_sdk.agent._build_callback_candidates",
        lambda value, port, include_defaults=True: [],
    )
    monkeypatch.setattr("haxen_sdk.agent._is_running_in_container", lambda: False)

    haxen = AgentHaxen(agent)
    await haxen.register_with_haxen_server(port=6000)

    assert agent.base_url == "http://host:6000"
    assert agent.client.register_calls[0]["base_url"] == "http://host:6000"


@pytest.mark.asyncio
async def test_register_with_haxen_preserves_container_urls(monkeypatch):
    agent = StubAgent(
        callback_url=None,
        base_url="http://service.railway.internal:5000",
        dev_mode=True,
    )
    agent.client = DummyHaxenClient()

    monkeypatch.setattr(
        "haxen_sdk.agent._build_callback_candidates",
        lambda value, port, include_defaults=True: [],
    )
    monkeypatch.setattr("haxen_sdk.agent._is_running_in_container", lambda: True)

    haxen = AgentHaxen(agent)
    await haxen.register_with_haxen_server(port=7000)

    assert agent.base_url == "http://service.railway.internal:5000"


@pytest.mark.asyncio
async def test_register_with_haxen_server_resolves_when_no_candidates(monkeypatch):
    agent = StubAgent(callback_url=None, base_url=None)
    agent.client = DummyHaxenClient()

    monkeypatch.setattr(
        "haxen_sdk.agent._build_callback_candidates", lambda *a, **k: []
    )
    monkeypatch.setattr(
        "haxen_sdk.agent._resolve_callback_url",
        lambda url, port: f"http://resolved:{port}",
    )
    monkeypatch.setattr("haxen_sdk.agent._is_running_in_container", lambda: False)

    haxen = AgentHaxen(agent)
    await haxen.register_with_haxen_server(port=7100)

    assert agent.base_url == "http://resolved:7100"
    assert agent.haxen_connected is True


@pytest.mark.asyncio
async def test_register_with_haxen_server_reorders_candidates(monkeypatch):
    agent = StubAgent(callback_url=None, base_url="http://preferred:8000")
    agent.client = DummyHaxenClient()
    agent.callback_candidates = ["http://other:8000", "http://preferred:8000"]

    monkeypatch.setattr(
        "haxen_sdk.agent._build_callback_candidates",
        lambda value, port, include_defaults=True: agent.callback_candidates,
    )
    monkeypatch.setattr("haxen_sdk.agent._is_running_in_container", lambda: False)

    haxen = AgentHaxen(agent)
    await haxen.register_with_haxen_server(port=8000)

    assert agent.callback_candidates[0] == "http://preferred:8000"


@pytest.mark.asyncio
async def test_register_with_haxen_server_propagates_request_exception(monkeypatch):
    class DummyResponse:
        def __init__(self):
            self.status_code = 503
            self.text = "unavailable"

    exception = requests.exceptions.RequestException("fail")
    exception.response = DummyResponse()

    async def failing_register(*args, **kwargs):
        raise exception

    agent = StubAgent(callback_url=None, base_url="http://already", dev_mode=False)
    agent.client = DummyHaxenClient()
    monkeypatch.setattr(agent.client, "register_agent", failing_register)
    monkeypatch.setattr(
        "haxen_sdk.agent._build_callback_candidates", lambda *a, **k: []
    )
    monkeypatch.setattr(
        "haxen_sdk.agent._resolve_callback_url", lambda url, port: "http://already"
    )
    monkeypatch.setattr("haxen_sdk.agent._is_running_in_container", lambda: False)

    haxen = AgentHaxen(agent)
    with pytest.raises(requests.exceptions.RequestException):
        await haxen.register_with_haxen_server(port=9001)
    assert agent.haxen_connected is False


@pytest.mark.asyncio
async def test_register_with_haxen_server_unsuccessful_response(monkeypatch):
    agent = StubAgent(callback_url=None, base_url="http://host:5000")
    agent.client = DummyHaxenClient()

    async def register_returns_false(*args, **kwargs):
        return False, None

    monkeypatch.setattr(agent.client, "register_agent", register_returns_false)
    monkeypatch.setattr(
        "haxen_sdk.agent._build_callback_candidates", lambda *a, **k: []
    )
    monkeypatch.setattr(
        "haxen_sdk.agent._resolve_callback_url", lambda url, port: "http://host:5000"
    )
    monkeypatch.setattr("haxen_sdk.agent._is_running_in_container", lambda: False)

    haxen = AgentHaxen(agent)
    await haxen.register_with_haxen_server(port=5000)
    assert agent.haxen_connected is False


@pytest.mark.asyncio
async def test_register_with_haxen_applies_discovery_payload(monkeypatch):
    from tests.helpers import create_test_agent

    agent, haxen_client = create_test_agent(monkeypatch)
    agent.callback_candidates = []

    async def fake_register(node_id, reasoners, skills, base_url, discovery=None):
        return True, {
            "resolved_base_url": "https://public:9000",
            "callback_discovery": {
                "candidates": ["https://public:9000", "http://fallback:9000"],
            },
        }

    monkeypatch.setattr(haxen_client, "register_agent", fake_register)
    monkeypatch.setattr(
        "haxen_sdk.agent._build_callback_candidates",
        lambda value, port, include_defaults=True: [f"http://detected:{port}"],
    )
    monkeypatch.setattr("haxen_sdk.agent._is_running_in_container", lambda: False)

    await agent.haxen_handler.register_with_haxen_server(port=9000)

    assert agent.base_url == "https://public:9000"
    assert agent.callback_candidates[0] == "https://public:9000"
    assert "http://fallback:9000" in agent.callback_candidates


def test_send_heartbeat(monkeypatch):
    agent = StubAgent()
    haxen = AgentHaxen(agent)

    calls = {}

    def fake_post(url, headers=None, timeout=None):
        calls["url"] = url

        class Dummy:
            status_code = 200
            text = "ok"

        return Dummy()

    monkeypatch.setattr("requests.post", fake_post)
    haxen.send_heartbeat()
    assert calls["url"].endswith(f"/api/v1/nodes/{agent.node_id}/heartbeat")


def test_send_heartbeat_warns_on_non_200(monkeypatch):
    agent = StubAgent()
    agent.haxen_connected = True
    haxen = AgentHaxen(agent)

    class Dummy:
        status_code = 500
        text = "error"

    monkeypatch.setattr("requests.post", lambda *a, **k: Dummy())
    haxen.send_heartbeat()


@pytest.mark.asyncio
async def test_enhanced_heartbeat_returns_false_when_disconnected():
    agent = StubAgent()
    haxen = AgentHaxen(agent)
    agent.haxen_connected = False
    assert await haxen.send_enhanced_heartbeat() is False


def test_start_and_stop_heartbeat(monkeypatch):
    agent = StubAgent()
    haxen = AgentHaxen(agent)

    called = []

    def fake_worker(interval):
        called.append(interval)

    monkeypatch.setattr(haxen, "heartbeat_worker", fake_worker)

    haxen.start_heartbeat(interval=1)
    assert isinstance(agent._heartbeat_thread, threading.Thread)
    haxen.stop_heartbeat()


@pytest.mark.asyncio
async def test_enhanced_heartbeat_and_shutdown(monkeypatch):
    agent = StubAgent()
    agent.client = DummyHaxenClient()
    agent.mcp_handler = type(
        "MCP", (), {"_get_mcp_server_health": lambda self: ["mcp"]}
    )()
    agent.dev_mode = True
    haxen = AgentHaxen(agent)

    success = await haxen.send_enhanced_heartbeat()
    assert success is True
    assert agent.client.heartbeat_calls

    success_shutdown = await haxen.notify_shutdown()
    assert success_shutdown is True
    assert agent.client.shutdown_calls == [agent.node_id]


@pytest.mark.asyncio
async def test_enhanced_heartbeat_failure_returns_false(monkeypatch):
    agent = StubAgent()
    agent.client = DummyHaxenClient()
    haxen = AgentHaxen(agent)

    async def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(agent.client, "send_enhanced_heartbeat", boom)
    agent.haxen_connected = True
    agent.dev_mode = True
    assert await haxen.send_enhanced_heartbeat() is False


@pytest.mark.asyncio
async def test_notify_shutdown_failure_returns_false(monkeypatch):
    agent = StubAgent()
    agent.client = DummyHaxenClient()
    haxen = AgentHaxen(agent)

    async def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(agent.client, "notify_graceful_shutdown", boom)
    agent.haxen_connected = True
    agent.dev_mode = True
    assert await haxen.notify_shutdown() is False


def test_send_heartbeat_handles_error(monkeypatch):
    agent = StubAgent()
    agent.haxen_connected = True
    haxen = AgentHaxen(agent)

    def boom(*args, **kwargs):
        raise requests.RequestException("boom")

    monkeypatch.setattr("requests.post", boom)
    haxen.send_heartbeat()


def test_start_heartbeat_skips_when_disconnected():
    agent = StubAgent()
    agent.haxen_connected = False
    haxen = AgentHaxen(agent)
    haxen.start_heartbeat()
    assert agent._heartbeat_thread is None
