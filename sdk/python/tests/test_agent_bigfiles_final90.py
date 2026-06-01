from __future__ import annotations

import io
import sys
import types
from types import MethodType
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agentfield.agent import (
    Agent,
    _PauseManager,
    _build_callback_candidates,
    _detect_container_ip,
    _detect_local_ip,
    _is_running_in_container,
    _normalize_candidate,
    _resolve_callback_url,
)
from agentfield.execution_context import ExecutionContext


class _SocketStub:
    def __init__(self, addr):
        self.addr = addr

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def connect(self, _target):
        return None

    def getsockname(self):
        return (self.addr, 12345)


class _Response:
    def __init__(self, status_code=404, text=""):
        self.status_code = status_code
        self.text = text


@pytest.fixture
def bare_agent():
    agent = object.__new__(Agent)
    agent.node_id = "node-1"
    agent.base_url = "http://agent.test"
    agent.callback_candidates = []
    agent.dev_mode = True
    agent.memory_event_client = None
    agent.agentfield_server = "http://cp.test"
    agent.api_key = "secret"
    agent._current_execution_context = None
    agent._reasoner_return_types = {"summarize": dict}
    agent.reasoners = [{"id": "legacy", "return_type": list}]
    return agent


def test_detect_container_ip_prefers_first_success(monkeypatch):
    requests_module = types.ModuleType("requests")
    calls = []

    def fake_get(url, **kwargs):
        calls.append(url)
        if "latest/meta-data" in url:
            return _Response(200, "203.0.113.10\n")
        raise AssertionError("unexpected fallback request")

    requests_module.get = fake_get
    monkeypatch.setitem(sys.modules, "requests", requests_module)

    assert _detect_container_ip() == "203.0.113.10"
    assert calls == ["http://169.254.169.254/latest/meta-data/public-ipv4"]


def test_detect_container_ip_falls_back_to_ipify(monkeypatch):
    requests_module = types.ModuleType("requests")

    def fake_get(url, **kwargs):
        if "ipify" in url:
            return _Response(200, "198.51.100.7")
        raise RuntimeError("metadata unavailable")

    requests_module.get = fake_get
    monkeypatch.setitem(sys.modules, "requests", requests_module)

    assert _detect_container_ip() == "198.51.100.7"


def test_detect_local_ip_and_container_helpers(monkeypatch):
    monkeypatch.setattr(
        "agentfield.agent.socket.socket", lambda *args, **kwargs: _SocketStub("10.0.0.5")
    )
    assert _detect_local_ip() == "10.0.0.5"

    monkeypatch.setattr("agentfield.agent.os.path.exists", lambda path: False)
    monkeypatch.setattr(
        "builtins.open", lambda *args, **kwargs: io.StringIO("12:memory:/kubepods"), raising=True
    )
    assert _is_running_in_container() is True


def test_callback_candidate_helpers(monkeypatch):
    monkeypatch.setattr("agentfield.agent._is_running_in_container", lambda: True)
    monkeypatch.setattr("agentfield.agent._detect_container_ip", lambda: "198.51.100.20")
    monkeypatch.setattr("agentfield.agent._detect_local_ip", lambda: "10.0.0.8")
    monkeypatch.setattr("agentfield.agent.socket.gethostname", lambda: "hostbox")
    monkeypatch.setenv("AGENT_CALLBACK_URL", "callback.internal")
    monkeypatch.setenv("RAILWAY_SERVICE_NAME", "svc")
    monkeypatch.setenv("RAILWAY_ENVIRONMENT", "prod")

    assert _normalize_candidate("http://[2001:db8::1]", 9000) == "http://[2001:db8::1]:9000"
    assert _normalize_candidate("https://example.com:8443", 9000) == "https://example.com:8443"

    candidates = _build_callback_candidates("api.example.com", 8001)
    assert candidates[0] == "http://api.example.com:8001"
    expected_urls = [
        "http://callback.internal:8001",
        "http://svc.railway.internal:8001",
        "http://198.51.100.20:8001",
        "http://10.0.0.8:8001",
        "http://hostbox:8001",
    ]
    for url in expected_urls:
        assert any(c == url for c in candidates), f"{url} not found in candidates"
    assert _resolve_callback_url(None, 7777) == "http://callback.internal:7777"


@pytest.mark.asyncio
async def test_pause_manager_register_resolve_and_cancel():
    manager = _PauseManager()
    future = await manager.register("approval-1", "exec-1")
    duplicate = await manager.register("approval-1", "exec-1")

    assert future is duplicate

    resolved = await manager.resolve_by_execution_id(
        "exec-1", SimpleNamespace(decision="approved")
    )
    assert resolved is True
    assert (await future).decision == "approved"

    pending = await manager.register("approval-2", "exec-2")
    await manager.cancel_all()
    assert pending.cancelled()


def test_agent_callback_discovery_and_conversion_helpers(bare_agent, monkeypatch):
    monkeypatch.setattr("agentfield.agent._is_running_in_container", lambda: True)
    bare_agent._reasoner_registry = {
        "legacy": SimpleNamespace(id="legacy", return_type=list)
    }
    bare_agent._entry_to_metadata = lambda entry, kind: {
        "id": entry.id,
        "return_type": getattr(entry, "return_type", None),
    }

    bare_agent.callback_candidates = ["http://agent.test", "http://backup.test"]
    payload = bare_agent._build_callback_discovery_payload()
    assert payload["preferred"] == "http://agent.test"
    assert payload["container"] is True
    assert payload["callback_candidates"][1] == "http://backup.test"

    bare_agent._apply_discovery_response(
        {
            "resolved_base_url": "http://resolved.test",
            "callback_discovery": {"candidates": ["http://backup.test"]},
        }
    )
    assert bare_agent.base_url == "http://resolved.test"
    assert bare_agent.callback_candidates[0] == "http://resolved.test"

    assert bare_agent._get_target_return_type("legacy") is list


def test_agent_current_context_and_workflow_event_emission(bare_agent, monkeypatch):
    post_calls = []

    requests_module = types.ModuleType("requests")
    requests_module.post = lambda url, json, headers, timeout: (
        post_calls.append((url, json, headers, timeout)) or _Response(200, "ok")
    )
    monkeypatch.setitem(sys.modules, "requests", requests_module)

    context = bare_agent._get_current_execution_context()
    assert isinstance(context, ExecutionContext)
    assert context.agent_node_id == "node-1"

    bare_agent._emit_workflow_event_sync(
        context,
        "summarize",
        "succeeded",
        input_data={"x": 1},
        result={"ok": True},
        duration_ms=12,
    )

    assert post_calls
    url, payload, headers, timeout = post_calls[0]
    assert url.endswith("/api/v1/workflow/executions/events")
    assert payload["reasoner_id"] == "summarize"
    assert headers["X-API-Key"] == "secret"
    assert timeout == 5


def test_agent_registers_memory_event_listeners(bare_agent, monkeypatch):
    async def handle_change(self, event):
        return event

    handle_change._memory_event_listener = True
    handle_change._memory_event_patterns = ["memory.*"]
    bare_agent.handle_change = MethodType(handle_change, bare_agent)

    subscriptions = []

    class EventClient:
        def __init__(self, *args):
            self.args = args

        def subscribe(self, patterns, listener):
            subscriptions.append((patterns, listener))

    monkeypatch.setattr("agentfield.agent.MemoryEventClient", EventClient)
    monkeypatch.setattr(
        "agentfield.agent.inspect.getmembers",
        lambda cls, predicate=None: [("handle_change", handle_change)],
    )
    bare_agent._get_current_execution_context = Mock(
        return_value=ExecutionContext.create_new("node-1", "wf")
    )

    bare_agent._register_memory_event_listeners()

    assert bare_agent.memory_event_client is not None
    assert subscriptions[0][0] == ["memory.*"]
