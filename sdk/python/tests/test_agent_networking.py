from __future__ import annotations

import pytest

from haxen_sdk import agent as agent_mod
from haxen_sdk.agent import (
    ExecutionContext,
    _build_callback_candidates,
    _normalize_candidate,
    _resolve_callback_url,
)
from types import SimpleNamespace

from tests.helpers import create_test_agent


def test_detect_container_ip_prefers_metadata(monkeypatch):
    calls = []

    class DummyResponse:
        def __init__(self, status, text=""):
            self.status_code = status
            self.text = text

        def json(self):
            return self.text

    def fake_get(url, headers=None, timeout=None):
        calls.append(url)
        if "latest/meta-data" in url:
            return DummyResponse(200, "198.51.100.5")
        if "metadata.google.internal" in url:
            return DummyResponse(200, "203.0.113.7")
        if "api.ipify.org" in url:
            return DummyResponse(200, "192.0.2.9")
        return DummyResponse(404, "")

    monkeypatch.setattr("requests.get", fake_get)

    detected = agent_mod._detect_container_ip()
    assert detected == "198.51.100.5"
    assert any("latest/meta-data" in url for url in calls)


def test_detect_container_ip_fallback_to_external(monkeypatch):
    class DummyResponse:
        def __init__(self, status, text=""):
            self.status_code = status
            self.text = text

        def json(self):
            raise ValueError

    sequence = [
        DummyResponse(404),
        DummyResponse(404),
        DummyResponse(404),
        DummyResponse(200, "203.0.113.9"),
    ]

    def fake_get(url, headers=None, timeout=None):
        return sequence.pop(0)

    monkeypatch.setattr("requests.get", fake_get)
    assert agent_mod._detect_container_ip() == "203.0.113.9"


def test_is_running_in_container_checks_dockerenv(monkeypatch, tmp_path):
    monkeypatch.setattr(agent_mod.os.path, "exists", lambda path: path == "/.dockerenv")
    monkeypatch.setattr(agent_mod.os, "environ", {})
    assert agent_mod._is_running_in_container() is True


def test_is_running_in_container_detects_env(monkeypatch):
    monkeypatch.setattr(agent_mod.os.path, "exists", lambda path: False)

    def fake_open(path, mode="r", *args, **kwargs):
        raise FileNotFoundError

    monkeypatch.setattr(agent_mod, "open", fake_open, raising=False)
    monkeypatch.setattr(agent_mod.os, "environ", {"KUBERNETES_SERVICE_HOST": "1"})

    assert agent_mod._is_running_in_container() is True


def test_normalize_candidate_variants():
    assert _normalize_candidate("example.com", 8080) == "http://example.com:8080"
    assert _normalize_candidate("https://demo:9090", 8080) == "https://demo:9090"
    assert _normalize_candidate("[2001:db8::1]", 7000) == "http://[2001:db8::1]:7000"
    assert _normalize_candidate("", 8000) is None


def test_build_callback_candidates_prefers_env(monkeypatch):
    monkeypatch.setattr(agent_mod, "_is_running_in_container", lambda: True)
    monkeypatch.setattr(agent_mod, "_detect_container_ip", lambda: "203.0.113.10")
    monkeypatch.setattr(agent_mod, "_detect_local_ip", lambda: "10.0.0.5")
    monkeypatch.setattr(agent_mod.socket, "gethostname", lambda: "agent-host")
    monkeypatch.setenv("AGENT_CALLBACK_URL", "https://env.example")
    monkeypatch.setenv("RAILWAY_SERVICE_NAME", "haxen")
    monkeypatch.setenv("RAILWAY_ENVIRONMENT", "prod")

    candidates = _build_callback_candidates(None, 9090)
    assert candidates[0] == "https://env.example:9090"
    assert any("railway.internal" in candidate for candidate in candidates)
    assert any(candidate.startswith("http://203.0.113.10") for candidate in candidates)
    assert any(candidate.startswith("http://10.0.0.5") for candidate in candidates)
    assert any(candidate.endswith(":9090") for candidate in candidates)


def test_resolve_callback_url_uses_first_candidate(monkeypatch):
    monkeypatch.setenv("AGENT_CALLBACK_URL", "http://from-env:7777")
    resolved = _resolve_callback_url(None, 7777)
    assert resolved == "http://from-env:7777"


def test_build_callback_discovery_payload_marks_container(monkeypatch):
    agent, _ = create_test_agent(monkeypatch)
    agent.callback_candidates = ["http://first:7000", "http://second:7000"]
    monkeypatch.setattr(agent_mod, "_is_running_in_container", lambda: True)

    payload = agent._build_callback_discovery_payload()
    assert payload["mode"] == "python-sdk:auto"
    assert payload["preferred"] is None
    assert payload["callback_candidates"][0] == "http://first:7000"
    assert payload["container"] is True


def test_apply_discovery_response_updates_candidates(monkeypatch):
    agent, _ = create_test_agent(monkeypatch)
    agent.callback_candidates = ["http://old:8000"]
    payload = {
        "resolved_base_url": "https://new:9000",
        "callback_discovery": {
            "candidates": ["https://new:9000", "http://fallback:9000"],
        },
    }

    agent._apply_discovery_response(payload)

    assert agent.base_url == "https://new:9000"
    assert agent.callback_candidates[0] == "https://new:9000"
    assert "http://fallback:9000" in agent.callback_candidates


def test_register_agent_with_did_enables_vc(monkeypatch):
    agent, _ = create_test_agent(monkeypatch)
    agent.reasoners = [
        {
            "id": "double",
            "input_schema": {"type": "object"},
            "output_schema": {"type": "object"},
        }
    ]
    agent.skills = [
        {
            "id": "helper",
            "input_schema": {"type": "object"},
            "tags": ["util"],
        }
    ]

    result = agent._register_agent_with_did()
    assert result is True
    assert agent.did_enabled is True
    assert agent.vc_generator.is_enabled() is True


def test_populate_execution_context_with_did(monkeypatch):
    agent, _ = create_test_agent(monkeypatch)
    execution = ExecutionContext.create_new(agent.node_id, "wf-1")
    did_context = SimpleNamespace(
        session_id="session-1",
        caller_did="did:caller:1",
        target_did="did:target:1",
        agent_node_did="did:agent:1",
    )

    agent._populate_execution_context_with_did(execution, did_context)

    assert execution.session_id == "session-1"
    assert execution.caller_did == "did:caller:1"
    assert execution.target_did == "did:target:1"
    assert execution.agent_node_did == "did:agent:1"


@pytest.mark.asyncio
async def test_on_change_decorator_registers_listener(monkeypatch):
    agent, _ = create_test_agent(monkeypatch)

    @agent.on_change(["user.*"])
    async def handler(event):
        return event

    monkeypatch.setattr(agent.__class__, "handle_user_change", handler, raising=False)

    # Trigger registration scan after method is attached
    agent._register_memory_event_listeners()

    subscriptions = getattr(agent.memory_event_client, "subscriptions", [])
    assert any(patterns == ["user.*"] for patterns, _ in subscriptions)
