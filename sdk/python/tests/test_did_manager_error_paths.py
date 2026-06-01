import json
from typing import Any, Dict, Optional

import requests

from agentfield.agent import Agent
from agentfield.did_manager import DIDManager


def make_package() -> Dict[str, Any]:
    return {
        "agent_did": {
            "did": "did:agent:123",
            "private_key_jwk": "priv",
            "public_key_jwk": "pub",
            "derivation_path": "m/0",
            "component_type": "agent",
        },
        "reasoner_dids": {},
        "skill_dids": {},
        "agentfield_server_id": "agentfield-1",
    }


class DummyResponse:
    def __init__(
        self,
        status_code: int,
        payload: Optional[Dict[str, Any]] = None,
        text: str = "",
        json_error: Optional[Exception] = None,
    ):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text
        self._json_error = json_error

    def json(self) -> Dict[str, Any]:
        if self._json_error is not None:
            raise self._json_error
        return self._payload


def test_register_agent_timeout_disables_manager_and_returns_false(monkeypatch):
    manager = DIDManager("http://agentfield", "node-1")

    def fake_post(*args, **kwargs):
        raise requests.exceptions.Timeout("timed out")

    monkeypatch.setattr("agentfield.did_manager.requests.post", fake_post)

    ok = manager.register_agent([], [])

    assert ok is False
    assert manager.enabled is False
    assert manager.is_enabled() is False
    assert manager.identity_package is None


def test_register_agent_http_errors_disable_manager_without_retry(monkeypatch):
    manager = DIDManager("http://agentfield", "node-1")
    calls = []

    def fake_post(*args, **kwargs):
        calls.append((args, kwargs))
        return DummyResponse(status_code=500, text="boom")

    monkeypatch.setattr("agentfield.did_manager.requests.post", fake_post)

    ok = manager.register_agent([], [])

    assert ok is False
    assert manager.enabled is False
    assert manager.identity_package is None
    assert len(calls) == 1


def test_register_agent_503_disables_manager_without_retry(monkeypatch):
    manager = DIDManager("http://agentfield", "node-1")
    calls = []

    def fake_post(*args, **kwargs):
        calls.append((args, kwargs))
        return DummyResponse(status_code=503, text="unavailable")

    monkeypatch.setattr("agentfield.did_manager.requests.post", fake_post)

    ok = manager.register_agent([], [])

    assert ok is False
    assert manager.enabled is False
    assert manager.identity_package is None
    assert len(calls) == 1


def test_register_agent_invalid_json_disables_manager_cleanly(monkeypatch):
    manager = DIDManager("http://agentfield", "node-1")

    def fake_post(*args, **kwargs):
        return DummyResponse(
            status_code=200,
            json_error=json.JSONDecodeError("bad json", "{", 1),
        )

    monkeypatch.setattr("agentfield.did_manager.requests.post", fake_post)

    ok = manager.register_agent([], [])

    assert ok is False
    assert manager.enabled is False
    assert manager.identity_package is None


def test_register_agent_forwards_api_key_header(monkeypatch):
    manager = DIDManager("http://agentfield", "node-1", api_key="secret-key")
    captured: Dict[str, Any] = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        captured["timeout"] = timeout
        return DummyResponse(
            status_code=200,
            payload={"success": True, "identity_package": make_package()},
        )

    monkeypatch.setattr("agentfield.did_manager.requests.post", fake_post)

    ok = manager.register_agent([{"id": "r"}], [{"id": "s"}])

    assert ok is True
    assert captured["url"].endswith("/api/v1/did/register")
    assert captured["headers"]["Content-Type"] == "application/json"
    assert captured["headers"]["X-API-Key"] == "secret-key"
    assert captured["timeout"] == 30


def test_agent_continues_to_function_after_did_registration_failure(monkeypatch):
    agent = Agent(
        node_id="did-failure-agent",
        agentfield_server="http://agentfield",
        auto_register=False,
        enable_mcp=False,
        enable_did=False,
    )
    agent.did_manager = DIDManager("http://agentfield", agent.node_id)

    @agent.reasoner()
    def echo(value: str) -> dict:
        return {"value": value}

    def fake_post(*args, **kwargs):
        raise requests.exceptions.Timeout("timed out")

    monkeypatch.setattr("agentfield.did_manager.requests.post", fake_post)

    did_ok = agent._register_agent_with_did()
    result = agent.handle_serverless({"reasoner": "echo", "input": {"value": "ok"}})

    assert did_ok is False
    assert agent.did_enabled is False
    assert agent.vc_generator is None or agent.vc_generator.is_enabled() is False
    assert result["statusCode"] == 200
    assert result["body"] == {"value": "ok"}
