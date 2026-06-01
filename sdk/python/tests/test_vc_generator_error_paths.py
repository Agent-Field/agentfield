import json
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Dict, Optional

import requests
import pytest

from agentfield.vc_generator import VCGenerator


def make_execution_context():
    return SimpleNamespace(
        execution_id="exec-1",
        workflow_id="wf-1",
        session_id="sess-1",
        caller_did="did:caller",
        target_did="did:target",
        agent_node_did="did:agent",
        timestamp=datetime.now(timezone.utc),
    )


def make_execution_payload() -> Dict[str, Any]:
    return {
        "vc_id": "vc-1",
        "execution_id": "exec-1",
        "workflow_id": "wf-1",
        "session_id": "sess-1",
        "issuer_did": "did:issuer",
        "target_did": "did:target",
        "caller_did": "did:caller",
        "vc_document": {"proof": {}},
        "signature": "sig",
        "input_hash": "hash-in",
        "output_hash": "hash-out",
        "status": "succeeded",
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[
            :-3
        ]
        + "Z",
    }


def make_workflow_payload() -> Dict[str, Any]:
    return {
        "workflow_id": "wf-1",
        "session_id": "sess-1",
        "component_vcs": ["vc-1"],
        "workflow_vc_id": "wvc-1",
        "status": "succeeded",
        "start_time": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[
            :-3
        ]
        + "Z",
        "end_time": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[
            :-3
        ]
        + "Z",
        "total_steps": 1,
        "completed_steps": 1,
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


def invoke_generator(generator: VCGenerator, method_name: str):
    if method_name == "generate_execution_vc":
        return generator.generate_execution_vc(
            make_execution_context(),
            {"input": True},
            {"output": True},
            status="succeeded",
        )
    if method_name == "create_workflow_vc":
        return generator.create_workflow_vc("wf-1", "sess-1", ["vc-1"])
    raise AssertionError(f"Unsupported method: {method_name}")


@pytest.mark.parametrize(
    ("method_name", "expected_suffix"),
    [
        ("generate_execution_vc", "/api/v1/execution/vc"),
        ("create_workflow_vc", "/api/v1/did/workflow/wf-1/vc"),
    ],
)
def test_vc_generation_timeout_returns_none(monkeypatch, method_name, expected_suffix):
    generator = VCGenerator("http://agentfield")
    generator.set_enabled(True)
    captured = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured["url"] = url
        raise requests.exceptions.Timeout("timed out")

    monkeypatch.setattr("agentfield.vc_generator.requests.post", fake_post)

    result = invoke_generator(generator, method_name)

    assert result is None
    assert captured["url"].endswith(expected_suffix)


@pytest.mark.parametrize("status_code", [500, 503])
@pytest.mark.parametrize("method_name", ["generate_execution_vc", "create_workflow_vc"])
def test_vc_generation_http_errors_return_none(monkeypatch, status_code, method_name):
    generator = VCGenerator("http://agentfield")
    generator.set_enabled(True)
    calls = []

    def fake_post(*args, **kwargs):
        calls.append((args, kwargs))
        return DummyResponse(status_code=status_code, text="server error")

    monkeypatch.setattr("agentfield.vc_generator.requests.post", fake_post)

    result = invoke_generator(generator, method_name)

    assert result is None
    assert len(calls) == 1


@pytest.mark.parametrize("method_name", ["generate_execution_vc", "create_workflow_vc"])
def test_vc_generation_invalid_json_returns_none(monkeypatch, method_name):
    generator = VCGenerator("http://agentfield")
    generator.set_enabled(True)

    def fake_post(*args, **kwargs):
        return DummyResponse(
            status_code=200,
            json_error=json.JSONDecodeError("bad json", "{", 1),
        )

    monkeypatch.setattr("agentfield.vc_generator.requests.post", fake_post)

    result = invoke_generator(generator, method_name)

    assert result is None


def test_vc_generation_forwards_api_key_header(monkeypatch):
    generator = VCGenerator("http://agentfield", api_key="secret-key")
    generator.set_enabled(True)
    captured: Dict[str, Any] = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        return DummyResponse(status_code=200, payload=make_execution_payload())

    monkeypatch.setattr("agentfield.vc_generator.requests.post", fake_post)

    vc = generator.generate_execution_vc(
        make_execution_context(),
        {"x": 1},
        {"y": 2},
        status="succeeded",
    )

    assert vc is not None
    assert captured["url"].endswith("/api/v1/execution/vc")
    assert captured["headers"]["Content-Type"] == "application/json"
    assert captured["headers"]["X-API-Key"] == "secret-key"


def test_disabled_vc_generator_does_not_make_http_call(monkeypatch):
    generator = VCGenerator("http://agentfield")
    generator.set_enabled(False)
    called = {"count": 0}

    def fake_post(*args, **kwargs):
        called["count"] += 1
        return DummyResponse(status_code=200, payload=make_execution_payload())

    monkeypatch.setattr("agentfield.vc_generator.requests.post", fake_post)

    result = generator.generate_execution_vc(
        make_execution_context(),
        None,
        None,
        status="succeeded",
    )

    assert result is None
    assert called["count"] == 0
