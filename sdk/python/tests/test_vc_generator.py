import base64
from datetime import datetime
from types import SimpleNamespace

from agentfield.vc_generator import VCGenerator


def make_execution_context():
    return SimpleNamespace(
        execution_id="exec-1",
        workflow_id="wf-1",
        session_id="sess-1",
        caller_did="did:caller",
        target_did="did:target",
        agent_node_did="did:agent",
        timestamp=datetime.utcnow(),
    )


def test_generate_execution_vc_success(monkeypatch):
    generator = VCGenerator("http://agentfield")
    generator.set_enabled(True)

    payload = {
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
        "created_at": datetime.utcnow().isoformat() + "Z",
    }

    def fake_post(url, json=None, timeout=None):
        assert url.endswith("/execution/vc")
        return SimpleNamespace(status_code=200, json=lambda: payload)

    monkeypatch.setattr("agentfield.vc_generator.requests.post", fake_post)

    vc = generator.generate_execution_vc(
        make_execution_context(), {"x": 1}, {"y": 2}, status="succeeded"
    )
    assert vc.execution_id == "exec-1"


def test_generate_execution_vc_disabled():
    generator = VCGenerator("http://agentfield")
    generator.set_enabled(False)
    assert (
        generator.generate_execution_vc(
            make_execution_context(), None, None, status="succeeded"
        )
        is None
    )


def test_verify_vc(monkeypatch):
    generator = VCGenerator("http://agentfield")

    def fake_post(url, json=None, timeout=None):
        return SimpleNamespace(status_code=200, json=lambda: {"valid": True})

    monkeypatch.setattr("agentfield.vc_generator.requests.post", fake_post)
    result = generator.verify_vc({"proof": {}})
    assert result == {"valid": True}


def test_create_workflow_vc(monkeypatch):
    generator = VCGenerator("http://agentfield")
    payload = {
        "workflow_id": "wf-1",
        "session_id": "sess-1",
        "component_vcs": ["vc-1"],
        "workflow_vc_id": "wvc-1",
        "status": "succeeded",
        "start_time": datetime.utcnow().isoformat() + "Z",
        "end_time": datetime.utcnow().isoformat() + "Z",
        "total_steps": 1,
        "completed_steps": 1,
    }

    def fake_post(url, json=None, timeout=None):
        return SimpleNamespace(status_code=200, json=lambda: payload)

    monkeypatch.setattr("agentfield.vc_generator.requests.post", fake_post)
    vc = generator.create_workflow_vc("wf-1", "sess-1", ["vc-1"])
    assert vc.workflow_vc_id == "wvc-1"


def test_get_workflow_vc_chain(monkeypatch):
    generator = VCGenerator("http://agentfield")

    def fake_get(url, timeout=None):
        return SimpleNamespace(status_code=200, json=lambda: {"chain": ["vc-1"]})

    monkeypatch.setattr("agentfield.vc_generator.requests.get", fake_get)
    chain = generator.get_workflow_vc_chain("wf-1")
    assert chain == {"chain": ["vc-1"]}


def test_serialize_data_for_json_base64():
    generator = VCGenerator("http://agentfield")
    generator.set_enabled(True)
    encoded = generator._serialize_data_for_json({"a": 1})
    decoded = base64.b64decode(encoded.encode()).decode()
    assert decoded == '{"a": 1}'
