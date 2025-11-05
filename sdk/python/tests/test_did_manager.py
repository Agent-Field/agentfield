import datetime

from agentfield.did_manager import DIDManager, DIDIdentityPackage


def make_package():
    return {
        "agent_did": {
            "did": "did:agent:123",
            "private_key_jwk": "priv",
            "public_key_jwk": "pub",
            "derivation_path": "m/0",
            "component_type": "agent",
        },
        "reasoner_dids": {
            "reasoner_a": {
                "did": "did:reasoner:a",
                "private_key_jwk": "priv_a",
                "public_key_jwk": "pub_a",
                "derivation_path": "m/1",
                "component_type": "reasoner",
            }
        },
        "skill_dids": {
            "skill_b": {
                "did": "did:skill:b",
                "private_key_jwk": "priv_b",
                "public_key_jwk": "pub_b",
                "derivation_path": "m/2",
                "component_type": "skill",
            }
        },
        "agentfield_server_id": "agentfield-1",
    }


def test_register_agent_success(monkeypatch):
    manager = DIDManager("http://agentfield", "node")

    class DummyResponse:
        status_code = 200

        @staticmethod
        def json():
            return {"success": True, "identity_package": make_package()}

    monkeypatch.setattr("requests.post", lambda *a, **k: DummyResponse())

    ok = manager.register_agent([], [])
    assert ok is True
    assert manager.is_enabled() is True
    assert manager.get_agent_did() == "did:agent:123"
    assert manager.get_function_did("reasoner_a") == "did:reasoner:a"
    assert manager.get_function_did("unknown") == "did:agent:123"


def test_register_agent_failure_status(monkeypatch):
    manager = DIDManager("http://agentfield", "node")

    class DummyResponse:
        status_code = 500
        text = "boom"

    monkeypatch.setattr("requests.post", lambda *a, **k: DummyResponse())
    ok = manager.register_agent([], [])
    assert ok is False
    assert manager.is_enabled() is False


def test_create_execution_context(monkeypatch):
    manager = DIDManager("http://agentfield", "node")
    package = manager._parse_identity_package(make_package())
    assert isinstance(package, DIDIdentityPackage)
    manager.identity_package = package
    manager.enabled = True

    ctx = manager.create_execution_context(
        execution_id="exec1",
        workflow_id="wf1",
        session_id="sess",
        caller_function="reasoner_a",
        target_function="skill_b",
    )
    assert ctx is not None
    assert ctx.caller_did == "did:reasoner:a"
    assert ctx.target_did == "did:skill:b"
    assert isinstance(ctx.timestamp, datetime.datetime)


def test_create_execution_context_missing_identity():
    manager = DIDManager("http://agentfield", "node")
    assert manager.create_execution_context("e", "w", "s", "a", "b") is None


def test_get_identity_summary_disabled():
    manager = DIDManager("http://agentfield", "node")
    summary = manager.get_identity_summary()
    assert summary["enabled"] is False
