from __future__ import annotations

import pytest
from urllib.parse import urlparse

from agentfield import agent as agent_mod
from agentfield.agent import (
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
        parsed = urlparse(url)
        if parsed.netloc == "169.254.169.254" and parsed.path == "/latest/meta-data/public-ipv4":
            return DummyResponse(200, "198.51.100.5")
        if parsed.netloc == "metadata.google.internal" and parsed.path == "/computeMetadata/v1/instance/network-interfaces/0/access-configs/0/external-ip":
            return DummyResponse(200, "203.0.113.7")
        if parsed.scheme == "https" and parsed.netloc == "api.ipify.org" and parsed.path in {"", "/"}:
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
    monkeypatch.delenv("AGENTFIELD_DISABLE_IP_DETECTION", raising=False)
    monkeypatch.setenv("AGENT_CALLBACK_URL", "https://env.example")
    monkeypatch.setenv("RAILWAY_SERVICE_NAME", "agentfield")
    monkeypatch.setenv("RAILWAY_ENVIRONMENT", "prod")

    candidates = _build_callback_candidates(None, 9090)
    assert candidates[0] == "https://env.example:9090"
    assert any("railway.internal" in candidate for candidate in candidates)
    assert any(candidate.startswith("http://10.0.0.5") for candidate in candidates)
    assert any(candidate.endswith(":9090") for candidate in candidates)
    # The configured callback URL suppresses the public-IP probe, so the
    # detected address is not offered as a candidate.
    assert not any(
        candidate.startswith("http://203.0.113.10") for candidate in candidates
    )


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
    # Verify DID credentials were wired to the HTTP client
    assert agent.client.did_credentials is not None
    assert agent.client.did_credentials[0] == "did:agent:test-agent"


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


def test_reasoner_and_skill_vc_metadata(monkeypatch):
    agent, _ = create_test_agent(monkeypatch)

    @agent.reasoner(vc_enabled=False)
    async def sample_reasoner(text: str) -> dict:
        return {"text": text}

    @agent.skill(vc_enabled=False)
    def sample_skill(amount: int) -> int:
        return amount

    assert agent.reasoners[-1]["vc_enabled"] is False
    assert agent.skills[-1]["vc_enabled"] is False


def test_vc_policy_overrides_precedence(monkeypatch):
    agent, _ = create_test_agent(monkeypatch, vc_enabled=False)
    agent.did_enabled = True
    if agent.vc_generator:
        agent.vc_generator.set_enabled(True)

    @agent.reasoner(name="critical", vc_enabled=True)
    async def critical_reasoner(text: str) -> dict:
        return {"text": text}

    @agent.skill(name="bulk", vc_enabled=True)
    def bulk_skill(amount: int) -> int:
        return amount

    assert agent._should_generate_vc("critical", agent._reasoner_vc_overrides) is True
    assert agent._should_generate_vc("fallback", agent._reasoner_vc_overrides) is False
    assert agent._should_generate_vc("bulk", agent._skill_vc_overrides) is True

    metadata = agent._build_vc_metadata()
    assert metadata["agent_default"] is False
    assert metadata["reasoner_overrides"]["critical"] is True
    assert metadata["effective_reasoners"].get("critical") is True


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


# ---------------------------------------------------------------------------
# Outbound IP detection opt-out (issue #624)
#
# _detect_container_ip() reaches out to the cloud metadata endpoints and to
# api.ipify.org. These tests pin down when that probe is allowed to fire.
# ---------------------------------------------------------------------------


def _stub_callback_environment(
    monkeypatch,
    *,
    container_ip="203.0.113.10",
    local_ip="10.0.0.5",
    hostname="agent-host",
    in_container=True,
):
    """Neutralise every networking side effect and record probe invocations.

    Returns the list that receives one entry per _detect_container_ip() call, so
    a test can assert on whether the probe fired rather than on how the code is
    structured.
    """

    probe_calls = []

    def fake_detect_container_ip():
        probe_calls.append(container_ip)
        return container_ip

    monkeypatch.setattr(agent_mod, "_is_running_in_container", lambda: in_container)
    monkeypatch.setattr(agent_mod, "_detect_container_ip", fake_detect_container_ip)
    monkeypatch.setattr(agent_mod, "_detect_local_ip", lambda: local_ip)
    monkeypatch.setattr(agent_mod.socket, "gethostname", lambda: hostname)
    for name in (
        "AGENT_CALLBACK_URL",
        "AGENTFIELD_DISABLE_IP_DETECTION",
        "RAILWAY_SERVICE_NAME",
        "RAILWAY_ENVIRONMENT",
    ):
        monkeypatch.delenv(name, raising=False)

    return probe_calls


def test_container_ip_probe_runs_when_nothing_is_configured(monkeypatch):
    """Default behaviour in a container is unchanged: probe, and use the result."""
    probe_calls = _stub_callback_environment(monkeypatch)

    candidates = _build_callback_candidates(None, 9000)

    assert probe_calls == ["203.0.113.10"]
    assert "http://203.0.113.10:9000" in candidates


def test_container_ip_probe_skipped_for_explicit_callback_argument(monkeypatch):
    probe_calls = _stub_callback_environment(monkeypatch)

    candidates = _build_callback_candidates("https://agent.svc.example:8443", 9000)

    assert probe_calls == []
    assert candidates[0] == "https://agent.svc.example:8443"
    assert not any("203.0.113.10" in candidate for candidate in candidates)


def test_container_ip_probe_skipped_for_callback_url_env_var(monkeypatch):
    probe_calls = _stub_callback_environment(monkeypatch)
    monkeypatch.setenv("AGENT_CALLBACK_URL", "http://my-agent.default.svc:8001")

    candidates = _build_callback_candidates(None, 9000)

    assert probe_calls == []
    assert candidates[0] == "http://my-agent.default.svc:8001"
    assert not any("203.0.113.10" in candidate for candidate in candidates)


def test_explicit_callback_argument_is_preferred_over_env_var(monkeypatch):
    probe_calls = _stub_callback_environment(monkeypatch)
    monkeypatch.setenv("AGENT_CALLBACK_URL", "http://from-env:8001")

    candidates = _build_callback_candidates("http://from-arg:8002", 9000)

    assert probe_calls == []
    assert candidates[0] == "http://from-arg:8002"
    assert "http://from-env:8001" in candidates


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "Yes", "  yes  "])
def test_disable_flag_truthy_values_skip_the_probe(monkeypatch, value):
    probe_calls = _stub_callback_environment(monkeypatch)
    monkeypatch.setenv("AGENTFIELD_DISABLE_IP_DETECTION", value)

    candidates = _build_callback_candidates(None, 9000)

    assert probe_calls == []
    assert not any("203.0.113.10" in candidate for candidate in candidates)


@pytest.mark.parametrize("value", ["", "0", "false", "no", "off", "maybe"])
def test_disable_flag_non_truthy_values_leave_the_probe_enabled(monkeypatch, value):
    probe_calls = _stub_callback_environment(monkeypatch)
    monkeypatch.setenv("AGENTFIELD_DISABLE_IP_DETECTION", value)

    candidates = _build_callback_candidates(None, 9000)

    assert probe_calls == ["203.0.113.10"]
    assert "http://203.0.113.10:9000" in candidates


def test_unusable_callback_url_still_allows_the_probe(monkeypatch):
    """A value that normalizes to nothing is not a configured callback URL."""
    probe_calls = _stub_callback_environment(monkeypatch)

    candidates = _build_callback_candidates("   ", 9000)

    assert probe_calls == ["203.0.113.10"]
    assert "http://203.0.113.10:9000" in candidates


def test_skipping_the_probe_keeps_every_other_candidate(monkeypatch):
    probe_calls = _stub_callback_environment(monkeypatch)
    monkeypatch.setenv("AGENTFIELD_DISABLE_IP_DETECTION", "1")
    monkeypatch.setenv("RAILWAY_SERVICE_NAME", "my-service")
    monkeypatch.setenv("RAILWAY_ENVIRONMENT", "production")

    candidates = _build_callback_candidates(None, 9000)

    assert probe_calls == []
    for expected in (
        "http://my-service.railway.internal:9000",
        "http://10.0.0.5:9000",
        "http://agent-host:9000",
        "http://host.docker.internal:9000",
        "http://localhost:9000",
        "http://127.0.0.1:9000",
    ):
        assert expected in candidates


def test_probe_never_runs_outside_a_container(monkeypatch):
    probe_calls = _stub_callback_environment(monkeypatch, in_container=False)

    candidates = _build_callback_candidates(None, 9000)

    assert probe_calls == []
    assert candidates[0] == "http://10.0.0.5:9000"


def test_resolve_callback_url_with_detection_disabled_falls_back_locally(monkeypatch):
    _stub_callback_environment(monkeypatch, local_ip=None, hostname="")
    monkeypatch.setenv("AGENTFIELD_DISABLE_IP_DETECTION", "true")

    assert _resolve_callback_url(None, 8080) == "http://host.docker.internal:8080"


# ---------------------------------------------------------------------------
# The same guarantee, one level lower.
#
# The tests above stub _detect_container_ip itself, so they prove "the helper is
# not called". The literal symptom in #624 is stronger: no HTTP request leaves
# the process. The two tests below leave _detect_container_ip in place and
# assert against its only outbound entry point, requests.get.
# ---------------------------------------------------------------------------


def _stub_everything_but_the_probe(monkeypatch):
    """Force the in-container branch without touching _detect_container_ip."""

    monkeypatch.setattr(agent_mod, "_is_running_in_container", lambda: True)
    monkeypatch.setattr(agent_mod, "_detect_local_ip", lambda: "10.0.0.5")
    monkeypatch.setattr(agent_mod.socket, "gethostname", lambda: "agent-host")
    for name in (
        "AGENT_CALLBACK_URL",
        "AGENTFIELD_DISABLE_IP_DETECTION",
        "RAILWAY_SERVICE_NAME",
        "RAILWAY_ENVIRONMENT",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize(
    "callback_url, env",
    [
        ("https://agent.svc.example:8443", {}),
        (None, {"AGENT_CALLBACK_URL": "http://my-agent.default.svc:8001"}),
        (None, {"AGENTFIELD_DISABLE_IP_DETECTION": "1"}),
    ],
    ids=["callback_argument", "callback_env_var", "disable_flag"],
)
def test_no_http_request_is_made_when_the_callback_url_is_known(
    monkeypatch, callback_url, env
):
    """No packet is aimed at the metadata services or api.ipify.org.

    ``requests.get`` is ``_detect_container_ip``'s only outbound entry point
    (it imports ``requests`` lazily and makes no other network call), so a
    tripwire there covers every probe target. The tripwire *records* each
    attempt as well as raising, because ``_detect_container_ip`` wraps each
    request in ``except Exception: pass`` — a bare ``AssertionError`` would be
    swallowed by the code under test, and the recorded list would not be.
    """

    _stub_everything_but_the_probe(monkeypatch)
    for name, value in env.items():
        monkeypatch.setenv(name, value)

    attempted = []

    def forbidden_get(url, *args, **kwargs):
        attempted.append(url)
        raise AssertionError("no HTTP probe expected")

    monkeypatch.setattr("requests.get", forbidden_get)

    candidates = _build_callback_candidates(callback_url, 9000)

    assert attempted == []
    assert candidates, "the agent still needs somewhere to be called back on"


def test_http_request_is_made_when_nothing_is_configured(monkeypatch):
    """Control case: the tripwire above would fire if the probe still ran."""

    _stub_everything_but_the_probe(monkeypatch)

    attempted = []

    class DummyResponse:
        status_code = 200
        text = "198.51.100.5"

    def fake_get(url, *args, **kwargs):
        attempted.append(url)
        return DummyResponse()

    monkeypatch.setattr("requests.get", fake_get)

    candidates = _build_callback_candidates(None, 9000)

    assert attempted == ["http://169.254.169.254/latest/meta-data/public-ipv4"]
    assert "http://198.51.100.5:9000" in candidates
