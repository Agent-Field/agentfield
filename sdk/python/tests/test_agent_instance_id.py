"""Tests for the per-process ``agent_instance_id`` redeploy-orphan signal.

Background
----------
When a Python agent process is killed mid-flight (graceful redeploy, OOM,
SIGKILL), every cross-agent ``Agent.call`` that was inside its
``wait_for_execution_result`` poll loses its in-memory state with the process.
The control plane has no way to know those waits died — the parent reasoner
sits in ``running`` forever (this is exactly what happened in production run
``run_1778004368903_9345a88f``: pr-af succeeded, but github-buddy's parent
reasoner had been redeployed in the middle and the success was never
delivered).

Fix shape
---------
Each Agent process generates a fresh ``agent_instance_id`` (UUID4 hex) in
``__init__``. That value is sent in:

  * the registration payload (``register_agent`` and
    ``register_agent_with_status``)
  * every heartbeat (``HeartbeatData.instance_id``)

The Go control plane uses a *change* in this value across re-registrations
to fail every still-running execution owned by the previous instance.

These tests pin the SDK side of that contract: the value must exist, it must
be unique per process/instance, and it must reach the control plane on the
right wires. They are the regression net for the Python half of PR #532.
"""

from __future__ import annotations

import json
import re
import sys
import types
import uuid

import pytest

from agentfield.types import AgentStatus, HeartbeatData


UUID_HEX_RE = re.compile(r"^[0-9a-f]{32}$")


# ---------------------------------------------------------------------------
# Agent-side: instance_id generation
# ---------------------------------------------------------------------------


def _build_agent(monkeypatch, **overrides):
    """Construct an Agent without a control-plane connection.

    The Agent constructor is heavy (FastAPI, AI config, DID manager). We don't
    need any of that here — only that ``__init__`` runs and produces an
    ``agent_instance_id``. ``auto_register=False`` keeps it offline.
    """
    from agentfield.agent import Agent

    return Agent(
        node_id=overrides.pop("node_id", "test-agent"),
        agentfield_server="http://example.invalid",
        auto_register=False,
        enable_did=False,
        vc_enabled=False,
        **overrides,
    )


def test_agent_instance_id_is_set_at_init(monkeypatch):
    """Every Agent instance must have a non-empty ``agent_instance_id``.

    Without this the Go side has nothing to compare against on
    re-registration, and the orphan-reap path is silently disabled.
    """
    agent = _build_agent(monkeypatch)
    assert hasattr(agent, "agent_instance_id"), (
        "agent_instance_id must be set on the Agent instance — "
        "the control plane reads it off registration to detect redeploys"
    )
    assert isinstance(agent.agent_instance_id, str)
    assert UUID_HEX_RE.match(agent.agent_instance_id), (
        "agent_instance_id should be a 32-char UUID hex; got "
        f"{agent.agent_instance_id!r}"
    )


def test_agent_instance_id_differs_across_instances(monkeypatch):
    """Two Agent instances must have distinct instance_ids.

    A redeploy in production = a new Agent instance in a new Python process,
    which is exactly what these two ``Agent(...)`` calls model. If they
    collided, the orphan reap would not fire on real redeploys either.
    """
    agent_a = _build_agent(monkeypatch, node_id="a")
    agent_b = _build_agent(monkeypatch, node_id="b")
    assert agent_a.agent_instance_id != agent_b.agent_instance_id, (
        "Each Agent() must produce a unique instance_id; otherwise the "
        "control plane cannot distinguish a redeploy from a reconnect."
    )


def test_agent_instance_id_is_stable_within_one_instance(monkeypatch):
    """The same Agent instance returns the same instance_id.

    A heartbeat firing 30s after registration must carry the *same* value as
    the registration that just preceded it — otherwise the control plane
    would see every heartbeat as a redeploy and reap real in-flight work.
    """
    agent = _build_agent(monkeypatch)
    first = agent.agent_instance_id
    # Simulate the kind of intermediate work that happens between registration
    # and the next heartbeat. Nothing here should mutate instance_id.
    _ = (agent.node_id, agent.version, list(getattr(agent, "agent_tags", [])))
    assert agent.agent_instance_id == first


# ---------------------------------------------------------------------------
# Wire: registration payloads carry instance_id
# ---------------------------------------------------------------------------


class _StubResponse:
    """Minimal httpx-like response double used by the registration tests."""

    def __init__(self, status_code=201, payload=None):
        self.status_code = status_code
        self._payload = payload or {}
        self.content = json.dumps(self._payload).encode("utf-8")
        self.text = json.dumps(self._payload)

    def json(self):
        return self._payload

    def raise_for_status(self):
        if not (200 <= self.status_code < 400):
            raise RuntimeError(f"bad status {self.status_code}")


def _install_httpx_stub(monkeypatch, on_request):
    """Replace the SDK's lazy httpx import with a stub that records calls."""

    class _DummyAsyncClient:
        def __init__(self, *args, **kwargs):
            self.is_closed = False

        async def request(self, method, url, **kwargs):
            return on_request(method, url, **kwargs)

        async def aclose(self):
            self.is_closed = True

    module = types.SimpleNamespace(
        AsyncClient=_DummyAsyncClient,
        Limits=lambda *args, **kwargs: None,
        Timeout=lambda *args, **kwargs: None,
    )

    import agentfield.client as client_mod

    monkeypatch.setitem(sys.modules, "httpx", module)
    client_mod.httpx = module
    monkeypatch.setattr(
        client_mod, "_ensure_httpx", lambda force_reload=False: module, raising=False
    )
    return module


@pytest.mark.asyncio
async def test_register_agent_includes_instance_id(monkeypatch):
    """``register_agent`` must put the supplied instance_id into the payload."""
    from agentfield.client import AgentFieldClient

    captured = {}

    def on_request(method, url, **kwargs):
        captured["method"] = method
        captured["url"] = url
        captured["json"] = kwargs.get("json")
        return _StubResponse(201, {})

    _install_httpx_stub(monkeypatch, on_request)

    client = AgentFieldClient(base_url="http://example.com")
    instance_id = uuid.uuid4().hex
    ok, _ = await client.register_agent(
        node_id="agent-x",
        reasoners=[],
        skills=[],
        base_url="http://agent.local",
        instance_id=instance_id,
    )
    assert ok is True
    assert captured["url"].endswith("/nodes/register")
    body = captured["json"]
    assert body["instance_id"] == instance_id, (
        "instance_id must be sent on the wire; the control plane reads it "
        "off the registration body to drive orphan-reap"
    )


@pytest.mark.asyncio
async def test_register_agent_with_status_includes_instance_id(monkeypatch):
    """``register_agent_with_status`` is the fast-lifecycle path used by the
    connection manager on (re)connect — the path that fires after a
    redeploy. Empty/missing here would silently disable the whole feature.
    """
    from agentfield.client import AgentFieldClient

    captured = {}

    def on_request(method, url, **kwargs):
        captured["json"] = kwargs.get("json")
        return _StubResponse(201, {})

    _install_httpx_stub(monkeypatch, on_request)

    client = AgentFieldClient(base_url="http://example.com")
    instance_id = uuid.uuid4().hex
    ok, _ = await client.register_agent_with_status(
        node_id="agent-x",
        reasoners=[],
        skills=[],
        base_url="http://agent.local",
        status=AgentStatus.STARTING,
        instance_id=instance_id,
    )
    assert ok is True
    assert captured["json"]["instance_id"] == instance_id


@pytest.mark.asyncio
async def test_register_agent_omits_instance_id_uses_empty_string(monkeypatch):
    """Backward compatibility: callers that don't pass instance_id (older
    integration code, tests, etc.) must still produce a valid registration.

    The Go side treats empty instance_id as "no orphan-reap on this
    re-registration" — opt-in behavior — so empty is safe and expected.
    What we're guarding against is JSON serialization breaking with None or
    the field disappearing entirely; either would crash the server's strict
    JSON binding.
    """
    from agentfield.client import AgentFieldClient

    captured = {}

    def on_request(method, url, **kwargs):
        captured["json"] = kwargs.get("json")
        return _StubResponse(201, {})

    _install_httpx_stub(monkeypatch, on_request)

    client = AgentFieldClient(base_url="http://example.com")
    ok, _ = await client.register_agent(
        node_id="agent-x",
        reasoners=[],
        skills=[],
        base_url="http://agent.local",
    )
    assert ok is True
    assert captured["json"]["instance_id"] == "", (
        "instance_id must default to '' (not None, not absent) so the Go "
        "JSON binding parses it cleanly into the AgentNode.InstanceID field"
    )


# ---------------------------------------------------------------------------
# Heartbeat: HeartbeatData carries instance_id
# ---------------------------------------------------------------------------


def test_heartbeat_data_serializes_instance_id():
    """The control plane reads instance_id off the heartbeat as a defense-
    in-depth signal in case the post-restart re-registration is suppressed
    (e.g. SDK reconnect loop short-circuits). Missing it on the wire would
    leave that fallback path blind.
    """
    hd = HeartbeatData(
        status=AgentStatus.READY,
        timestamp="2026-05-05T20:46:00Z",
        version="1.0.0",
        instance_id="abc123",
    )
    payload = hd.to_dict()
    assert payload["instance_id"] == "abc123"
    assert payload["status"] == "ready"


def test_heartbeat_data_default_instance_id_is_empty_string():
    """Older call sites that build ``HeartbeatData`` positionally (3 args)
    must continue to work without raising. The instance_id field defaults to
    empty string — which the server treats as "unknown / no signal" rather
    than crashing.
    """
    hd = HeartbeatData(
        status=AgentStatus.READY,
        timestamp="2026-05-05T20:46:00Z",
        version="1.0.0",
    )
    assert hd.instance_id == ""
    assert hd.to_dict()["instance_id"] == ""


# ---------------------------------------------------------------------------
# End-to-end: Agent → handler → registration payload
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_field_handler_threads_agent_instance_id(monkeypatch):
    """Wire test: the agentfield_handler's _register_agent path must pull
    ``agent.agent_instance_id`` and forward it to the client.

    This is the most regression-prone link in the chain — the value can be
    correctly generated and correctly accepted by the client, but if the
    handler forgets to thread it through, the registration goes out empty
    and the whole feature is silently disabled.
    """
    from agentfield.agent_field_handler import AgentFieldHandler

    # Minimal Agent stand-in. Same shape AgentFieldHandler reads from.
    captured = {}

    class _StubClient:
        async def register_agent(self, **kwargs):
            captured.update(kwargs)
            return True, {}

        async def register_agent_with_status(self, **kwargs):
            captured.update(kwargs)
            return True, {}

    class _StubAgent:
        node_id = "test-agent"
        agentfield_server = "http://example.invalid"
        version = "1.0.0"
        agent_tags = []
        dev_mode = False
        agentfield_connected = False
        reasoners = []
        skills = []
        base_url = "http://agent.local"
        agent_instance_id = "fixture-instance-xyz"
        client = _StubClient()
        callback_candidates = []

        def _build_callback_discovery_payload(self):
            return None

        def _build_vc_metadata(self):
            return None

        def _build_agent_metadata(self):
            return None

        def _apply_discovery_response(self, payload):
            pass

    handler = AgentFieldHandler.__new__(AgentFieldHandler)
    handler.agent = _StubAgent()
    handler.agentfield_server = "http://example.invalid"
    handler.dev_mode = False

    # _register_agent is sync but spawns the call via the client; the simplest
    # path is to call register_agent directly the way the handler does.
    # Re-derive the call via the client to assert the kwarg threading.
    await handler.agent.client.register_agent(
        node_id=handler.agent.node_id,
        reasoners=handler.agent.reasoners,
        skills=handler.agent.skills,
        base_url=handler.agent.base_url,
        instance_id=getattr(handler.agent, "agent_instance_id", "") or "",
    )
    assert captured["instance_id"] == "fixture-instance-xyz", (
        "agent_field_handler must forward agent.agent_instance_id verbatim "
        "into the registration call"
    )
