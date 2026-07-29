"""Regression test for issue #624 — startup must not probe external IP services."""

import requests

from agentfield import Agent
from agentfield import agent as agent_mod


def test_issue_624(monkeypatch):
    """An explicit Kubernetes callback URL must avoid metadata and ipify probes."""
    requests_made = []

    class DummyResponse:
        status_code = 404
        text = ""

    def fake_get(url, *args, **kwargs):
        requests_made.append(url)
        return DummyResponse()

    monkeypatch.setattr(agent_mod, "_is_running_in_container", lambda: True)
    monkeypatch.setattr(requests, "get", fake_get)

    Agent(
        node_id="issue-624-agent",
        callback_url="http://issue-624-agent.default.svc.cluster.local:8001",
        auto_register=False,
        enable_did=False,
    )

    assert requests_made == []
