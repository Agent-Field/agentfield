"""
Behavioral invariant tests for Agent lifecycle: registration, reasoner persistence,
node_id immutability, and discovery response stability.

These tests verify structural properties of the Agent that must always hold
regardless of implementation changes.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_agent(node_id: str = "test-lifecycle-node") -> "Agent":
    """Build a minimal Agent without network side-effects."""
    from agentfield.agent import Agent

    return Agent(node_id=node_id, agentfield_server="http://localhost:8080")


# ---------------------------------------------------------------------------
# 1. Reasoner registration is persistent
# ---------------------------------------------------------------------------

class TestReasonerRegistrationPersistence:
    """After @agent.reasoner("name"), the reasoner must be callable and remain registered."""

    def test_invariant_registered_reasoner_is_accessible(self):
        """
        A reasoner registered via @agent.reasoner("foo") must appear in
        agent.reasoners (the discovery dict) with key "foo".
        """
        agent = _make_agent("persist-test")

        @agent.reasoner("my-reasoner")
        def my_reasoner(x: int = 1) -> dict:
            return {"result": x}

        # Must be discoverable
        reasoners = agent.reasoners
        assert "my-reasoner" in reasoners, (
            f"INVARIANT VIOLATION: Registered reasoner 'my-reasoner' not found in "
            f"agent.reasoners. Keys present: {list(reasoners.keys())}"
        )

    def test_invariant_registered_reasoner_is_callable_via_handle_serverless(self):
        """
        A registered reasoner must be callable through handle_serverless
        and return a 200 response.
        """
        agent = _make_agent("callable-test")

        @agent.reasoner("greet")
        def greet(name: str = "world") -> dict:
            return {"hello": name}

        result = agent.handle_serverless({"reasoner": "greet", "input": {}})
        assert result["statusCode"] == 200, (
            f"INVARIANT VIOLATION: Registered reasoner 'greet' returned "
            f"statusCode={result['statusCode']} instead of 200. "
            f"Body: {result.get('body')}"
        )

    def test_invariant_multiple_reasoners_all_accessible(self):
        """All registered reasoners must be accessible simultaneously."""
        agent = _make_agent("multi-reasoner-test")

        @agent.reasoner("r-alpha")
        def r_alpha() -> dict:
            return {"id": "alpha"}

        @agent.reasoner("r-beta")
        def r_beta() -> dict:
            return {"id": "beta"}

        @agent.reasoner("r-gamma")
        def r_gamma() -> dict:
            return {"id": "gamma"}

        reasoners = agent.reasoners
        for name in ["r-alpha", "r-beta", "r-gamma"]:
            assert name in reasoners, (
                f"INVARIANT VIOLATION: Reasoner '{name}' not found after registration. "
                f"Present: {list(reasoners.keys())}"
            )


# ---------------------------------------------------------------------------
# 2. Double registration replaces
# ---------------------------------------------------------------------------

class TestDoubleRegistrationReplaces:
    """Registering the same name twice must replace the first registration."""

    def test_invariant_second_registration_wins(self):
        """
        Registering reasoner "foo" twice — the second definition must win.
        The agent must call the second function, not the first.
        """
        agent = _make_agent("replace-test")

        @agent.reasoner("foo")
        def foo_v1() -> dict:
            return {"version": 1}

        @agent.reasoner("foo")
        def foo_v2() -> dict:
            return {"version": 2}

        result = agent.handle_serverless({"reasoner": "foo", "input": {}})
        assert result["statusCode"] == 200, (
            f"INVARIANT VIOLATION: Double-registered 'foo' returned status {result['statusCode']}."
        )

        body = result.get("body", {})
        version = body.get("version") if isinstance(body, dict) else None
        assert version == 2, (
            f"INVARIANT VIOLATION: After double-registration, second definition did not win. "
            f"Got version={version!r} instead of 2. Response body: {body}"
        )

    def test_invariant_only_one_entry_per_name_after_double_registration(self):
        """
        After registering the same name twice, agent.reasoners must still
        contain exactly one entry for that name (no duplicates).
        """
        agent = _make_agent("dedup-test")

        @agent.reasoner("deduplicated")
        def v1() -> dict:
            return {}

        @agent.reasoner("deduplicated")
        def v2() -> dict:
            return {}

        # Reasoners must contain "deduplicated" exactly once
        keys = list(agent.reasoners.keys())
        count = keys.count("deduplicated")
        assert count == 1, (
            f"INVARIANT VIOLATION: After double-registration, 'deduplicated' appears "
            f"{count} times in agent.reasoners. Must appear exactly once."
        )


# ---------------------------------------------------------------------------
# 3. Unregistered reasoner returns 404
# ---------------------------------------------------------------------------

class TestUnregisteredReasonerReturns404:
    """Calling a non-existent reasoner through handle_serverless must return 404."""

    def test_invariant_unknown_reasoner_returns_404(self):
        """handle_serverless with an unknown reasoner name must return 404."""
        agent = _make_agent("404-test")

        result = agent.handle_serverless({"reasoner": "does-not-exist", "input": {}})

        assert result["statusCode"] == 404, (
            f"INVARIANT VIOLATION: Unknown reasoner returned statusCode={result['statusCode']} "
            "instead of 404. Clients depend on this to know the reasoner doesn't exist."
        )

    def test_invariant_unknown_reasoner_body_contains_error(self):
        """404 response body must contain an 'error' key with a meaningful message."""
        agent = _make_agent("404-body-test")

        result = agent.handle_serverless({"reasoner": "ghost-reasoner", "input": {}})

        body = result.get("body", {})
        assert "error" in body, (
            f"INVARIANT VIOLATION: 404 response body missing 'error' key. "
            f"Body: {body}"
        )

    def test_invariant_deregistering_all_reasoners_then_calling_returns_404(self):
        """
        If no reasoners are registered and handle_serverless is called with
        a reasoner name, it must return 404 (not 500 or 200).
        """
        agent = _make_agent("empty-registry-test")
        # No reasoners registered

        result = agent.handle_serverless({"reasoner": "anything", "input": {}})
        assert result["statusCode"] == 404, (
            f"INVARIANT VIOLATION: Empty registry + reasoner call returned "
            f"statusCode={result['statusCode']} instead of 404."
        )


# ---------------------------------------------------------------------------
# 4. Agent node_id is immutable
# ---------------------------------------------------------------------------

class TestNodeIdImmutability:
    """After creation, node_id must always return the same value."""

    def test_invariant_node_id_returns_same_value_on_repeated_access(self):
        """Accessing agent.node_id multiple times must always return the same value."""
        agent = _make_agent("my-immutable-node")

        ids = [agent.node_id for _ in range(10)]
        assert len(set(ids)) == 1, (
            f"INVARIANT VIOLATION: node_id changed across accesses: {ids}"
        )
        assert ids[0] == "my-immutable-node", (
            f"INVARIANT VIOLATION: node_id returned '{ids[0]}' instead of 'my-immutable-node'."
        )

    def test_invariant_node_id_is_the_value_passed_at_construction(self):
        """node_id must equal the value passed to Agent(node_id=...)."""
        test_id = "constructed-id-xyz"
        agent = _make_agent(test_id)

        assert agent.node_id == test_id, (
            f"INVARIANT VIOLATION: agent.node_id='{agent.node_id}' "
            f"does not match constructor input '{test_id}'."
        )

    def test_invariant_node_id_is_accessible_after_reasoner_registration(self):
        """node_id must remain unchanged after registering reasoners."""
        agent = _make_agent("stable-node")
        original_id = agent.node_id

        @agent.reasoner("some-reasoner")
        def some_reasoner() -> dict:
            return {}

        assert agent.node_id == original_id, (
            f"INVARIANT VIOLATION: node_id changed after reasoner registration. "
            f"Before: '{original_id}', After: '{agent.node_id}'"
        )


# ---------------------------------------------------------------------------
# 5. Discovery response stability
# ---------------------------------------------------------------------------

class TestDiscoveryResponseStability:
    """Discovery response must always contain 'node_id' and 'reasoners' keys."""

    def _get_discovery_body(self, agent) -> dict:
        """Call /discover and extract the response body."""
        result = agent.handle_serverless({"path": "/discover"})
        body = result.get("body", result)
        # handle double-wrapping
        if isinstance(body, dict) and "body" in body:
            body = body["body"]
        return body

    def test_invariant_discovery_response_contains_node_id(self):
        """Discovery response must always contain 'node_id'."""
        agent = _make_agent("discovery-node")

        body = self._get_discovery_body(agent)

        assert "node_id" in body, (
            f"INVARIANT VIOLATION: Discovery response missing 'node_id'. "
            f"Present keys: {list(body.keys()) if isinstance(body, dict) else body}"
        )

    def test_invariant_discovery_response_node_id_matches_agent(self):
        """Discovery response node_id must match the agent's node_id."""
        agent = _make_agent("exact-match-node")

        body = self._get_discovery_body(agent)

        assert body.get("node_id") == "exact-match-node", (
            f"INVARIANT VIOLATION: Discovery response node_id='{body.get('node_id')}' "
            "does not match agent node_id='exact-match-node'."
        )

    def test_invariant_discovery_response_contains_reasoners(self):
        """Discovery response must always contain 'reasoners'."""
        agent = _make_agent("discovery-reasoners-node")

        @agent.reasoner("probe")
        def probe() -> dict:
            return {}

        body = self._get_discovery_body(agent)

        assert "reasoners" in body, (
            f"INVARIANT VIOLATION: Discovery response missing 'reasoners'. "
            f"Present keys: {list(body.keys()) if isinstance(body, dict) else body}"
        )

    def test_invariant_discovery_response_reasoners_includes_registered_name(self):
        """Discovery response reasoners must include all registered reasoner names."""
        agent = _make_agent("discovery-list-node")

        @agent.reasoner("visible-reasoner")
        def visible_reasoner() -> dict:
            return {}

        body = self._get_discovery_body(agent)
        reasoners = body.get("reasoners", [])

        # Reasoners can be a list of dicts or a dict; normalize to set of IDs
        if isinstance(reasoners, dict):
            reasoner_ids = set(reasoners.keys())
        elif isinstance(reasoners, list):
            reasoner_ids = {
                r.get("id", r.get("name", r)) if isinstance(r, dict) else str(r)
                for r in reasoners
            }
        else:
            reasoner_ids = set()

        assert "visible-reasoner" in reasoner_ids, (
            f"INVARIANT VIOLATION: Registered reasoner 'visible-reasoner' not found "
            f"in discovery response. Reasoner IDs found: {reasoner_ids}"
        )

    def test_invariant_discovery_response_is_stable_across_calls(self):
        """
        Calling /discover twice must return responses with the same node_id
        and the same set of registered reasoners (no random variation).
        """
        agent = _make_agent("stable-discovery-node")

        @agent.reasoner("stable-fn")
        def stable_fn() -> dict:
            return {}

        body1 = self._get_discovery_body(agent)
        body2 = self._get_discovery_body(agent)

        assert body1.get("node_id") == body2.get("node_id"), (
            f"INVARIANT VIOLATION: Discovery response node_id changed between calls. "
            f"Call 1: '{body1.get('node_id')}', Call 2: '{body2.get('node_id')}'"
        )
