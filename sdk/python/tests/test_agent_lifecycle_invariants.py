"""
Behavioral invariant tests for Agent lifecycle: registration, reasoner persistence,
node_id immutability, and discovery response stability.

Confirmed behavioral facts:
- agent._reasoner_registry: Dict[str, ReasonerEntry] keyed by function __name__
- agent.reasoners: List[Dict] where each dict has 'id' = function __name__
- handle_serverless({'reasoner': name}) uses function __name__ for lookup
- Discovery via {'path': '/discover'} returns flat dict: node_id, version, reasoners, skills
- Each reasoner dict in discovery list has 'id' = function.__name__
"""
from __future__ import annotations


def _make_agent(node_id: str = "test-lifecycle-node"):
    from agentfield.agent import Agent
    return Agent(node_id=node_id, agentfield_server="http://localhost:8080")


def _registry_ids(agent) -> set:
    return set(agent._reasoner_registry.keys())


def _discovery_reasoner_ids(agent) -> set:
    result = agent.handle_serverless({"path": "/discover"})
    reasoners = result.get("reasoners", [])
    if isinstance(reasoners, list):
        return {r.get("id") for r in reasoners if isinstance(r, dict) and "id" in r}
    if isinstance(reasoners, dict):
        return set(reasoners.keys())
    return set()


class TestReasonerRegistrationPersistence:
    """After @agent.reasoner(...), the reasoner must remain registered and callable."""

    def test_invariant_registered_reasoner_appears_in_internal_registry(self):
        agent = _make_agent("persist-test")

        @agent.reasoner("any-label")
        def my_registered_fn(x: int = 1) -> dict:
            return {"result": x}

        assert "my_registered_fn" in _registry_ids(agent), (
            f"INVARIANT VIOLATION: 'my_registered_fn' not in registry. "
            f"Keys: {_registry_ids(agent)}"
        )

    def test_invariant_registered_reasoner_callable_via_handle_serverless(self):
        agent = _make_agent("callable-test")

        @agent.reasoner("greet")
        def greet(name: str = "world") -> dict:
            return {"hello": name}

        result = agent.handle_serverless({"reasoner": "greet", "input": {}})
        assert result["statusCode"] == 200, (
            f"INVARIANT VIOLATION: 'greet' returned {result['statusCode']} not 200. "
            f"Body: {result.get('body')}"
        )

    def test_invariant_multiple_reasoners_all_in_registry(self):
        agent = _make_agent("multi-reasoner-test")

        @agent.reasoner("a")
        def r_alpha() -> dict:
            return {}

        @agent.reasoner("b")
        def r_beta() -> dict:
            return {}

        @agent.reasoner("c")
        def r_gamma() -> dict:
            return {}

        registered = _registry_ids(agent)
        for name in ["r_alpha", "r_beta", "r_gamma"]:
            assert name in registered, (
                f"INVARIANT VIOLATION: '{name}' not in registry. Present: {registered}"
            )


class TestDoubleRegistrationReplaces:
    """Registering a function with the same __name__ twice replaces the first."""

    def test_invariant_second_registration_wins(self):
        agent = _make_agent("replace-test")

        @agent.reasoner("first")
        def foo() -> dict:
            return {"version": 1}

        @agent.reasoner("second")
        def foo() -> dict:  # noqa: F811
            return {"version": 2}

        result = agent.handle_serverless({"reasoner": "foo", "input": {}})
        assert result["statusCode"] == 200, (
            f"INVARIANT VIOLATION: Double-registered 'foo' returned {result['statusCode']}."
        )
        body = result.get("body", {})
        assert body.get("version") == 2, (
            f"INVARIANT VIOLATION: Second registration did not win. Got: {body}"
        )

    def test_invariant_only_one_entry_per_name_in_registry(self):
        agent = _make_agent("dedup-test")

        @agent.reasoner("v1")
        def deduplicated() -> dict:
            return {}

        @agent.reasoner("v2")
        def deduplicated() -> dict:  # noqa: F811
            return {}

        count = sum(1 for k in agent._reasoner_registry if k == "deduplicated")
        assert count == 1, (
            f"INVARIANT VIOLATION: 'deduplicated' appears {count} times in registry."
        )


class TestUnregisteredReasonerReturns404:
    """Calling a non-existent reasoner through handle_serverless must return 404."""

    def test_invariant_unknown_reasoner_returns_404(self):
        agent = _make_agent("404-test")
        result = agent.handle_serverless({"reasoner": "does_not_exist", "input": {}})
        assert result["statusCode"] == 404, (
            f"INVARIANT VIOLATION: Unknown reasoner returned {result['statusCode']} not 404."
        )

    def test_invariant_unknown_reasoner_body_contains_error(self):
        agent = _make_agent("404-body-test")
        result = agent.handle_serverless({"reasoner": "ghost_reasoner", "input": {}})
        body = result.get("body", {})
        assert "error" in body, (
            f"INVARIANT VIOLATION: 404 body missing 'error' key. Body: {body}"
        )

    def test_invariant_empty_registry_returns_404(self):
        agent = _make_agent("empty-registry-test")
        result = agent.handle_serverless({"reasoner": "anything", "input": {}})
        assert result["statusCode"] == 404, (
            f"INVARIANT VIOLATION: Empty registry call returned {result['statusCode']} not 404."
        )


class TestNodeIdImmutability:
    """After creation, node_id must always return the same value."""

    def test_invariant_node_id_stable_on_repeated_access(self):
        agent = _make_agent("my-immutable-node")
        ids = [agent.node_id for _ in range(10)]
        assert len(set(ids)) == 1, (
            f"INVARIANT VIOLATION: node_id changed across accesses: {ids}"
        )
        assert ids[0] == "my-immutable-node"

    def test_invariant_node_id_equals_constructor_argument(self):
        test_id = "constructed-id-xyz"
        agent = _make_agent(test_id)
        assert agent.node_id == test_id, (
            f"INVARIANT VIOLATION: node_id='{agent.node_id}' != '{test_id}'"
        )

    def test_invariant_node_id_stable_after_reasoner_registration(self):
        agent = _make_agent("stable-node")
        original_id = agent.node_id

        @agent.reasoner("some")
        def some_reasoner() -> dict:
            return {}

        assert agent.node_id == original_id, (
            f"INVARIANT VIOLATION: node_id changed from '{original_id}' to '{agent.node_id}'"
        )

    def test_invariant_node_id_stable_after_handle_serverless(self):
        agent = _make_agent("stable-exec")
        original_id = agent.node_id

        @agent.reasoner("probe")
        def probe() -> dict:
            return {}

        agent.handle_serverless({"reasoner": "probe", "input": {}})
        assert agent.node_id == original_id, (
            "INVARIANT VIOLATION: node_id changed after handle_serverless."
        )


class TestDiscoveryResponseStability:
    """Discovery response must always contain 'node_id' and 'reasoners'."""

    def test_invariant_discovery_contains_node_id(self):
        agent = _make_agent("discovery-node")
        result = agent.handle_serverless({"path": "/discover"})
        assert "node_id" in result, (
            f"INVARIANT VIOLATION: Discovery missing 'node_id'. Keys: {list(result.keys())}"
        )

    def test_invariant_discovery_node_id_matches_agent(self):
        agent = _make_agent("exact-match-node")
        result = agent.handle_serverless({"path": "/discover"})
        assert result.get("node_id") == "exact-match-node", (
            f"INVARIANT VIOLATION: Discovery node_id='{result.get('node_id')}' != 'exact-match-node'"
        )

    def test_invariant_discovery_contains_reasoners(self):
        agent = _make_agent("discovery-reasoners-node")

        @agent.reasoner("probe")
        def probe_fn() -> dict:
            return {}

        result = agent.handle_serverless({"path": "/discover"})
        assert "reasoners" in result, (
            f"INVARIANT VIOLATION: Discovery missing 'reasoners'. Keys: {list(result.keys())}"
        )

    def test_invariant_discovery_reasoners_includes_registered_function_name(self):
        agent = _make_agent("discovery-list-node")

        @agent.reasoner("some-label")
        def visible_reasoner_fn() -> dict:
            return {}

        reasoner_ids = _discovery_reasoner_ids(agent)
        assert "visible_reasoner_fn" in reasoner_ids, (
            f"INVARIANT VIOLATION: 'visible_reasoner_fn' not in discovery. "
            f"Found: {reasoner_ids}"
        )

    def test_invariant_discovery_stable_across_calls(self):
        agent = _make_agent("stable-discovery-node")

        @agent.reasoner("stable")
        def stable_fn() -> dict:
            return {}

        result1 = agent.handle_serverless({"path": "/discover"})
        result2 = agent.handle_serverless({"path": "/discover"})

        assert result1.get("node_id") == result2.get("node_id"), (
            "INVARIANT VIOLATION: Discovery node_id changed between calls."
        )
        ids1 = {r.get("id") for r in result1.get("reasoners", []) if isinstance(r, dict)}
        ids2 = {r.get("id") for r in result2.get("reasoners", []) if isinstance(r, dict)}
        assert ids1 == ids2, (
            f"INVARIANT VIOLATION: Discovery reasoner IDs changed: {ids1} vs {ids2}"
        )
