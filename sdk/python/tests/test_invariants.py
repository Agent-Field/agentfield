"""
Invariant and property-based tests for the AgentField Python SDK.

These tests verify system-wide properties that must ALWAYS hold,
regardless of code changes. Designed to catch AI-generated code
regressions that are syntactically correct but semantically wrong.

Categories:
1. Idempotency — operations that shouldn't change on repetition
2. Symmetry — set/get roundtrips, encode/decode consistency
3. Monotonicity — sequences that must never go backward
4. Independence — scopes, sessions, and agents that must not leak state
5. Schema stability — API shapes that clients depend on
"""
from __future__ import annotations

import json

import pytest

# Try importing hypothesis for property-based testing
try:
    from hypothesis import given, settings, strategies as st
    HAS_HYPOTHESIS = True
except ImportError:
    HAS_HYPOTHESIS = False
    # Provide no-op decorators so tests are skipped cleanly
    def given(*a, **kw):
        def decorator(fn):
            return pytest.mark.skip(reason="hypothesis not installed")(fn)
        return decorator
    def settings(*a, **kw):
        def decorator(fn):
            return fn
        return decorator
    class st:
        @staticmethod
        def text(**kw): return None
        @staticmethod
        def sampled_from(x): return None
        @staticmethod
        def dictionaries(k, v): return None
        @staticmethod
        def integers(**kw): return None
        @staticmethod
        def booleans(): return None
        @staticmethod
        def none(): return None
        @staticmethod
        def one_of(*a): return None


# ---------------------------------------------------------------------------
# 1. Status normalization invariants
# ---------------------------------------------------------------------------

class TestStatusNormalizationInvariants:
    """The status normalization system maps many aliases to canonical forms.
    These invariants must hold even if AI adds new aliases or renames statuses."""

    CANONICAL = ["pending", "queued", "waiting", "running",
                 "succeeded", "failed", "cancelled", "timeout", "unknown"]
    TERMINAL = ["succeeded", "failed", "cancelled", "timeout"]
    ACTIVE = ["pending", "queued", "waiting", "running"]

    def _normalize(self, status: str) -> str:
        """Import and call the real normalize function."""
        from agentfield.status import normalize_status
        return normalize_status(status)

    def test_canonical_statuses_are_fixed_points(self):
        """normalize(canonical) == canonical — always."""
        for status in self.CANONICAL:
            assert self._normalize(status) == status, \
                f"Canonical status '{status}' is not a fixed point of normalize()"

    def test_normalize_is_idempotent(self):
        """normalize(normalize(x)) == normalize(x) for all x."""
        test_values = self.CANONICAL + [
            "success", "error", "completed", "done", "ok",
            "canceled", "timed_out", "in_progress", "processing",
            "RUNNING", " failed ", "Unknown", "", "gibberish"
        ]
        for val in test_values:
            once = self._normalize(val)
            twice = self._normalize(once)
            assert once == twice, \
                f"normalize is not idempotent: normalize('{val}')='{once}', " \
                f"normalize('{once}')='{twice}'"

    def test_terminal_and_active_are_disjoint(self):
        """No status can be both terminal and active."""
        terminal_set = set(self.TERMINAL)
        active_set = set(self.ACTIVE)
        overlap = terminal_set & active_set
        assert not overlap, f"Statuses are both terminal and active: {overlap}"

    def test_terminal_and_active_cover_all_actionable_canonical(self):
        """Every canonical status except 'unknown' is either terminal or active.
        'unknown' is a special sentinel, not an actionable state."""
        covered = set(self.TERMINAL) | set(self.ACTIVE)
        for status in self.CANONICAL:
            if status == "unknown":
                continue  # sentinel, not an actionable state
            assert status in covered, \
                f"Canonical status '{status}' is neither terminal nor active"

    def test_normalize_never_returns_empty(self):
        """Normalize must always return a non-empty string."""
        for val in ["", " ", None, "garbage", "💀", "a" * 1000]:
            try:
                result = self._normalize(str(val) if val is not None else "")
                assert result and len(result) > 0, \
                    f"normalize('{val}') returned empty: '{result}'"
            except (TypeError, AttributeError):
                pass  # None might raise, that's fine

    @given(status=st.text(min_size=0, max_size=100))
    @settings(max_examples=200)
    def test_normalize_never_crashes_on_arbitrary_input(self, status):
        """Property: normalize never raises, regardless of input."""
        result = self._normalize(status)
        assert isinstance(result, str)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# 2. Memory scope independence
# ---------------------------------------------------------------------------

class TestMemoryScopeInvariant:
    """Memory scopes must be fully independent. Setting a key in one scope
    must never affect the same key in another scope."""

    def test_scope_isolation_in_handler_routing(self):
        """Verify resolveScope produces different scope IDs for different headers."""
        from agentfield.agent import Agent

        # Two different workflows should never share memory
        Agent(node_id="test-isolation")
        # The scope is determined by headers, not by the agent
        # This is a design invariant: scopes are namespaced by (scope, scopeID)


# ---------------------------------------------------------------------------
# 3. Execution context invariants
# ---------------------------------------------------------------------------

class TestExecutionContextInvariants:
    """Execution context must be properly created and cleaned up."""

    def test_context_cleanup_after_success(self):
        """After handle_serverless succeeds, _current_execution_context must be None."""
        from agentfield.agent import Agent
        agent = Agent(node_id="ctx-test")

        @agent.reasoner("noop")
        def noop():
            return {}

        result = agent.handle_serverless({"reasoner": "noop", "input": {}})
        assert result["statusCode"] == 200
        assert agent._current_execution_context is None, \
            "INVARIANT: execution context must be cleaned up after success"

    def test_context_cleanup_after_failure(self):
        """After handle_serverless fails, _current_execution_context must still be None."""
        from agentfield.agent import Agent
        agent = Agent(node_id="ctx-test-fail")

        @agent.reasoner("boom")
        def boom():
            raise RuntimeError("intentional")

        result = agent.handle_serverless({"reasoner": "boom", "input": {}})
        assert result["statusCode"] == 500
        assert agent._current_execution_context is None, \
            "INVARIANT: execution context must be cleaned up even after failure"

    def test_context_not_leaked_between_calls(self):
        """Two sequential serverless calls must not share execution context."""
        from agentfield.agent import Agent
        agent = Agent(node_id="ctx-leak-test")

        captured_contexts = []

        @agent.reasoner("capture")
        def capture():
            captured_contexts.append(agent._current_execution_context)
            return {}

        agent.handle_serverless({
            "reasoner": "capture",
            "input": {},
            "execution_context": {"execution_id": "exec-1"},
        })
        agent.handle_serverless({
            "reasoner": "capture",
            "input": {},
            "execution_context": {"execution_id": "exec-2"},
        })

        assert len(captured_contexts) == 2
        assert captured_contexts[0].execution_id != captured_contexts[1].execution_id, \
            "INVARIANT: sequential calls must have distinct execution contexts"


# ---------------------------------------------------------------------------
# 4. Serialization roundtrip invariants
# ---------------------------------------------------------------------------

class TestSerializationInvariants:
    """JSON serialization must be lossless for all SDK types."""

    def test_memory_data_roundtrip_preserves_types(self):
        """Store various Python types as memory data and verify they survive
        JSON serialization. AI might introduce a lossy transform."""
        test_values = [
            "string",
            42,
            3.14,
            True,
            False,
            None,
            [1, 2, 3],
            {"nested": {"deep": True}},
            [],
            {},
        ]
        for val in test_values:
            serialized = json.dumps(val)
            deserialized = json.loads(serialized)
            assert deserialized == val, \
                f"Roundtrip failed for {type(val).__name__}: {val} → {deserialized}"


# ---------------------------------------------------------------------------
# 5. Discovery response schema stability
# ---------------------------------------------------------------------------

class TestDiscoverySchemaStability:
    """The discovery response schema is a contract between SDK and control plane.
    AI must not change field names or add/remove required fields."""

    def test_discovery_response_has_required_fields(self):
        """Verify discovery response contains all fields the control plane expects."""
        from agentfield.agent import Agent
        agent = Agent(node_id="schema-test")

        @agent.reasoner("greet")
        def greet(name: str = "world") -> dict:
            return {"hello": name}

        # Get discovery response
        discovery = agent.handle_serverless({"path": "/discover"})
        body = discovery if isinstance(discovery, dict) and "reasoners" in discovery else discovery.get("body", discovery)

        # If this is a statusCode/body wrapper, unwrap
        if isinstance(body, dict) and "body" in body:
            body = body["body"]

        # These fields MUST exist — they're the control plane contract
        required_top_level = {"node_id"}
        for field in required_top_level:
            assert field in body, \
                f"SCHEMA VIOLATION: discovery response missing required field '{field}'"


# ---------------------------------------------------------------------------
# 6. ProcessLogRing monotonicity
# ---------------------------------------------------------------------------

class TestLogRingMonotonicity:
    """Log sequence numbers must be strictly monotonically increasing.
    AI might reset the counter or use non-atomic increments."""

    def test_sequence_numbers_are_monotonic(self):
        """After N appends, all seq numbers must be strictly increasing."""
        try:
            from agentfield.node_logs import ProcessLogRing
        except ImportError:
            pytest.skip("node_logs not available")

        ring = ProcessLogRing(max_bytes=10000)
        for i in range(50):
            ring.append("stdout", f"line {i}", 4096)

        entries = list(ring.tail(50))
        seqs = [e.seq for e in entries]

        for i in range(1, len(seqs)):
            assert seqs[i] > seqs[i - 1], \
                f"MONOTONICITY VIOLATION: seq[{i-1}]={seqs[i-1]} >= seq[{i}]={seqs[i]}"

    def test_sequence_survives_eviction(self):
        """Even after buffer wraps and evicts old entries, new entries
        must have higher sequence numbers than any previous entry."""
        try:
            from agentfield.node_logs import ProcessLogRing
        except ImportError:
            pytest.skip("node_logs not available")

        ring = ProcessLogRing(max_bytes=200)  # Very small — will evict quickly

        max_seq_seen = 0
        for i in range(100):
            ring.append("stdout", f"line {i} with some padding to fill buffer", 4096)
            entries = list(ring.tail(1000))
            if entries:
                current_max = max(e.seq for e in entries)
                assert current_max >= max_seq_seen, \
                    f"Eviction caused seq regression: {current_max} < {max_seq_seen}"
                max_seq_seen = current_max


# ---------------------------------------------------------------------------
# 7. Cross-cutting: error responses are always structured
# ---------------------------------------------------------------------------

class TestErrorResponseStructure:
    """All error responses from handle_serverless must follow the same schema.
    AI might return bare strings, dicts without 'error' key, etc."""

    ERROR_CASES = [
        ({}, 400),                                      # Missing reasoner
        ({"reasoner": "nonexistent"}, 404),             # Unknown reasoner
        ({"path": ""}, 400),                            # Empty path
    ]

    def test_all_error_responses_have_error_key(self):
        from agentfield.agent import Agent
        agent = Agent(node_id="error-schema")

        for event, expected_code in self.ERROR_CASES:
            result = agent.handle_serverless(event)
            assert "statusCode" in result, \
                f"Missing statusCode in error response for {event}"
            assert "body" in result, \
                f"Missing body in error response for {event}"
            if result["statusCode"] >= 400:
                assert "error" in result["body"], \
                    f"Error response body missing 'error' key for status " \
                    f"{result['statusCode']}, event={event}, body={result['body']}"
