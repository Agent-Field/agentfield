"""
Tests for Agent.handle_serverless — serverless invocation path.
"""
from __future__ import annotations

from unittest.mock import patch


from agentfield.agent import Agent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent() -> Agent:
    """Construct a minimal Agent without network side-effects."""
    with patch("agentfield.agent._detect_container_ip", return_value=None), \
         patch("agentfield.agent._detect_local_ip", return_value="127.0.0.1"), \
         patch("agentfield.agent._is_running_in_container", return_value=False):
        agent = Agent(
            node_id="test-agent",
            agentfield_server="http://localhost:8080",
            dev_mode=True,
            callback_url="http://localhost:8001",
            auto_register=False,
        )
    return agent


# ---------------------------------------------------------------------------
# Discovery path
# ---------------------------------------------------------------------------


class TestHandleServerlessDiscovery:
    def test_discover_path(self):
        agent = _make_agent()
        result = agent.handle_serverless({"path": "/discover"})
        # Should return agent metadata (dict)
        assert isinstance(result, dict)

    def test_discover_via_action(self):
        agent = _make_agent()
        result = agent.handle_serverless({"action": "discover"})
        assert isinstance(result, dict)

    def test_discover_path_prefix(self):
        agent = _make_agent()
        result = agent.handle_serverless({"path": "/api/v1/discover"})
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# Missing reasoner
# ---------------------------------------------------------------------------


class TestHandleServerlessMissingReasoner:
    def test_missing_reasoner_key_returns_400(self):
        agent = _make_agent()
        result = agent.handle_serverless({"input": {"x": 1}})
        assert result["statusCode"] == 400
        assert "error" in result["body"]

    def test_missing_reasoner_with_empty_event(self):
        agent = _make_agent()
        result = agent.handle_serverless({})
        assert result["statusCode"] == 400

    def test_unknown_function_returns_404(self):
        agent = _make_agent()
        result = agent.handle_serverless({"reasoner": "nonexistent_fn", "input": {}})
        assert result["statusCode"] == 404
        assert "not found" in result["body"]["error"]


# ---------------------------------------------------------------------------
# Adapter path
# ---------------------------------------------------------------------------


class TestHandleServerlessAdapter:
    def test_adapter_transforms_event(self):
        agent = _make_agent()

        def adapter(event):
            return {"path": "/discover"}

        result = agent.handle_serverless({"raw": "data"}, adapter=adapter)
        assert isinstance(result, dict)

    def test_adapter_returning_none_uses_original(self):
        agent = _make_agent()

        def adapter(event):
            return None  # return None → use original event

        result = agent.handle_serverless({"path": "/discover"}, adapter=adapter)
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# Basic invocation with a sync reasoner
# ---------------------------------------------------------------------------


class TestHandleServerlessBasicInvocation:
    def test_sync_function_invocation(self):
        agent = _make_agent()

        def greet(name: str) -> dict:
            return {"greeting": f"hello {name}"}

        # Patch getattr to find our function
        with patch.object(agent, "greet", greet, create=True):
            result = agent.handle_serverless({
                "reasoner": "greet",
                "input": {"name": "world"},
            })

        assert result["statusCode"] == 200
        assert result["body"]["greeting"] == "hello world"

    def test_async_function_invocation(self):
        agent = _make_agent()

        async def async_greet(name: str) -> dict:
            return {"greeting": f"async hello {name}"}

        with patch.object(agent, "async_greet", async_greet, create=True):
            result = agent.handle_serverless({
                "reasoner": "async_greet",
                "input": {"name": "world"},
            })

        assert result["statusCode"] == 200
        assert result["body"]["greeting"] == "async hello world"

    def test_function_exception_returns_500(self):
        agent = _make_agent()

        def bad_fn() -> dict:
            raise ValueError("something went wrong")

        with patch.object(agent, "bad_fn", bad_fn, create=True):
            result = agent.handle_serverless({
                "reasoner": "bad_fn",
                "input": {},
            })

        assert result["statusCode"] == 500
        assert "something went wrong" in result["body"]["error"]


# ---------------------------------------------------------------------------
# Event format variations
# ---------------------------------------------------------------------------


class TestHandleServerlessEventFormats:
    def test_target_key_as_alias_for_reasoner(self):
        agent = _make_agent()

        def echo(msg: str) -> dict:
            return {"msg": msg}

        with patch.object(agent, "echo", echo, create=True):
            result = agent.handle_serverless({
                "target": "echo",
                "input": {"msg": "hi"},
            })

        assert result["statusCode"] == 200

    def test_skill_key_as_alias_for_reasoner(self):
        agent = _make_agent()

        def echo(msg: str) -> dict:
            return {"msg": msg}

        with patch.object(agent, "echo", echo, create=True):
            result = agent.handle_serverless({
                "skill": "echo",
                "input": {"msg": "test"},
            })

        assert result["statusCode"] == 200

    def test_path_extraction_execute_prefix(self):
        agent = _make_agent()

        def echo(msg: str) -> dict:
            return {"msg": msg}

        with patch.object(agent, "echo", echo, create=True):
            result = agent.handle_serverless({
                "path": "/execute/echo",
                "input": {"msg": "via-path"},
            })

        assert result["statusCode"] == 200

    def test_execution_context_from_event(self):
        agent = _make_agent()

        def noop() -> dict:
            return {}

        with patch.object(agent, "noop", noop, create=True):
            result = agent.handle_serverless({
                "reasoner": "noop",
                "input": {},
                "execution_context": {
                    "execution_id": "custom-exec-id",
                    "run_id": "custom-run-id",
                },
            })

        assert result["statusCode"] == 200


# ---------------------------------------------------------------------------
# Execution context cleanup
# ---------------------------------------------------------------------------


class TestHandleServerlessContextCleanup:
    def test_execution_context_cleared_after_success(self):
        agent = _make_agent()

        def noop() -> dict:
            return {}

        with patch.object(agent, "noop", noop, create=True):
            agent.handle_serverless({"reasoner": "noop", "input": {}})

        assert agent._current_execution_context is None

    def test_execution_context_cleared_after_exception(self):
        agent = _make_agent()

        def fail():
            raise RuntimeError("boom")

        with patch.object(agent, "fail", fail, create=True):
            agent.handle_serverless({"reasoner": "fail", "input": {}})

        assert agent._current_execution_context is None

    def test_async_mode_disabled_for_serverless(self):
        agent = _make_agent()
        agent.async_config.enable_async_execution = True

        def noop() -> dict:
            return {}

        with patch.object(agent, "noop", noop, create=True):
            agent.handle_serverless({"reasoner": "noop", "input": {}})

        assert agent.async_config.enable_async_execution is False

    def test_async_mode_was_previously_true(self):
        """Verify async mode is actively CHANGED, not just coincidentally False."""
        agent = _make_agent()
        # Explicitly enable before calling serverless
        agent.async_config.enable_async_execution = True
        assert agent.async_config.enable_async_execution is True  # precondition

        def noop() -> dict:
            return {}

        with patch.object(agent, "noop", noop, create=True):
            agent.handle_serverless({"reasoner": "noop", "input": {}})

        # Must have been flipped from True to False
        assert agent.async_config.enable_async_execution is False


class TestServerlessRealRegistration:
    """Tests that exercise the REAL reasoner registration mechanism,
    not just patch.object. This catches regressions if the dispatch
    path changes from getattr to a registry dict."""

    def test_registered_reasoner_invoked_via_serverless(self):
        """Register a function the real way and invoke through handle_serverless."""
        agent = _make_agent()

        # Use the actual Agent.reasoner decorator to register
        @agent.reasoner("real_greet")
        def real_greet(name: str = "world") -> dict:
            return {"hello": name}

        result = agent.handle_serverless({
            "reasoner": "real_greet",
            "input": {"name": "test"},
        })

        assert result["statusCode"] == 200
        assert result["body"]["hello"] == "test"

    def test_registered_reasoner_not_found_returns_404(self):
        """Verify unregistered name returns 404 even when other reasoners exist."""
        agent = _make_agent()

        @agent.reasoner("exists")
        def exists() -> dict:
            return {}

        result = agent.handle_serverless({
            "reasoner": "does_not_exist",
            "input": {},
        })

        assert result["statusCode"] == 404
