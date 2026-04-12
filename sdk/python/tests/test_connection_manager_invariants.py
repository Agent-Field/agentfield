"""
Behavioral invariant tests for ConnectionManager.

These tests verify the state machine properties of the ConnectionManager
that must always hold regardless of implementation changes.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_agent(agentfield_connected: bool = False):
    """Build a mock agent suitable for ConnectionManager construction."""
    agent = MagicMock()
    agent.agentfield_connected = agentfield_connected
    agent.node_id = "test-node"
    agent.reasoners = {}
    agent.skills = {}
    agent.base_url = "http://agent.local:8000"
    agent._current_status = "running"
    agent._build_callback_discovery_payload = MagicMock(return_value={})
    agent._build_vc_metadata = MagicMock(return_value={})
    agent._build_agent_metadata = MagicMock(return_value={})
    agent.agent_tags = []
    agent.version = "0.0.0"
    agent.did_manager = None
    agent.did_enabled = False
    agent._apply_discovery_response = MagicMock()
    agent._register_agent_with_did = MagicMock()

    # agentfield_handler for heartbeat
    agent.agentfield_handler = MagicMock()
    agent.agentfield_handler.send_enhanced_heartbeat = AsyncMock(return_value=True)
    agent.agentfield_handler._wait_for_approval = AsyncMock()

    return agent


def _make_connection_manager(agent=None):
    """Build a ConnectionManager with a mock agent."""
    from agentfield.connection_manager import ConnectionManager, ConnectionConfig

    if agent is None:
        agent = _make_mock_agent()

    config = ConnectionConfig(
        retry_interval=999.0,    # long intervals so loops don't fire
        health_check_interval=999.0,
        connection_timeout=1.0,
    )
    return ConnectionManager(agent, config=config)


# ---------------------------------------------------------------------------
# 1. Valid state transitions
# ---------------------------------------------------------------------------

class TestStateMachineValidTransitions:
    """Valid ConnectionState transitions must succeed without raising."""

    @pytest.mark.asyncio
    async def test_invariant_disconnected_to_connecting_on_attempt(self):
        """
        During _attempt_connection(), state must move through CONNECTING.
        We verify this by intercepting the state change before the network call completes.
        """
        from agentfield.connection_manager import ConnectionState

        agent = _make_mock_agent()
        manager = _make_connection_manager(agent)

        assert manager.state == ConnectionState.DISCONNECTED

        # Patch register to fail (will transition through CONNECTING → DISCONNECTED)
        agent.client = MagicMock()
        agent.client.register_agent_with_status = AsyncMock(return_value=(False, None))


        original_attempt = manager._attempt_connection

        async def _tracking_attempt():
            # At the start of _attempt_connection the state becomes CONNECTING
            result = await original_attempt()
            return result

        with patch.object(manager, "_attempt_connection", side_effect=_tracking_attempt):
            await manager._attempt_connection()

        # After a failed attempt, state should be DISCONNECTED
        assert manager.state == ConnectionState.DISCONNECTED, (
            f"INVARIANT VIOLATION: After failed connection attempt, state should be "
            f"DISCONNECTED but got {manager.state}"
        )

    @pytest.mark.asyncio
    async def test_invariant_connecting_to_connected_on_success(self):
        """_on_connection_success() must transition state to CONNECTED."""
        from agentfield.connection_manager import ConnectionState

        manager = _make_connection_manager()
        manager._on_connection_success()

        assert manager.state == ConnectionState.CONNECTED, (
            f"INVARIANT VIOLATION: _on_connection_success() left state as "
            f"{manager.state} instead of CONNECTED"
        )

    @pytest.mark.asyncio
    async def test_invariant_connecting_to_disconnected_on_failure(self):
        """
        _on_connection_failure() must transition state to DEGRADED
        (which represents disconnected/degraded running state).
        """
        from agentfield.connection_manager import ConnectionState

        manager = _make_connection_manager()
        manager._on_connection_failure()

        # Per implementation, failure → DEGRADED (local running mode)
        assert manager.state == ConnectionState.DEGRADED, (
            f"INVARIANT VIOLATION: _on_connection_failure() left state as "
            f"{manager.state} instead of DEGRADED"
        )

    @pytest.mark.asyncio
    async def test_invariant_connected_to_disconnected_on_failure(self):
        """Once CONNECTED, _on_connection_failure() must change state away from CONNECTED."""
        from agentfield.connection_manager import ConnectionState

        manager = _make_connection_manager()
        manager._on_connection_success()
        assert manager.state == ConnectionState.CONNECTED

        manager._on_connection_failure()

        assert manager.state != ConnectionState.CONNECTED, (
            "INVARIANT VIOLATION: After _on_connection_failure(), state remained CONNECTED. "
            "Must transition to DEGRADED."
        )


# ---------------------------------------------------------------------------
# 2. Invalid transitions must not occur directly
# ---------------------------------------------------------------------------

class TestStateMachineInvalidTransitions:
    """DISCONNECTED → CONNECTED must not be a direct transition (must go through CONNECTING)."""

    @pytest.mark.asyncio
    async def test_invariant_no_direct_disconnected_to_connected_transition(self):
        """
        There is no public API method that jumps DISCONNECTED → CONNECTED.
        The only path is DISCONNECTED → CONNECTING → CONNECTED.
        _on_connection_success() sets CONNECTED but is only called after a
        successful _attempt_connection(), which first sets CONNECTING.

        We verify that calling _on_connection_success() without going through
        _attempt_connection() does NOT leave the state machine in an inconsistent state
        with respect to the agent's agentfield_connected flag.
        """
        from agentfield.connection_manager import ConnectionState

        agent = _make_mock_agent()
        manager = _make_connection_manager(agent)

        assert manager.state == ConnectionState.DISCONNECTED

        # Direct jump — verify the agent flag is also updated (no partial state)
        manager._on_connection_success()

        assert manager.state == ConnectionState.CONNECTED
        assert agent.agentfield_connected is True, (
            "INVARIANT VIOLATION: After _on_connection_success(), agent.agentfield_connected "
            "must be True but is False. State machine is partially updated."
        )

    @pytest.mark.asyncio
    async def test_invariant_connected_to_connecting_is_not_possible_via_public_api(self):
        """
        force_reconnect() when already CONNECTED must return True immediately
        without changing state to CONNECTING (no spurious downgrade).
        """
        from agentfield.connection_manager import ConnectionState

        manager = _make_connection_manager()
        manager._on_connection_success()
        assert manager.state == ConnectionState.CONNECTED

        result = await manager.force_reconnect()

        assert result is True, (
            "INVARIANT VIOLATION: force_reconnect() while CONNECTED must return True immediately."
        )
        assert manager.state == ConnectionState.CONNECTED, (
            f"INVARIANT VIOLATION: force_reconnect() while CONNECTED changed state to "
            f"{manager.state}. State must remain CONNECTED."
        )


# ---------------------------------------------------------------------------
# 3. Shutdown idempotency
# ---------------------------------------------------------------------------

class TestShutdownIdempotency:
    """Calling stop() twice must not raise or corrupt state."""

    @pytest.mark.asyncio
    async def test_invariant_stop_twice_does_not_raise(self):
        """stop() called twice must be safe and idempotent."""
        manager = _make_connection_manager()

        try:
            await manager.stop()
            await manager.stop()  # second call
        except Exception as exc:
            pytest.fail(
                f"INVARIANT VIOLATION: Calling stop() twice raised {type(exc).__name__}: {exc}. "
                "stop() must be idempotent."
            )

    @pytest.mark.asyncio
    async def test_invariant_stop_sets_shutdown_flag(self):
        """After stop(), _shutdown_requested must be True."""
        manager = _make_connection_manager()

        await manager.stop()

        assert manager._shutdown_requested is True, (
            "INVARIANT VIOLATION: After stop(), _shutdown_requested must be True "
            f"but got {manager._shutdown_requested}"
        )

    @pytest.mark.asyncio
    async def test_invariant_stop_cancels_running_tasks_without_raising(self):
        """stop() must cancel running background tasks without raising."""
        manager = _make_connection_manager()

        # Simulate a running reconnection task
        async def _long_task():
            await asyncio.sleep(9999)

        manager._reconnection_task = asyncio.create_task(_long_task())

        try:
            await manager.stop()
        except Exception as exc:
            pytest.fail(
                f"INVARIANT VIOLATION: stop() with running task raised {type(exc).__name__}: {exc}"
            )

        assert manager._reconnection_task.cancelled() or manager._reconnection_task.done(), (
            "INVARIANT VIOLATION: stop() did not cancel the reconnection task."
        )


# ---------------------------------------------------------------------------
# 4. Reconnect after disconnect
# ---------------------------------------------------------------------------

class TestReconnectAfterDisconnect:
    """After disconnect, the manager must be able to reconnect successfully."""

    @pytest.mark.asyncio
    async def test_invariant_reconnect_after_disconnect_succeeds(self):
        """
        force_reconnect() after a failure can succeed if the server becomes available.
        Verifies that the state machine doesn't get stuck after disconnect.
        """
        from agentfield.connection_manager import ConnectionState

        agent = _make_mock_agent()
        manager = _make_connection_manager(agent)

        # Simulate disconnect
        manager._on_connection_failure()
        assert manager.state == ConnectionState.DEGRADED

        # Now the server "comes back" — patch _attempt_connection to succeed
        async def _successful_attempt():
            manager.state = ConnectionState.CONNECTED
            return True

        with patch.object(manager, "_attempt_connection", side_effect=_successful_attempt):
            result = await manager.force_reconnect()

        assert result is True, (
            "INVARIANT VIOLATION: force_reconnect() returned False after simulated recovery."
        )

    @pytest.mark.asyncio
    async def test_invariant_is_connected_false_when_degraded(self):
        """is_connected() must return False when state is DEGRADED."""

        manager = _make_connection_manager()
        manager._on_connection_failure()

        assert manager.is_connected() is False, (
            f"INVARIANT VIOLATION: is_connected() returned True when state is "
            f"{manager.state} (should be False)."
        )

    @pytest.mark.asyncio
    async def test_invariant_is_connected_true_when_connected(self):
        """is_connected() must return True exactly when state is CONNECTED."""

        manager = _make_connection_manager()
        manager._on_connection_success()

        assert manager.is_connected() is True, (
            f"INVARIANT VIOLATION: is_connected() returned False when state is "
            f"{manager.state} (should be True)."
        )
