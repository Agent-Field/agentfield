import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

from agentfield.connection_manager import (
    ConnectionConfig,
    ConnectionManager,
    ConnectionState,
)

# Test Fixtures


@pytest.fixture
def mock_agent():
    """Create a mock agent for testing."""
    agent = MagicMock()
    agent.node_id = "test-agent"
    agent.reasoners = []
    agent.skills = []
    agent.base_url = "http://localhost:9000"
    agent._current_status = "ready"
    agent.did_manager = None
    agent.did_enabled = False
    agent.agentfield_connected = False

    agent._build_callback_discovery_payload = Mock(return_value={"callbacks": []})
    agent._build_vc_metadata = Mock(return_value={"agent_default": True})
    agent._apply_discovery_response = Mock()
    agent._register_agent_with_did = Mock()

    # Default client mock - fails by default
    agent.client = MagicMock()
    agent.client.register_agent_with_status = AsyncMock(return_value=(False, None))

    # Default handler mock - heartbeat succeeds by default
    agent.agentfield_handler = MagicMock()
    agent.agentfield_handler.send_enhanced_heartbeat = AsyncMock(return_value=True)

    return agent


@pytest.fixture
def fast_config():
    """Create fast config for quick tests."""
    return ConnectionConfig(
        retry_interval=0.01,
        health_check_interval=0.01,
        connection_timeout=0.1,
    )


# ConnectionState Tests


@pytest.mark.unit
class TestConnectionState:
    """Tests for ConnectionState enum."""

    def test_all_states_exist(self):
        """Test that all expected states are defined."""
        assert ConnectionState.DISCONNECTED.value == "disconnected"
        assert ConnectionState.CONNECTING.value == "connecting"
        assert ConnectionState.CONNECTED.value == "connected"
        assert ConnectionState.RECONNECTING.value == "reconnecting"
        assert ConnectionState.DEGRADED.value == "degraded"


# ConnectionConfig Tests


@pytest.mark.unit
class TestConnectionConfig:
    """Tests for ConnectionConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ConnectionConfig()
        assert config.retry_interval == 10.0
        assert config.health_check_interval == 30.0
        assert config.connection_timeout == 10.0

    def test_custom_values(self):
        """Test custom configuration values."""
        config = ConnectionConfig(
            retry_interval=5.0,
            health_check_interval=15.0,
            connection_timeout=5.0,
        )
        assert config.retry_interval == 5.0
        assert config.health_check_interval == 15.0
        assert config.connection_timeout == 5.0


# ConnectionManager Initialization Tests


@pytest.mark.unit
class TestConnectionManagerInit:
    """Tests for ConnectionManager initialization."""

    def test_init_with_defaults(self, mock_agent):
        """Test initialization with default config."""
        manager = ConnectionManager(mock_agent)

        assert manager.agent is mock_agent
        assert manager.config is not None
        assert manager.config.retry_interval == 10.0
        assert manager.state == ConnectionState.DISCONNECTED
        assert manager.last_successful_connection is None
        assert manager._reconnection_task is None
        assert manager._health_check_task is None
        assert manager._shutdown_requested is False
        assert manager.on_connected is None
        assert manager.on_disconnected is None
        assert manager.on_degraded is None

    def test_init_with_custom_config(self, mock_agent, fast_config):
        """Test initialization with custom config."""
        manager = ConnectionManager(mock_agent, fast_config)

        assert manager.config is fast_config
        assert manager.config.retry_interval == 0.01


# ConnectionManager.start() Tests


@pytest.mark.unit
class TestConnectionManagerStart:
    """Tests for ConnectionManager.start() method."""

    @pytest.mark.asyncio
    async def test_start_success_connects(self, mock_agent, fast_config):
        """Test that successful start connects and starts health check."""
        mock_agent.client.register_agent_with_status = AsyncMock(
            return_value=(True, {"key": "value"})
        )
        manager = ConnectionManager(mock_agent, fast_config)

        result = await manager.start()

        assert result is True
        assert manager.state == ConnectionState.CONNECTED
        assert manager._health_check_task is not None
        assert manager._reconnection_task is None
        assert mock_agent.agentfield_connected is True
        mock_agent._apply_discovery_response.assert_called_once_with({"key": "value"})

        await manager.stop()

    @pytest.mark.asyncio
    async def test_start_failure_enters_degraded_mode(self, mock_agent, fast_config):
        """Test that failed start enters degraded mode and starts reconnection."""
        mock_agent.client.register_agent_with_status = AsyncMock(
            return_value=(False, None)
        )
        manager = ConnectionManager(mock_agent, fast_config)

        result = await manager.start()

        assert result is False
        assert manager.state in (ConnectionState.DEGRADED, ConnectionState.RECONNECTING)
        assert mock_agent.agentfield_connected is False

        await manager.stop()

    @pytest.mark.asyncio
    async def test_start_calls_client_register(self, mock_agent, fast_config):
        """Test that start calls client registration with correct args."""
        mock_agent.client.register_agent_with_status = AsyncMock(
            return_value=(True, None)
        )
        manager = ConnectionManager(mock_agent, fast_config)

        await manager.start()

        mock_agent.client.register_agent_with_status.assert_called_once()
        call_kwargs = mock_agent.client.register_agent_with_status.call_args.kwargs
        assert call_kwargs["node_id"] == "test-agent"
        assert call_kwargs["base_url"] == "http://localhost:9000"
        assert call_kwargs["suppress_errors"] is True

        await manager.stop()


# ConnectionManager.stop() Tests


@pytest.mark.unit
class TestConnectionManagerStop:
    """Tests for ConnectionManager.stop() method."""

    @pytest.mark.asyncio
    async def test_stop_cancels_reconnection_task(self, mock_agent, fast_config):
        """Test that stop cancels reconnection task."""
        mock_agent.client.register_agent_with_status = AsyncMock(
            return_value=(False, None)
        )
        manager = ConnectionManager(mock_agent, fast_config)

        await manager.start()
        await asyncio.sleep(0.01)

        assert manager._reconnection_task is not None

        await manager.stop()

        assert manager._shutdown_requested is True

    @pytest.mark.asyncio
    async def test_stop_cancels_health_check_task(self, mock_agent, fast_config):
        """Test that stop cancels health check task."""
        mock_agent.client.register_agent_with_status = AsyncMock(
            return_value=(True, None)
        )
        manager = ConnectionManager(mock_agent, fast_config)

        await manager.start()

        assert manager._health_check_task is not None

        await manager.stop()

        assert manager._shutdown_requested is True

    @pytest.mark.asyncio
    async def test_stop_without_start(self, mock_agent):
        """Test that stop works even without start."""
        manager = ConnectionManager(mock_agent)

        await manager.stop()

        assert manager._shutdown_requested is True


# ConnectionManager._attempt_connection() Tests


@pytest.mark.unit
class TestAttemptConnection:
    """Tests for _attempt_connection method."""

    @pytest.mark.asyncio
    async def test_attempt_connection_success(self, mock_agent):
        """Test successful connection attempt."""
        mock_agent.client.register_agent_with_status = AsyncMock(
            return_value=(True, {"config": "value"})
        )
        manager = ConnectionManager(mock_agent)

        result = await manager._attempt_connection()

        assert result is True
        assert manager.state == ConnectionState.CONNECTED

    @pytest.mark.asyncio
    async def test_attempt_connection_failure(self, mock_agent):
        """Test failed connection attempt."""
        mock_agent.client.register_agent_with_status = AsyncMock(
            return_value=(False, None)
        )
        manager = ConnectionManager(mock_agent)

        result = await manager._attempt_connection()

        assert result is False
        assert manager.state == ConnectionState.DISCONNECTED

    @pytest.mark.asyncio
    async def test_attempt_connection_exception(self, mock_agent):
        """Test connection attempt that raises exception."""
        mock_agent.client.register_agent_with_status = AsyncMock(
            side_effect=Exception("Network error")
        )
        manager = ConnectionManager(mock_agent)

        result = await manager._attempt_connection()

        assert result is False
        assert manager.state == ConnectionState.DISCONNECTED

    @pytest.mark.asyncio
    async def test_attempt_connection_timeout(self, mock_agent):
        """Test connection attempt that times out."""
        mock_agent.client.register_agent_with_status = AsyncMock(
            side_effect=asyncio.TimeoutError()
        )
        manager = ConnectionManager(mock_agent)

        result = await manager._attempt_connection()

        assert result is False
        assert manager.state == ConnectionState.DISCONNECTED

    @pytest.mark.asyncio
    async def test_attempt_connection_sets_connecting_state(self, mock_agent):
        """Test that attempt sets CONNECTING state during attempt."""
        states_observed = []

        async def capture_state(**kwargs):
            states_observed.append(manager.state)
            return True, None

        mock_agent.client.register_agent_with_status = capture_state
        manager = ConnectionManager(mock_agent)

        await manager._attempt_connection()

        assert ConnectionState.CONNECTING in states_observed


# Reconnection Loop Tests


@pytest.mark.unit
class TestReconnectionLoop:
    """Tests for _reconnection_loop behavior."""

    @pytest.mark.asyncio
    async def test_reconnection_loop_retries_on_failure(self, mock_agent, fast_config):
        """Test that reconnection loop retries after failure."""
        call_count = 0

        async def failing_then_success(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return False, None
            return True, None

        mock_agent.client.register_agent_with_status = failing_then_success
        manager = ConnectionManager(mock_agent, fast_config)

        manager.state = ConnectionState.DISCONNECTED
        reconnect_task = asyncio.create_task(manager._reconnection_loop())

        await asyncio.wait_for(reconnect_task, timeout=1.0)

        assert call_count == 3
        assert manager.state == ConnectionState.CONNECTED

        await manager.stop()

    @pytest.mark.asyncio
    async def test_reconnection_loop_respects_shutdown(self, mock_agent, fast_config):
        """Test that reconnection loop stops on shutdown."""
        mock_agent.client.register_agent_with_status = AsyncMock(
            return_value=(False, None)
        )
        manager = ConnectionManager(mock_agent, fast_config)

        manager.state = ConnectionState.DISCONNECTED
        reconnect_task = asyncio.create_task(manager._reconnection_loop())

        await asyncio.sleep(0.02)
        manager._shutdown_requested = True

        await asyncio.wait_for(reconnect_task, timeout=1.0)

    @pytest.mark.asyncio
    async def test_reconnection_starts_health_check_on_success(
        self, mock_agent, fast_config
    ):
        """Test that health check is started after successful reconnection."""
        attempt = 0

        async def succeed_on_second(**kwargs):
            nonlocal attempt
            attempt += 1
            return attempt >= 2, None

        mock_agent.client.register_agent_with_status = succeed_on_second
        manager = ConnectionManager(mock_agent, fast_config)
        manager.state = ConnectionState.DISCONNECTED

        reconnect_task = asyncio.create_task(manager._reconnection_loop())
        await asyncio.wait_for(reconnect_task, timeout=1.0)

        assert manager._health_check_task is not None

        await manager.stop()


# Health Check Loop Tests


@pytest.mark.unit
class TestHealthCheckLoop:
    """Tests for _health_check_loop behavior."""

    @pytest.mark.asyncio
    async def test_health_check_sends_heartbeat(self, mock_agent, fast_config):
        """Test that health check sends heartbeats."""
        mock_agent.client.register_agent_with_status = AsyncMock(
            return_value=(True, None)
        )
        mock_agent.agentfield_handler.send_enhanced_heartbeat = AsyncMock(
            return_value=True
        )

        manager = ConnectionManager(mock_agent, fast_config)
        await manager.start()

        await asyncio.sleep(0.05)

        assert mock_agent.agentfield_handler.send_enhanced_heartbeat.call_count >= 1

        await manager.stop()

    @pytest.mark.asyncio
    async def test_health_check_failure_triggers_reconnection(
        self, mock_agent, fast_config
    ):
        """Test that failed health check triggers reconnection."""
        mock_agent.client.register_agent_with_status = AsyncMock(
            return_value=(True, None)
        )
        mock_agent.agentfield_handler.send_enhanced_heartbeat = AsyncMock(
            return_value=False
        )

        manager = ConnectionManager(mock_agent, fast_config)
        await manager.start()

        assert manager.state == ConnectionState.CONNECTED

        # Make future registrations fail so reconnection doesn't succeed immediately
        mock_agent.client.register_agent_with_status = AsyncMock(
            return_value=(False, None)
        )

        for _ in range(10):
            await asyncio.sleep(0.02)
            if manager.state != ConnectionState.CONNECTED:
                break

        assert manager.state in (ConnectionState.DEGRADED, ConnectionState.RECONNECTING)

        await manager.stop()

    @pytest.mark.asyncio
    async def test_health_check_stops_on_shutdown(self, mock_agent, fast_config):
        """Test that health check loop stops on shutdown."""
        mock_agent.client.register_agent_with_status = AsyncMock(
            return_value=(True, None)
        )
        mock_agent.agentfield_handler.send_enhanced_heartbeat = AsyncMock(
            return_value=True
        )

        manager = ConnectionManager(mock_agent, fast_config)
        await manager.start()

        await manager.stop()

        if manager._health_check_task:
            assert (
                manager._health_check_task.done()
                or manager._health_check_task.cancelled()
            )


# Callback Tests


@pytest.mark.unit
class TestCallbacks:
    """Tests for connection/disconnection callbacks."""

    @pytest.mark.asyncio
    async def test_on_connected_callback_called(self, mock_agent, fast_config):
        """Test that on_connected callback is called on successful connection."""
        mock_agent.client.register_agent_with_status = AsyncMock(
            return_value=(True, None)
        )

        on_connected = Mock()
        manager = ConnectionManager(mock_agent, fast_config)
        manager.on_connected = on_connected

        await manager.start()

        on_connected.assert_called_once()

        await manager.stop()

    @pytest.mark.asyncio
    async def test_on_disconnected_callback_called(self, mock_agent, fast_config):
        """Test that on_disconnected callback is called on connection failure."""
        mock_agent.client.register_agent_with_status = AsyncMock(
            return_value=(False, None)
        )

        on_disconnected = Mock()
        manager = ConnectionManager(mock_agent, fast_config)
        manager.on_disconnected = on_disconnected

        await manager.start()

        on_disconnected.assert_called_once()

        await manager.stop()

    @pytest.mark.asyncio
    async def test_callback_exception_does_not_crash(self, mock_agent, fast_config):
        """Test that callback exceptions are caught and logged."""
        mock_agent.client.register_agent_with_status = AsyncMock(
            return_value=(True, None)
        )

        on_connected = Mock(side_effect=RuntimeError("Callback error"))
        manager = ConnectionManager(mock_agent, fast_config)
        manager.on_connected = on_connected

        # Should not raise
        await manager.start()

        assert manager.state == ConnectionState.CONNECTED
        on_connected.assert_called_once()

        await manager.stop()

    @pytest.mark.asyncio
    async def test_disconnected_callback_exception_handled(
        self, mock_agent, fast_config
    ):
        """Test that disconnected callback exceptions are handled."""
        mock_agent.client.register_agent_with_status = AsyncMock(
            return_value=(False, None)
        )

        on_disconnected = Mock(side_effect=RuntimeError("Disconnected callback error"))
        manager = ConnectionManager(mock_agent, fast_config)
        manager.on_disconnected = on_disconnected

        # Should not raise
        await manager.start()

        on_disconnected.assert_called_once()

        await manager.stop()


# Helper Method Tests


@pytest.mark.unit
class TestHelperMethods:
    """Tests for is_connected, is_degraded, and other helper methods."""

    def test_is_connected_true_when_connected(self, mock_agent):
        """Test is_connected returns True when connected."""
        manager = ConnectionManager(mock_agent)
        manager.state = ConnectionState.CONNECTED

        assert manager.is_connected() is True

    def test_is_degraded_true_when_degraded(self, mock_agent):
        """Test is_degraded returns True when degraded."""
        manager = ConnectionManager(mock_agent)
        manager.state = ConnectionState.DEGRADED

        assert manager.is_degraded() is True


# force_reconnect() Tests


@pytest.mark.unit
class TestForceReconnect:
    """Tests for force_reconnect method."""

    @pytest.mark.asyncio
    async def test_force_reconnect_when_already_connected(self, mock_agent):
        """Test force_reconnect returns True when already connected."""
        manager = ConnectionManager(mock_agent)
        manager.state = ConnectionState.CONNECTED

        result = await manager.force_reconnect()

        assert result is True
        assert manager.state == ConnectionState.CONNECTED

    @pytest.mark.asyncio
    async def test_force_reconnect_success(self, mock_agent, fast_config):
        """Test force_reconnect successfully reconnects."""
        mock_agent.client.register_agent_with_status = AsyncMock(
            return_value=(True, None)
        )

        manager = ConnectionManager(mock_agent, fast_config)
        manager.state = ConnectionState.DEGRADED

        result = await manager.force_reconnect()

        assert result is True
        assert manager.state == ConnectionState.CONNECTED
        assert manager._health_check_task is not None

        await manager.stop()

    @pytest.mark.asyncio
    async def test_force_reconnect_failure(self, mock_agent):
        """Test force_reconnect returns False on failure."""
        mock_agent.client.register_agent_with_status = AsyncMock(
            return_value=(False, None)
        )

        manager = ConnectionManager(mock_agent)
        manager.state = ConnectionState.DEGRADED

        result = await manager.force_reconnect()

        assert result is False

    @pytest.mark.asyncio
    async def test_force_reconnect_cancels_existing_reconnection_task(
        self, mock_agent, fast_config
    ):
        """Test that force_reconnect cancels existing reconnection task."""
        mock_agent.client.register_agent_with_status = AsyncMock(
            return_value=(False, None)
        )
        manager = ConnectionManager(mock_agent, fast_config)

        manager.state = ConnectionState.RECONNECTING
        old_task = asyncio.create_task(manager._reconnection_loop())
        manager._reconnection_task = old_task
        await asyncio.sleep(0.01)

        # Now make client succeed
        mock_agent.client.register_agent_with_status = AsyncMock(
            return_value=(True, None)
        )

        result = await manager.force_reconnect()

        assert result is True
        await asyncio.sleep(0.02)
        assert old_task.done() or old_task.cancelled()

        await manager.stop()


# Connection Lifecycle Tests


@pytest.mark.unit
class TestConnectionLifecycle:
    """Tests for full connection lifecycle scenarios."""

    @pytest.mark.asyncio
    async def test_full_lifecycle_connect_disconnect(self, mock_agent, fast_config):
        """Test full lifecycle: start connected, stop."""
        mock_agent.client.register_agent_with_status = AsyncMock(
            return_value=(True, None)
        )
        mock_agent.agentfield_handler.send_enhanced_heartbeat = AsyncMock(
            return_value=True
        )

        manager = ConnectionManager(mock_agent, fast_config)

        result = await manager.start()
        assert result is True
        assert manager.is_connected()
        assert mock_agent.agentfield_connected is True

        await manager.stop()
        assert manager._shutdown_requested is True

    @pytest.mark.asyncio
    async def test_lifecycle_degraded_to_connected(self, mock_agent, fast_config):
        """Test lifecycle: start degraded, then reconnect."""
        attempts = 0

        async def succeed_later(**kwargs):
            nonlocal attempts
            attempts += 1
            return attempts >= 2, None

        mock_agent.client.register_agent_with_status = succeed_later

        manager = ConnectionManager(mock_agent, fast_config)

        result = await manager.start()
        assert result is False

        await asyncio.sleep(0.1)

        assert manager.is_connected()
        assert mock_agent.agentfield_connected is True

        await manager.stop()

    @pytest.mark.asyncio
    async def test_last_successful_connection_updated(self, mock_agent, fast_config):
        """Test that last_successful_connection is updated on connect."""
        mock_agent.client.register_agent_with_status = AsyncMock(
            return_value=(True, None)
        )

        manager = ConnectionManager(mock_agent, fast_config)
        assert manager.last_successful_connection is None

        before = time.time()
        await manager.start()
        after = time.time()

        assert manager.last_successful_connection is not None
        assert before <= manager.last_successful_connection <= after

        await manager.stop()


# Error Handling Tests


@pytest.mark.unit
class TestErrorHandling:
    """Tests for various error scenarios."""

    @pytest.mark.asyncio
    async def test_connection_error_handled_gracefully(self, mock_agent, fast_config):
        """Test that connection errors are handled gracefully."""
        mock_agent.client.register_agent_with_status = AsyncMock(
            side_effect=ConnectionError("Connection refused")
        )

        manager = ConnectionManager(mock_agent, fast_config)

        result = await manager._attempt_connection()

        assert result is False
        assert manager.state == ConnectionState.DISCONNECTED

    @pytest.mark.asyncio
    async def test_health_check_error_triggers_reconnection(
        self, mock_agent, fast_config
    ):
        """Test that health check errors trigger reconnection."""
        mock_agent.client.register_agent_with_status = AsyncMock(
            return_value=(True, None)
        )

        call_count = 0

        async def heartbeat_then_fail():
            nonlocal call_count
            call_count += 1
            if call_count > 1:
                raise Exception("Heartbeat error")
            return True

        mock_agent.agentfield_handler.send_enhanced_heartbeat = heartbeat_then_fail

        manager = ConnectionManager(mock_agent, fast_config)
        await manager.start()

        await asyncio.sleep(0.05)

        assert manager.state in (ConnectionState.DEGRADED, ConnectionState.RECONNECTING)

        await manager.stop()

    @pytest.mark.asyncio
    async def test_reconnection_loop_handles_exceptions(self, mock_agent, fast_config):
        """Test that reconnection loop handles unexpected exceptions."""
        call_count = 0

        async def fail_then_succeed(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Unexpected error")
            return True, None

        mock_agent.client.register_agent_with_status = fail_then_succeed

        manager = ConnectionManager(mock_agent, fast_config)
        manager.state = ConnectionState.DISCONNECTED

        reconnect_task = asyncio.create_task(manager._reconnection_loop())

        await asyncio.wait_for(reconnect_task, timeout=1.0)

        assert manager.state == ConnectionState.CONNECTED

        await manager.stop()


# Timeout Handling Tests


@pytest.mark.unit
class TestTimeoutHandling:
    """Tests for timeout scenarios."""

    @pytest.mark.asyncio
    async def test_connection_timeout_treated_as_failure(self, mock_agent):
        """Test that connection timeout is treated as failure."""
        mock_agent.client.register_agent_with_status = AsyncMock(
            side_effect=asyncio.TimeoutError()
        )

        manager = ConnectionManager(mock_agent)

        result = await manager._attempt_connection()

        assert result is False
        assert manager.state == ConnectionState.DISCONNECTED

    @pytest.mark.asyncio
    async def test_task_cancellation_during_reconnection(self, mock_agent, fast_config):
        """Test that task cancellation during reconnection is handled gracefully."""
        mock_agent.client.register_agent_with_status = AsyncMock(
            return_value=(False, None)
        )

        manager = ConnectionManager(mock_agent, fast_config)
        manager.state = ConnectionState.DISCONNECTED

        reconnect_task = asyncio.create_task(manager._reconnection_loop())
        await asyncio.sleep(0.01)

        reconnect_task.cancel()

        try:
            await asyncio.wait_for(reconnect_task, timeout=0.5)
        except asyncio.CancelledError:
            pass

        assert reconnect_task.done()


# Integration-style Tests


@pytest.mark.unit
class TestIntegration:
    """Integration-style tests for complex scenarios."""

    @pytest.mark.asyncio
    async def test_multiple_reconnection_cycles(self, mock_agent, fast_config):
        """Test multiple disconnect/reconnect cycles."""
        cycle = 0

        async def alternate_success(**kwargs):
            nonlocal cycle
            cycle += 1
            return (cycle % 2 == 1), None

        mock_agent.client.register_agent_with_status = alternate_success
        mock_agent.agentfield_handler.send_enhanced_heartbeat = AsyncMock(
            return_value=True
        )

        manager = ConnectionManager(mock_agent, fast_config)

        await manager.start()
        assert manager.is_connected()

        manager.state = ConnectionState.DEGRADED

        result = await manager.force_reconnect()  # cycle 2 - fails
        assert result is False

        result = await manager.force_reconnect()  # cycle 3 - succeeds
        assert result is True

        await manager.stop()

    @pytest.mark.asyncio
    async def test_rapid_start_stop_cycles(self, mock_agent, fast_config):
        """Test rapid start/stop cycles don't cause issues."""
        mock_agent.client.register_agent_with_status = AsyncMock(
            return_value=(True, None)
        )

        for _ in range(3):
            manager = ConnectionManager(mock_agent, fast_config)
            await manager.start()
            await manager.stop()

    @pytest.mark.asyncio
    async def test_connection_reuse_after_health_failure(self, mock_agent, fast_config):
        """Test that connection is properly reestablished after health failure."""
        mock_agent.client.register_agent_with_status = AsyncMock(
            return_value=(True, None)
        )

        call_idx = 0
        heartbeat_results = [True, False]

        async def varying_heartbeat():
            nonlocal call_idx
            result = heartbeat_results[min(call_idx, len(heartbeat_results) - 1)]
            call_idx += 1
            return result

        mock_agent.agentfield_handler.send_enhanced_heartbeat = varying_heartbeat

        manager = ConnectionManager(mock_agent, fast_config)
        await manager.start()

        await asyncio.sleep(0.05)

        assert (
            manager._reconnection_task is not None
            or manager.state == ConnectionState.CONNECTED
        )

        await manager.stop()
