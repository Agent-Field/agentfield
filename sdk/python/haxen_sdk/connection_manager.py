"""
Haxen SDK Connection Manager

Provides resilient connection handling for Haxen server connectivity.
Handles automatic reconnection, graceful degradation, and connection health monitoring.
"""

import asyncio
import time
from enum import Enum
from typing import Optional, Callable, Any, Dict
from dataclasses import dataclass
from haxen_sdk.logger import log_debug, log_info, log_warn, log_error


class ConnectionState(Enum):
    """Connection states for Haxen server connectivity"""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    DEGRADED = "degraded"  # Running locally without Haxen server


@dataclass
class ConnectionConfig:
    """Configuration for connection management"""

    retry_interval: float = 10.0  # Consistent retry interval in seconds
    health_check_interval: float = 30.0  # Health check interval in seconds
    connection_timeout: float = 10.0  # Connection timeout in seconds


class ConnectionManager:
    """
    Manages resilient connections to Haxen server with automatic reconnection,
    graceful degradation, and health monitoring.

    Uses a simple, consistent retry interval to ensure immediate reconnection
    when Haxen server becomes available.
    """

    def __init__(self, agent, config: Optional[ConnectionConfig] = None):
        self.agent = agent
        self.config = config or ConnectionConfig()

        # Connection state
        self.state = ConnectionState.DISCONNECTED
        self.last_successful_connection = None

        # Tasks
        self._reconnection_task: Optional[asyncio.Task] = None
        self._health_check_task: Optional[asyncio.Task] = None
        self._shutdown_requested = False

        # Callbacks
        self.on_connected: Optional[Callable] = None
        self.on_disconnected: Optional[Callable] = None
        self.on_degraded: Optional[Callable] = None

    async def start(self) -> bool:
        """
        Start the connection manager and attempt initial connection.

        Returns:
            True if initial connection successful, False if entering degraded mode
        """
        log_info("Starting connection manager")

        # Attempt initial connection
        success = await self._attempt_connection()

        if success:
            self._on_connection_success()
            # Start health monitoring
            self._health_check_task = asyncio.create_task(self._health_check_loop())
        else:
            self._on_connection_failure()
            # Start reconnection attempts
            self._reconnection_task = asyncio.create_task(self._reconnection_loop())

        return success

    async def stop(self):
        """Stop the connection manager and cleanup tasks"""
        log_info("Stopping connection manager")
        self._shutdown_requested = True

        # Cancel tasks
        if self._reconnection_task and not self._reconnection_task.done():
            self._reconnection_task.cancel()
            try:
                await self._reconnection_task
            except asyncio.CancelledError:
                pass

        if self._health_check_task and not self._health_check_task.done():
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

    async def _attempt_connection(self) -> bool:
        """
        Attempt to connect to Haxen server.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.state = ConnectionState.CONNECTING
            log_debug("Attempting connection to Haxen server")

            # Try to register with Haxen server - suppress verbose error logging
            import logging

            # Temporarily suppress httpx and httpcore logging to avoid verbose connection errors
            httpx_logger = logging.getLogger("httpx")
            httpcore_logger = logging.getLogger("httpcore")
            original_httpx_level = httpx_logger.level
            original_httpcore_level = httpcore_logger.level

            # Set to ERROR level to suppress connection attempt logs
            httpx_logger.setLevel(logging.ERROR)
            httpcore_logger.setLevel(logging.ERROR)

            discovery_payload = self.agent._build_callback_discovery_payload()

            success = False
            payload: Optional[Dict[str, Any]] = None

            try:
                success, payload = await self.agent.client.register_agent_with_status(
                    node_id=self.agent.node_id,
                    reasoners=self.agent.reasoners,
                    skills=self.agent.skills,
                    base_url=self.agent.base_url,
                    status=self.agent._current_status,
                    discovery=discovery_payload,
                    suppress_errors=True,  # Suppress verbose error logging for connection attempts
                )
            finally:
                # Restore original logging levels
                httpx_logger.setLevel(original_httpx_level)
                httpcore_logger.setLevel(original_httpcore_level)

            if success:
                if payload:
                    self.agent._apply_discovery_response(payload)
                self.state = ConnectionState.CONNECTED
                return True
            else:
                self.state = ConnectionState.DISCONNECTED
                return False

        except Exception as e:
            # Only log at debug level to avoid spam
            log_debug(f"Connection attempt failed: {type(e).__name__}")
            self.state = ConnectionState.DISCONNECTED
            return False

    async def _health_check_loop(self):
        """Background loop for monitoring connection health"""
        while not self._shutdown_requested and self.state == ConnectionState.CONNECTED:
            try:
                await asyncio.sleep(self.config.health_check_interval)

                if self._shutdown_requested:
                    break

                # Try to send a heartbeat to check connection health
                success = await self.agent.haxen_handler.send_enhanced_heartbeat()

                if not success:
                    log_warn("Health check failed - connection lost")
                    self._on_connection_failure()
                    # Start reconnection attempts
                    self._reconnection_task = asyncio.create_task(
                        self._reconnection_loop()
                    )
                    break

            except asyncio.CancelledError:
                break
            except Exception as e:
                log_error(f"Health check error: {e}")
                self._on_connection_failure()
                # Start reconnection attempts
                self._reconnection_task = asyncio.create_task(self._reconnection_loop())
                break

    async def _reconnection_loop(self):
        """Background loop for attempting reconnection"""
        self.state = ConnectionState.RECONNECTING

        while not self._shutdown_requested and self.state != ConnectionState.CONNECTED:
            try:
                log_debug(
                    f"Attempting reconnection in {self.config.retry_interval} seconds..."
                )
                await asyncio.sleep(self.config.retry_interval)

                if self._shutdown_requested:
                    break

                success = await self._attempt_connection()

                if success:
                    self._on_connection_success()
                    # Start health monitoring again
                    self._health_check_task = asyncio.create_task(
                        self._health_check_loop()
                    )
                    break
                else:
                    log_debug("Reconnection attempt failed, will retry")

            except asyncio.CancelledError:
                break
            except Exception as e:
                log_error(f"Reconnection error: {e}")
                # Continue trying

    def _on_connection_success(self):
        """Handle successful connection"""
        self.state = ConnectionState.CONNECTED
        self.last_successful_connection = time.time()
        self.agent.haxen_connected = True

        log_info("Connected to Haxen server")

        if self.on_connected:
            try:
                self.on_connected()
            except Exception as e:
                log_error(f"Error in connection callback: {e}")

    def _on_connection_failure(self):
        """Handle connection failure"""
        self.state = ConnectionState.DEGRADED
        self.agent.haxen_connected = False

        log_warn("Haxen server unavailable - running in degraded mode")

        if self.on_disconnected:
            try:
                self.on_disconnected()
            except Exception as e:
                log_error(f"Error in disconnection callback: {e}")

    def is_connected(self) -> bool:
        """Check if currently connected to Haxen server"""
        return self.state == ConnectionState.CONNECTED

    def is_degraded(self) -> bool:
        """Check if running in degraded mode"""
        return self.state == ConnectionState.DEGRADED

    async def force_reconnect(self):
        """Force an immediate reconnection attempt"""
        if self.state == ConnectionState.CONNECTED:
            return True

        log_info("Forcing reconnection attempt")
        success = await self._attempt_connection()

        if success:
            self._on_connection_success()
            # Cancel existing reconnection task if running
            if self._reconnection_task and not self._reconnection_task.done():
                self._reconnection_task.cancel()
            # Start health monitoring
            self._health_check_task = asyncio.create_task(self._health_check_loop())

        return success
