import asyncio
import os
import signal
import threading
from datetime import datetime

import requests
from agentfield.types import AgentStatus, HeartbeatData
from agentfield.logger import (
    log_heartbeat,
    log_debug,
    log_warn,
    log_error,
    log_success,
    log_setup,
    log_info,
)


class AgentFieldHandler:
    """
    AgentField Server Communication handler for Agent class.

    This class encapsulates all AgentField server communication functionality including:
    - Agent registration with AgentField server
    - Heartbeat management (both simple and enhanced)
    - Fast lifecycle management
    - Graceful shutdown notifications
    - Signal handling for fast shutdown
    """

    def __init__(self, agent_instance):
        """
        Initialize the AgentField handler with a reference to the agent instance.

        Args:
            agent_instance: The Agent instance this handler belongs to
        """
        self.agent = agent_instance

    async def register_with_agentfield_server(self, port: int):
        """Register this agent node with AgentField server"""
        # Import the callback URL resolution function
        from agentfield.agent import (
            _build_callback_candidates,
            _resolve_callback_url,
            _is_running_in_container,
        )

        # Enhanced debugging for callback URL resolution
        log_debug("Starting callback URL resolution")
        log_debug(f"Original callback_url parameter: {self.agent.callback_url}")
        log_debug(
            f"AGENT_CALLBACK_URL env var: {os.environ.get('AGENT_CALLBACK_URL', 'NOT_SET')}"
        )
        log_debug(f"Port: {port}")
        log_debug(f"Running in container: {_is_running_in_container()}")
        log_debug(
            f"All env vars containing 'AGENT': {[k for k in os.environ.keys() if 'AGENT' in k.upper()]}"
        )

        # 🔥 FIX: Only resolve callback URL if not already set
        # This prevents overwriting the URL resolved in Agent.__init__()
        if not self.agent.base_url:
            self.agent.callback_candidates = _build_callback_candidates(
                self.agent.callback_url, port
            )
            if self.agent.callback_candidates:
                self.agent.base_url = self.agent.callback_candidates[0]
                log_debug(
                    f"Resolved callback URL during registration: {self.agent.base_url}"
                )
            else:
                self.agent.base_url = _resolve_callback_url(
                    self.agent.callback_url, port
                )
                log_debug(
                    f"Resolved callback URL during registration: {self.agent.base_url}"
                )
        else:
            # Update port in existing base_url if needed, but preserve Railway internal URLs
            import urllib.parse

            parsed = urllib.parse.urlparse(self.agent.base_url)

            # Don't modify Railway internal URLs or other container-specific URLs
            if "railway.internal" in parsed.netloc or "internal" in parsed.netloc:
                log_debug(
                    f"Preserving container-specific callback URL: {self.agent.base_url}"
                )
            elif parsed.port != port:
                # Update the port in the existing URL
                self.agent.base_url = f"{parsed.scheme}://{parsed.hostname}:{port}"
                log_debug(
                    f"Updated port in existing callback URL: {self.agent.base_url}"
                )
            else:
                log_debug(f"Using existing callback URL: {self.agent.base_url}")

        if not self.agent.callback_candidates:
            self.agent.callback_candidates = _build_callback_candidates(
                self.agent.base_url, port
            )
        elif (
            self.agent.base_url
            and self.agent.callback_candidates[0] != self.agent.base_url
        ):
            # Keep resolved base URL at front for clarity
            if self.agent.base_url in self.agent.callback_candidates:
                self.agent.callback_candidates.remove(self.agent.base_url)
            self.agent.callback_candidates.insert(0, self.agent.base_url)

        # Always log the resolved callback URL for debugging
        log_info(f"Final callback URL: {self.agent.base_url}")

        if self.agent.dev_mode:
            log_debug(f"Final callback URL: {self.agent.base_url}")

        try:
            log_debug(
                f"Attempting to register with AgentField server at {self.agent.agentfield_server}"
            )
            discovery_payload = self.agent._build_callback_discovery_payload()

            success, payload = await self.agent.client.register_agent(
                node_id=self.agent.node_id,
                reasoners=self.agent.reasoners,
                skills=self.agent.skills,
                base_url=self.agent.base_url,
                discovery=discovery_payload,
                vc_metadata=self.agent._build_vc_metadata(),
            )
            if success:
                if payload:
                    self.agent._apply_discovery_response(payload)
                log_success(
                    f"Registered node '{self.agent.node_id}' with AgentField server"
                )
                self.agent.agentfield_connected = True

                # Attempt DID registration after successful AgentField registration
                if self.agent.did_manager:
                    did_success = self.agent._register_agent_with_did()
                    if not did_success and self.agent.dev_mode:
                        log_warn(
                            "DID registration failed, continuing without DID functionality"
                        )
            else:
                log_error("Registration failed")
                self.agent.agentfield_connected = False

        except Exception as e:
            self.agent.agentfield_connected = False
            if self.agent.dev_mode:
                log_warn(f"AgentField server not available: {e}")
                log_setup("Running in development mode - agent will work standalone")
                log_info(
                    f"To connect to AgentField server, start it at {self.agent.agentfield_server}"
                )
            else:
                log_error(f"Failed to register with AgentField server: {e}")
                if (
                    isinstance(e, requests.exceptions.RequestException)
                    and e.response is not None
                ):
                    log_warn(f"Response status: {e.response.status_code}")
                    log_warn(f"Response text: {e.response.text}")
                raise

    def send_heartbeat(self):
        """Send heartbeat to AgentField server"""
        if not self.agent.agentfield_connected:
            return  # Skip heartbeat if not connected to AgentField

        try:
            headers = {"Content-Type": "application/json"}
            if self.agent.api_key:
                headers["X-API-Key"] = self.agent.api_key
            response = requests.post(
                f"{self.agent.agentfield_server}/api/v1/nodes/{self.agent.node_id}/heartbeat",
                headers=headers,
                timeout=5,
            )
            if response.status_code == 200:
                log_heartbeat("Heartbeat sent successfully")
            else:
                log_warn(
                    f"Heartbeat failed with status {response.status_code}: {response.text}"
                )
        except Exception as e:
            log_error(f"Failed to send heartbeat: {e}")

    def heartbeat_worker(
        self, interval: int = 30
    ):  # pragma: no cover - long-running thread loop
        """Background worker that sends periodic heartbeats"""
        if not self.agent.agentfield_connected:
            log_heartbeat(
                "Heartbeat worker skipped - not connected to AgentField server"
            )
            return

        log_heartbeat(f"Starting heartbeat worker (interval: {interval}s)")
        while not self.agent._heartbeat_stop_event.wait(interval):
            self.send_heartbeat()
        log_heartbeat("Heartbeat worker stopped")

    def start_heartbeat(self, interval: int = 30):
        """Start the heartbeat background thread"""
        if not self.agent.agentfield_connected:
            return  # Skip heartbeat if not connected to AgentField

        if (
            self.agent._heartbeat_thread is None
            or not self.agent._heartbeat_thread.is_alive()
        ):
            self.agent._heartbeat_stop_event.clear()
            self.agent._heartbeat_thread = threading.Thread(
                target=self.heartbeat_worker, args=(interval,), daemon=True
            )
            self.agent._heartbeat_thread.start()

    def stop_heartbeat(self):
        """Stop the heartbeat background thread"""
        if self.agent._heartbeat_thread and self.agent._heartbeat_thread.is_alive():
            log_debug("Stopping heartbeat worker...")
            self.agent._heartbeat_stop_event.set()
            self.agent._heartbeat_thread.join(timeout=5)

    async def send_enhanced_heartbeat(self) -> bool:
        """
        Send enhanced heartbeat with current status and MCP information.

        Returns:
            True if heartbeat was successful, False otherwise
        """
        if not self.agent.agentfield_connected:
            return False

        try:
            # Get MCP server health information
            mcp_servers = self.agent.mcp_handler._get_mcp_server_health()

            # Create heartbeat data
            heartbeat_data = HeartbeatData(
                status=self.agent._current_status,
                mcp_servers=mcp_servers,
                timestamp=datetime.now().isoformat(),
            )

            # Send enhanced heartbeat
            success = await self.agent.client.send_enhanced_heartbeat(
                self.agent.node_id, heartbeat_data
            )

            if success:
                log_heartbeat(
                    f"Enhanced heartbeat sent - Status: {self.agent._current_status.value}"
                )

            return success

        except Exception as e:
            if self.agent.dev_mode:
                log_error(f"Enhanced heartbeat failed: {e}")
            return False

    async def notify_shutdown(self) -> bool:
        """
        Notify AgentField server of graceful shutdown.

        Returns:
            True if notification was successful, False otherwise
        """
        if not self.agent.agentfield_connected:
            return False

        try:
            success = await self.agent.client.notify_graceful_shutdown(
                self.agent.node_id
            )
            if self.agent.dev_mode and success:
                log_success("Graceful shutdown notification sent")
            return success
        except Exception as e:
            if self.agent.dev_mode:
                log_error(f"Shutdown notification failed: {e}")
            return False

    def setup_fast_lifecycle_signal_handlers(
        self,
    ) -> None:  # pragma: no cover - requires OS signal integration
        """
        Setup signal handler for fast lifecycle status while allowing uvicorn to perform graceful shutdown.

        - Only intercepts SIGTERM to mark the agent offline and notify AgentField immediately.
        - Leaves SIGINT (Ctrl+C) to uvicorn so its shutdown hooks run and resources are cleaned up.
        """

        def signal_handler(signum: int, frame) -> None:
            """Handle SIGTERM: mark offline, notify AgentField, then re-emit the signal for default handling."""
            signal_name = "SIGTERM" if signum == signal.SIGTERM else "SIGINT"

            if self.agent.dev_mode:
                log_warn(
                    f"{signal_name} received - initiating graceful shutdown via uvicorn"
                )

            # Set shutdown flag
            self.agent._shutdown_requested = True
            self.agent._current_status = AgentStatus.OFFLINE

            # Best-effort immediate notification to AgentField
            try:
                success = self.agent.client.notify_graceful_shutdown_sync(
                    self.agent.node_id
                )
                if self.agent.dev_mode:
                    state = "sent" if success else "failed"
                    log_info(f"Shutdown notification {state}")
            except Exception as e:
                if self.agent.dev_mode:
                    log_error(f"Shutdown notification error: {e}")

            # IMPORTANT: Do not perform heavy cleanup here. Let FastAPI/uvicorn shutdown events handle it.
            # Re-install default handler and re-emit the same signal so uvicorn orchestrates cleanup.
            try:
                signal.signal(signum, signal.SIG_DFL)
                os.kill(os.getpid(), signum)
            except Exception:
                # Fallback: polite exit (still allows finally blocks/atexit to run)
                import sys

                sys.exit(0)

        try:
            # Only register for SIGTERM; leave SIGINT (Ctrl+C) to uvicorn
            signal.signal(signal.SIGTERM, signal_handler)

            if self.agent.dev_mode:
                log_debug("Fast lifecycle signal handler registered (SIGTERM only)")
        except Exception as e:
            if self.agent.dev_mode:
                log_error(f"Failed to setup signal handlers: {e}")

    async def register_with_fast_lifecycle(
        self, port: int
    ) -> bool:  # pragma: no cover - fast-path relies on external coordination
        """
        Register agent with immediate status reporting for fast lifecycle.

        Args:
            port: The port the agent is running on

        Returns:
            True if registration was successful, False otherwise
        """
        from agentfield.agent import _build_callback_candidates, _resolve_callback_url

        if not self.agent.base_url:
            self.agent.callback_candidates = _build_callback_candidates(
                self.agent.callback_url, port
            )
            if self.agent.callback_candidates:
                self.agent.base_url = self.agent.callback_candidates[0]
                log_debug(
                    f"Fast lifecycle - Resolved callback URL during registration: {self.agent.base_url}"
                )
            else:
                self.agent.base_url = _resolve_callback_url(
                    self.agent.callback_url, port
                )
                log_debug(
                    f"Fast lifecycle - Resolved callback URL during registration: {self.agent.base_url}"
                )
        else:
            import urllib.parse

            parsed = urllib.parse.urlparse(self.agent.base_url)
            if parsed.port != port:
                self.agent.base_url = f"{parsed.scheme}://{parsed.hostname}:{port}"
                log_debug(
                    f"Fast lifecycle - Updated port in existing callback URL: {self.agent.base_url}"
                )
            else:
                log_debug(
                    f"Fast lifecycle - Using existing callback URL: {self.agent.base_url}"
                )

        if not self.agent.callback_candidates:
            self.agent.callback_candidates = _build_callback_candidates(
                self.agent.base_url, port
            )
        elif (
            self.agent.base_url
            and self.agent.callback_candidates
            and self.agent.callback_candidates[0] != self.agent.base_url
        ):
            if self.agent.base_url in self.agent.callback_candidates:
                self.agent.callback_candidates.remove(self.agent.base_url)
            self.agent.callback_candidates.insert(0, self.agent.base_url)

        log_debug(f"Fast lifecycle - Final callback URL: {self.agent.base_url}")
        log_debug(
            f"Fast lifecycle - Original callback_url parameter: {self.agent.callback_url}"
        )
        log_debug(
            f"Fast lifecycle - AGENT_CALLBACK_URL env var: {os.environ.get('AGENT_CALLBACK_URL', 'NOT_SET')}"
        )
        log_debug(f"Fast lifecycle - Port: {port}")

        try:
            if self.agent.dev_mode:
                log_info(
                    f"Fast registration with AgentField server at {self.agent.agentfield_server}"
                )
                log_info(f"Using callback URL: {self.agent.base_url}")

            # Register with STARTING status for immediate visibility
            discovery_payload = self.agent._build_callback_discovery_payload()

            success, payload = await self.agent.client.register_agent_with_status(
                node_id=self.agent.node_id,
                reasoners=self.agent.reasoners,
                skills=self.agent.skills,
                base_url=self.agent.base_url,
                status=AgentStatus.STARTING,
                discovery=discovery_payload,
                vc_metadata=self.agent._build_vc_metadata(),
            )

            if success:
                if payload:
                    self.agent._apply_discovery_response(payload)
                if self.agent.dev_mode:
                    log_success(
                        f"Fast registration successful - Status: {AgentStatus.STARTING.value}"
                    )
                self.agent.agentfield_connected = True

                # Attempt DID registration after successful AgentField registration
                if self.agent.did_manager:
                    did_success = self.agent._register_agent_with_did()
                    if not did_success and self.agent.dev_mode:
                        log_warn(
                            "DID registration failed, continuing without DID functionality"
                        )

                return True
            else:
                if self.agent.dev_mode:
                    log_error("Fast registration failed")
                self.agent.agentfield_connected = False
                return False

        except Exception as e:
            self.agent.agentfield_connected = False
            if self.agent.dev_mode:
                log_warn(f"Fast registration error: {e}")
            return False

    async def enhanced_heartbeat_loop(self, interval: int) -> None:
        """
        Background loop for sending enhanced heartbeats with status and MCP information.

        Args:
            interval: Heartbeat interval in seconds
        """
        if self.agent.dev_mode:
            log_debug(f"Enhanced heartbeat loop started (interval: {interval}s)")

        while not self.agent._shutdown_requested:
            try:
                # Send enhanced heartbeat
                success = await self.send_enhanced_heartbeat()

                if not success and self.agent.dev_mode:
                    log_warn("Enhanced heartbeat failed - retrying next cycle")

                # Wait for next heartbeat interval
                await asyncio.sleep(interval)

            except asyncio.CancelledError:
                if self.agent.dev_mode:
                    log_debug("Enhanced heartbeat loop cancelled")
                break
            except Exception as e:
                if self.agent.dev_mode:
                    log_error(f"Enhanced heartbeat loop error: {e}")
                # Continue loop even on errors
                await asyncio.sleep(interval)

        if self.agent.dev_mode:
            log_debug("Enhanced heartbeat loop stopped")
