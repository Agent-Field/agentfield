import asyncio
import importlib.util
import os
import signal
import urllib.parse
from datetime import datetime
from typing import Optional

import uvicorn
from agentfield.agent_utils import AgentUtils
from agentfield.logger import log_debug, log_error, log_info, log_success, log_warn
from agentfield.utils import get_free_port
from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.routing import APIRoute


class AgentServer:
    """Server management functionality for AgentField Agent"""

    def __init__(self, agent_instance):
        """
        Initialize the AgentServer with a reference to the agent instance.

        Args:
            agent_instance: The Agent instance this server manages
        """
        self.agent = agent_instance
        self._in_flight_tasks: set[asyncio.Task] = set()

    def _track_task(self, task: asyncio.Task) -> asyncio.Task:
        """Track an in-flight task until completion."""
        self._in_flight_tasks.add(task)
        task.add_done_callback(self._in_flight_tasks.discard)
        return task

    def setup_agentfield_routes(self):
        """Setup standard routes that AgentField server expects"""
        from agentfield.node_logs import install_stdio_tee

        install_stdio_tee()

        @self.agent.get("/agentfield/v1/logs")
        async def agentfield_process_logs(request: Request):
            """NDJSON tail/stream of captured stdout/stderr (control plane proxy)."""
            from agentfield import node_logs

            if not node_logs.logs_enabled():
                return JSONResponse(
                    status_code=404,
                    content={"error": "logs_disabled", "message": "Process logs API is disabled"},
                )
            auth = request.headers.get("authorization") or request.headers.get(
                "Authorization"
            )
            if not node_logs.verify_internal_bearer(auth):
                return JSONResponse(
                    status_code=401,
                    content={"error": "unauthorized", "message": "Valid Authorization Bearer required"},
                )
            qp = request.query_params
            try:
                tail_lines = int(qp.get("tail_lines") or "0")
            except ValueError:
                tail_lines = 0
            try:
                since_seq = int(qp.get("since_seq") or "0")
            except ValueError:
                since_seq = 0
            follow = (qp.get("follow") or "").lower() in ("1", "true", "yes")
            max_tail = int(os.getenv("AGENTFIELD_LOG_MAX_TAIL_LINES", "50000"))
            if tail_lines > max_tail:
                return JSONResponse(
                    status_code=413,
                    content={
                        "error": "tail_too_large",
                        "message": f"tail_lines exceeds max {max_tail}",
                    },
                )
            if tail_lines <= 0 and since_seq <= 0 and not follow:
                tail_lines = 200
            gen = node_logs.iter_tail_ndjson(tail_lines, since_seq, follow)
            return StreamingResponse(
                gen,
                media_type="application/x-ndjson",
                headers={
                    "Cache-Control": "no-store",
                    "X-Content-Type-Options": "nosniff",
                },
            )

        @self.agent.get("/debug/tasks")
        async def debug_tasks():
            """Dump every live asyncio task with its current stack frames.

            Use this to find deadlocked workflows: any task whose stack stays
            the same across two calls is suspended on an await that never
            resolves.
            """
            import io

            tasks = list(asyncio.all_tasks())
            out = []
            for t in tasks:
                buf = io.StringIO()
                try:
                    name = t.get_name()
                except Exception:
                    name = "?"
                buf.write(f"=== Task {name} done={t.done()} cancelled={t.cancelled()} ===\n")
                try:
                    coro = t.get_coro()
                    buf.write(f"coro: {coro!r}\n")
                except Exception:
                    pass
                try:
                    stack = t.get_stack(limit=30)
                    if stack:
                        for frame in stack:
                            buf.write(
                                f"  {frame.f_code.co_filename}:{frame.f_lineno} in {frame.f_code.co_name}\n"
                            )
                    else:
                        buf.write("  <no stack — task is suspended on a Future/awaitable>\n")
                except Exception as e:
                    buf.write(f"  <stack error: {e}>\n")
                out.append(buf.getvalue())
            return JSONResponse(
                content={"count": len(tasks), "tasks": out},
                media_type="application/json",
            )

        @self.agent.get("/health")
        async def health():
            health_response = {
                "status": "healthy",
                "node_id": self.agent.node_id,
                "version": self.agent.version,
                "timestamp": datetime.now().isoformat(),
            }

            return health_response

        @self.agent.get("/reasoners")
        async def list_reasoners():
            return {"reasoners": self.agent.reasoners}

        @self.agent.get("/skills")
        async def list_skills():
            return {"skills": self.agent.skills}

        @self.agent.post("/shutdown")
        async def shutdown_agent(request: Request):
            """
            Graceful shutdown endpoint for the agent.

            This endpoint allows the AgentField server to request a graceful shutdown
            instead of using process signals.
            """
            try:
                # Parse request body for shutdown options
                body = (
                    await request.json()
                    if request.headers.get("content-type") == "application/json"
                    else {}
                )
                graceful = body.get("graceful", True)
                timeout_seconds = body.get("timeout_seconds", 30)

                if self.agent.dev_mode:
                    log_info(
                        f"Shutdown request received (graceful={graceful}, timeout={timeout_seconds}s)"
                    )

                # Set shutdown status
                from agentfield.agent import AgentStatus

                self.agent._shutdown_requested = True
                self.agent._current_status = AgentStatus.OFFLINE

                # Notify AgentField server of shutdown initiation
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

                # Schedule graceful shutdown
                if graceful:
                    self._track_task(
                        asyncio.create_task(self._graceful_shutdown(timeout_seconds))
                    )

                    return {
                        "status": "shutting_down",
                        "graceful": True,
                        "timeout_seconds": timeout_seconds,
                        "estimated_shutdown_time": datetime.now().isoformat(),
                        "message": "Graceful shutdown initiated",
                    }
                else:
                    # Immediate shutdown
                    self._track_task(asyncio.create_task(self._immediate_shutdown()))

                    return {
                        "status": "shutting_down",
                        "graceful": False,
                        "message": "Immediate shutdown initiated",
                    }

            except Exception as e:
                if self.agent.dev_mode:
                    log_error(f"Shutdown endpoint error: {e}")
                return {
                    "status": "error",
                    "message": f"Failed to initiate shutdown: {str(e)}",
                }

        @self.agent.get("/status")
        async def get_agent_status():
            """
            Get detailed agent status information.

            This endpoint provides comprehensive status information about the agent,
            including uptime, resource usage, and current state.
            """
            try:
                import time

                import psutil

                # Get process info
                process = psutil.Process()

                # Calculate uptime
                start_time = getattr(self.agent, "_start_time", time.time())
                uptime_seconds = time.time() - start_time
                uptime_formatted = self._format_uptime(uptime_seconds)

                status_response = {
                    "status": (
                        "running"
                        if not getattr(self.agent, "_shutdown_requested", False)
                        else "stopping"
                    ),
                    "uptime": uptime_formatted,
                    "uptime_seconds": int(uptime_seconds),
                    "pid": os.getpid(),
                    "version": self.agent.version,
                    "node_id": self.agent.node_id,
                    "last_activity": datetime.now().isoformat(),
                    "resources": {
                        "memory_mb": round(process.memory_info().rss / 1024 / 1024, 2),
                        "cpu_percent": process.cpu_percent(),
                        "threads": process.num_threads(),
                    },
                }

                return status_response

            except ImportError:
                # Fallback if psutil is not available
                return {
                    "status": (
                        "running"
                        if not getattr(self.agent, "_shutdown_requested", False)
                        else "stopping"
                    ),
                    "pid": os.getpid(),
                    "version": self.agent.version,
                    "node_id": self.agent.node_id,
                    "last_activity": datetime.now().isoformat(),
                    "message": "Limited status info (psutil not available)",
                }
            except Exception as e:
                if self.agent.dev_mode:
                    log_error(f"Status endpoint error: {e}")
                return {"status": "error", "message": f"Failed to get status: {str(e)}"}

        @self.agent.get("/info")
        async def node_info():
            return {
                "node_id": self.agent.node_id,
                "version": self.agent.version,
                "base_url": self.agent.base_url,
                "reasoners": self.agent.reasoners,
                "skills": self.agent.skills,
                "registered_at": datetime.now().isoformat(),
            }

        # -----------------------------------------------------------------
        # Approval webhook — receives callbacks from the control plane when
        # an execution's approval state resolves.  Auto-registered so every
        # agent gets this endpoint at ``POST /webhooks/approval``.
        # -----------------------------------------------------------------
        @self.agent.post("/webhooks/approval")
        async def approval_webhook(request: Request):
            """Receive approval resolution callback from the control plane."""
            from agentfield.client import ApprovalResult
            import json as _json

            try:
                body = await request.json()
            except Exception:
                return {"error": "invalid JSON"}, 400

            execution_id = body.get("execution_id", "")
            decision = body.get("decision", "")
            feedback = body.get("feedback", "")
            approval_request_id = body.get("approval_request_id", "")

            if not execution_id or not decision:
                return {"error": "execution_id and decision are required", "status": 400}

            # Parse the raw response field (may be a JSON string or dict)
            raw_response = None
            resp_field = body.get("response")
            if resp_field:
                if isinstance(resp_field, str):
                    try:
                        raw_response = _json.loads(resp_field)
                    except (ValueError, _json.JSONDecodeError):
                        raw_response = {"raw": resp_field}
                elif isinstance(resp_field, dict):
                    raw_response = resp_field

            result = ApprovalResult(
                decision=decision,
                feedback=feedback,
                execution_id=execution_id,
                approval_request_id=approval_request_id,
                raw_response=raw_response,
            )

            # Try to resolve by approval_request_id first, then by execution_id
            resolved = False
            if approval_request_id:
                resolved = await self.agent._pause_manager.resolve(approval_request_id, result)
            if not resolved and execution_id:
                resolved = await self.agent._pause_manager.resolve_by_execution_id(execution_id, result)

            if self.agent.dev_mode:
                log_debug(
                    f"Approval webhook: execution_id={execution_id} "
                    f"decision={decision} resolved={resolved}"
                )

            return {"status": "received", "resolved": resolved}

    async def _graceful_shutdown(self, timeout_seconds: int = 30):
        """
        Perform graceful shutdown with cleanup.

        Args:
            timeout_seconds: Maximum time to wait for graceful shutdown
        """
        try:
            if self.agent.dev_mode:
                log_info(f"Starting graceful shutdown (timeout: {timeout_seconds}s)")

            # Stop heartbeat
            try:
                if (
                    hasattr(self.agent, "agentfield_handler")
                    and self.agent.agentfield_handler
                ):
                    self.agent.agentfield_handler.stop_heartbeat()
                    if self.agent.dev_mode:
                        log_debug("Heartbeat stopped")
            except Exception as e:
                if self.agent.dev_mode:
                    log_error(f"Heartbeat stop error: {e}")

            # Clear agent registry
            try:
                from agentfield.agent_registry import clear_current_agent

                clear_current_agent()
            except Exception as e:
                if self.agent.dev_mode:
                    log_error(f"Registry clear error: {e}")

            # Drain in-flight tasks, then force-cancel anything that misses the deadline.
            tracked_tasks: set[asyncio.Task] = set(self._in_flight_tasks)

            current_task = asyncio.current_task()
            tracked_tasks = {
                task
                for task in tracked_tasks
                if task is not None and task is not current_task and not task.done()
            }

            if tracked_tasks:
                done, pending = await asyncio.wait(
                    tracked_tasks,
                    timeout=max(0, timeout_seconds),
                )
                if self.agent.dev_mode:
                    log_debug(
                        f"Graceful shutdown drain: done={len(done)} pending={len(pending)}"
                    )

                if pending:
                    for task in list(pending):
                        task.cancel()
                    await asyncio.gather(*pending, return_exceptions=True)

            # Clear tracked registries after drain/cancel pass.
            self._in_flight_tasks.clear()

            # Small yield so cancellations/cleanup callbacks run before process exit.
            await asyncio.sleep(0)

            if self.agent.dev_mode:
                log_success("Graceful shutdown completed")

            # Exit the process
            os._exit(0)

        except Exception as e:
            if self.agent.dev_mode:
                log_error(f"Graceful shutdown error: {e}")
            # Fallback to immediate shutdown
            await self._immediate_shutdown()

    async def _immediate_shutdown(self):
        """
        Perform immediate shutdown without cleanup.
        """
        try:
            if self.agent.dev_mode:
                log_warn("Immediate shutdown initiated")

            # Exit immediately
            os._exit(0)

        except Exception as e:
            if self.agent.dev_mode:
                log_error(f"Immediate shutdown error: {e}")
            os._exit(1)

    def _format_uptime(self, uptime_seconds: float) -> str:
        """
        Format uptime seconds into a human-readable string.

        Args:
            uptime_seconds: Uptime in seconds

        Returns:
            Formatted uptime string (e.g., "2h 30m 15s")
        """
        try:
            hours = int(uptime_seconds // 3600)
            minutes = int((uptime_seconds % 3600) // 60)
            seconds = int(uptime_seconds % 60)

            parts = []
            if hours > 0:
                parts.append(f"{hours}h")
            if minutes > 0:
                parts.append(f"{minutes}m")
            if seconds > 0 or not parts:  # Always show seconds if no other parts
                parts.append(f"{seconds}s")

            return " ".join(parts)
        except Exception:
            return f"{int(uptime_seconds)}s"

    def _validate_ssl_config(
        self, ssl_keyfile: Optional[str], ssl_certfile: Optional[str]
    ) -> bool:
        """
        Validate SSL configuration files exist and are readable.

        Args:
            ssl_keyfile: Path to SSL key file
            ssl_certfile: Path to SSL certificate file

        Returns:
            True if SSL configuration is valid, False otherwise
        """
        if not ssl_keyfile or not ssl_certfile:
            return False

        try:
            # Check if files exist and are readable
            if not os.path.isfile(ssl_keyfile):
                if self.agent.dev_mode:
                    log_error(f"SSL key file not found: {ssl_keyfile}")
                return False

            if not os.path.isfile(ssl_certfile):
                if self.agent.dev_mode:
                    log_error(f"SSL certificate file not found: {ssl_certfile}")
                return False

            # Check file permissions
            if not os.access(ssl_keyfile, os.R_OK):
                if self.agent.dev_mode:
                    log_error(f"SSL key file not readable: {ssl_keyfile}")
                return False

            if not os.access(ssl_certfile, os.R_OK):
                if self.agent.dev_mode:
                    log_error(f"SSL certificate file not readable: {ssl_certfile}")
                return False

            return True

        except Exception as e:
            if self.agent.dev_mode:
                log_error(f"SSL validation error: {e}")
            return False

    def _get_optimal_workers(self, workers: Optional[int] = None) -> Optional[int]:
        """
        Determine optimal number of workers based on system resources.

        Args:
            workers: Explicitly requested number of workers

        Returns:
            Optimal number of workers or None for single process
        """
        if workers is not None:
            return workers

        # Check environment variable
        env_workers = os.getenv("UVICORN_WORKERS")
        if env_workers and env_workers.isdigit():
            return int(env_workers)

        # Auto-detect based on CPU cores (only in production)
        try:
            import multiprocessing

            cpu_count = multiprocessing.cpu_count()

            # Use 2 * CPU cores for I/O bound workloads, but cap at 8
            optimal_workers = min(cpu_count * 2, 8)

            if self.agent.dev_mode:
                log_debug(
                    f"Detected {cpu_count} CPU cores, optimal workers: {optimal_workers}"
                )

            return optimal_workers

        except Exception:
            return None

    def _check_performance_dependencies(self) -> dict:
        """
        Check availability of performance-enhancing dependencies.

        Returns:
            Dictionary with availability status of optional dependencies
        """
        deps = {
            "uvloop": False,
            "psutil": False,
            "orjson": False,
        }

        if importlib.util.find_spec("uvloop") is not None:
            deps["uvloop"] = True

        if importlib.util.find_spec("psutil") is not None:
            deps["psutil"] = True

        if importlib.util.find_spec("orjson") is not None:
            deps["orjson"] = True

        return deps

    def setup_signal_handlers(self) -> None:
        """
        Setup signal handlers for graceful shutdown.

        This method registers signal handlers for SIGTERM and SIGINT
        to ensure proper cleanup when the agent shuts down.
        """
        try:
            # Register signal handlers for graceful shutdown
            signal.signal(signal.SIGTERM, self.signal_handler)
            signal.signal(signal.SIGINT, self.signal_handler)

            if self.agent.dev_mode:
                log_debug("Signal handlers registered for graceful shutdown")

        except Exception as e:
            if self.agent.dev_mode:
                log_error(f"Failed to setup signal handlers: {e}")
            # Continue without signal handlers - not critical

    def signal_handler(self, signum: int, frame) -> None:
        """
        Handle shutdown signals gracefully.

        Args:
            signum: Signal number
            frame: Current stack frame
        """
        signal_name = "SIGTERM" if signum == signal.SIGTERM else "SIGINT"

        if self.agent.dev_mode:
            log_warn(f"{signal_name} received, shutting down gracefully...")

        # Exit gracefully
        os._exit(0)

    def serve(
        self,
        port: Optional[int] = None,
        host: str = "0.0.0.0",
        dev: bool = False,
        heartbeat_interval: int = 2,  # Fast heartbeat for real-time detection
        auto_port: bool = False,
        workers: Optional[int] = None,
        ssl_keyfile: Optional[str] = None,
        ssl_certfile: Optional[str] = None,
        log_level: str = "info",
        access_log: bool = True,
        **kwargs,
    ):
        """
        Start the agent node server with intelligent port management and production-ready configuration.

        This method implements smart port resolution that seamlessly works with AgentField CLI
        or standalone execution. The port selection priority is:
        1. Explicit port parameter (highest priority)
        2. PORT environment variable (AgentField CLI integration)
        3. auto_port=True: find free port automatically
        4. Default fallback with availability check

        Args:
            port (int, optional): The port on which the agent server will listen.
                                If specified, this takes highest priority.
            host (str): The host address for the agent server. Defaults to "0.0.0.0".
            dev (bool): If True, enables development mode features (e.g., hot reload, debug UI).
            heartbeat_interval (int): The interval in seconds for sending heartbeats to the AgentField server.
                                      Defaults to 2 seconds (fast detection architecture).
            auto_port (bool): If True, automatically find an available port. Defaults to False.
            workers (int, optional): Number of worker processes for production. If None, uses single process.
            ssl_keyfile (str, optional): Path to SSL key file for HTTPS.
            ssl_certfile (str, optional): Path to SSL certificate file for HTTPS.
            log_level (str): Log level for uvicorn. Defaults to "info".
            access_log (bool): Enable/disable access logging. Defaults to True.
            **kwargs: Additional keyword arguments to pass to `uvicorn.run`.
        """
        # Smart port resolution with priority order
        if port is None:
            # Check for AgentField CLI integration via environment variable
            env_port = os.getenv("PORT")
            if env_port and env_port.isdigit():
                suggested_port = int(env_port)
                if AgentUtils.is_port_available(suggested_port):
                    port = suggested_port
                    if self.agent.dev_mode:
                        log_debug(f"Using port from AgentField CLI: {port}")
                else:
                    # AgentField CLI suggested port is taken, find next available
                    try:
                        port = get_free_port(start_port=suggested_port)
                        if self.agent.dev_mode:
                            log_debug(
                                f"AgentField CLI port {suggested_port} taken, using {port}"
                            )
                    except RuntimeError:
                        port = get_free_port()  # Fallback to default range
                        if self.agent.dev_mode:
                            log_debug(f"Using fallback port: {port}")
            elif auto_port or os.getenv("AGENTFIELD_AUTO_PORT") == "true":
                # Auto-port mode: find any available port
                try:
                    port = get_free_port()
                    if self.agent.dev_mode:
                        log_debug(f"Auto-assigned port: {port}")
                except RuntimeError as e:
                    log_error(f"Failed to find free port: {e}")
                    port = 8001  # Fallback to default
            else:
                # Default behavior: try 8001, find alternative if taken
                if AgentUtils.is_port_available(8001):
                    port = 8001
                else:
                    try:
                        port = get_free_port()
                        if self.agent.dev_mode:
                            log_debug(f"Default port 8001 taken, using {port}")
                    except RuntimeError:
                        port = 8001  # Force use even if taken (will fail gracefully)
        else:
            # Explicit port provided - validate it's available
            if not AgentUtils.is_port_available(port):
                if self.agent.dev_mode:
                    log_warn(f"Requested port {port} is not available")
                # Try to find an alternative near the requested port
                try:
                    alternative_port = get_free_port(start_port=port)
                    if self.agent.dev_mode:
                        log_debug(f"Using alternative port: {alternative_port}")
                    port = alternative_port
                except RuntimeError:
                    if self.agent.dev_mode:
                        log_warn(
                            f"No alternative ports found, attempting to use {port}"
                        )
                    # Continue with original port (will fail if truly unavailable)

        log_info(f"Starting agent node '{self.agent.node_id}' on port {port}")

        # Set base_url for registration - preserve explicit callback URL if set
        if not self.agent.base_url:
            # Check AGENT_CALLBACK_URL environment variable before defaulting to localhost
            env_callback_url = os.getenv("AGENT_CALLBACK_URL")
            if env_callback_url:
                # Parse the environment variable URL to extract the hostname
                try:
                    parsed = urllib.parse.urlparse(env_callback_url)
                    if parsed.hostname:
                        self.agent.base_url = (
                            f"{parsed.scheme or 'http'}://{parsed.hostname}:{port}"
                        )
                        if self.agent.dev_mode:
                            log_debug(
                                f"Using AGENT_CALLBACK_URL from environment: {self.agent.base_url}"
                            )
                    else:
                        # Invalid URL in env var, fall back to localhost
                        self.agent.base_url = f"http://localhost:{port}"
                except Exception:
                    # Failed to parse env var, fall back to localhost
                    self.agent.base_url = f"http://localhost:{port}"
            else:
                # No env var set, use localhost
                self.agent.base_url = f"http://localhost:{port}"
        else:
            # Update port in existing base_url if needed
            parsed = urllib.parse.urlparse(self.agent.base_url)
            if parsed.port != port:
                # Update the port in the existing URL, but preserve the hostname
                self.agent.base_url = f"{parsed.scheme}://{parsed.hostname}:{port}"
                if self.agent.dev_mode:
                    log_debug(f"Updated port in callback URL: {self.agent.base_url}")
            elif self.agent.dev_mode:
                log_debug(f"Using explicit callback URL: {self.agent.base_url}")

        # Start heartbeat worker
        self.agent.agentfield_handler.start_heartbeat(heartbeat_interval)

        log_info(f"Agent server running at http://{host}:{port}")
        log_info("Available endpoints:")
        for route in self.agent.routes:
            # Check if the route is an APIRoute (has .path and .methods)
            if isinstance(route, APIRoute):
                for method in route.methods:
                    if method != "HEAD":  # Skip HEAD methods
                        log_debug(f"Endpoint registered: {method} {route.path}")

        # Setup fast lifecycle signal handlers
        self.agent.agentfield_handler.setup_fast_lifecycle_signal_handlers()

        # Add startup event handler for resilient lifecycle
        @self.agent.on_event("startup")
        async def startup_resilient_lifecycle():
            """Resilient lifecycle startup: connection manager handles AgentField server connectivity"""

            # Initialize connection manager
            from agentfield.connection_manager import (
                ConnectionConfig,
                ConnectionManager,
            )

            # Configure connection manager with reasonable retry interval
            config = ConnectionConfig(
                retry_interval=10.0,  # Check every 10 seconds for AgentField server
                health_check_interval=30.0,
                connection_timeout=10.0,
            )

            self.agent.connection_manager = ConnectionManager(self.agent, config)

            # Set up callbacks for connection state changes
            def on_connected():
                if self.agent.dev_mode:
                    log_info(
                        "Connected to AgentField server - full functionality available"
                    )
                # Kick a heartbeat immediately so the control plane renews the lease
                try:
                    self._track_task(
                        asyncio.create_task(
                            self.agent.agentfield_handler.send_enhanced_heartbeat()
                        )
                    )
                except RuntimeError:
                    # Event loop not running; the heartbeat worker will recover shortly
                    pass
                # Start enhanced heartbeat when connected
                if (
                    not hasattr(self.agent, "_heartbeat_task")
                    or self.agent._heartbeat_task.done()
                ):
                    self.agent._heartbeat_task = self._track_task(
                        asyncio.create_task(
                            self.agent.agentfield_handler.enhanced_heartbeat_loop(
                                heartbeat_interval
                            )
                        )
                    )

            def on_disconnected():
                if self.agent.dev_mode:
                    log_warn("AgentField server disconnected - running in local mode")
                # Cancel heartbeat task when disconnected
                if (
                    hasattr(self.agent, "_heartbeat_task")
                    and not self.agent._heartbeat_task.done()
                ):
                    self.agent._heartbeat_task.cancel()

            self.agent.connection_manager.on_connected = on_connected
            self.agent.connection_manager.on_disconnected = on_disconnected

            # Start connection manager (non-blocking)
            connected = await self.agent.connection_manager.start()

            # Connect memory event client - works independently of AgentField server connection
            if self.agent.memory_event_client:
                try:
                    await self.agent.memory_event_client.connect()
                except Exception as e:
                    if self.agent.dev_mode:
                        log_error(f"Memory event client connection failed: {e}")

            if connected:
                if self.agent.dev_mode:
                    log_info("Agent started with AgentField server connection")
            else:
                if self.agent.dev_mode:
                    log_info(
                        "Agent started in local mode - will connect to AgentField server when available"
                    )

        # Add shutdown event handler for cleanup
        @self.agent.on_event("shutdown")
        async def shutdown_cleanup():
            """Cleanup all resources when FastAPI shuts down"""

            # Stop connection manager
            if self.agent.connection_manager:
                await self.agent.connection_manager.stop()

            # Close memory event client
            if self.agent.memory_event_client:
                await self.agent.memory_event_client.close()

            if getattr(self.agent, "client", None):
                try:
                    await self.agent.client.aclose()
                except Exception as e:
                    if self.agent.dev_mode:
                        log_error(f"AgentField client shutdown error: {e}")

            # Clear agent from thread-local storage during shutdown
            from agentfield.agent_registry import clear_current_agent

            clear_current_agent()

        # Configure uvicorn parameters based on environment and requirements
        uvicorn_config = {
            "host": host,
            "port": port,
            "reload": dev
            and workers is None,  # Only enable reload in dev mode with single worker
            "access_log": access_log,
            "log_level": log_level,
            "ws": "websockets-sansio",
            "timeout_graceful_shutdown": 30,  # Allow 30 seconds for graceful shutdown
            **kwargs,
        }

        # Add SSL configuration if provided and valid
        if ssl_keyfile and ssl_certfile:
            if self._validate_ssl_config(ssl_keyfile, ssl_certfile):
                uvicorn_config.update(
                    {
                        "ssl_keyfile": ssl_keyfile,
                        "ssl_certfile": ssl_certfile,
                    }
                )
                if self.agent.dev_mode:
                    log_info("HTTPS enabled with SSL certificates")
            else:
                log_error("Invalid SSL configuration, falling back to HTTP")
                ssl_keyfile = ssl_certfile = None

        # Configure workers for production
        if workers and workers > 1:
            uvicorn_config["workers"] = workers
            if self.agent.dev_mode:
                log_debug(f"Multi-process mode: {workers} workers")
        elif self.agent.dev_mode:
            log_debug("Single-process mode")

        # Performance optimizations for production
        if not dev:
            # Add production-specific configurations
            production_config = {
                "limit_concurrency": 1000,  # Limit concurrent connections
                "backlog": 2048,  # Connection queue size
            }

            # Only apply request limit for multi-worker deployments
            # Single-process apps don't benefit from this and it causes unwanted shutdowns
            if workers and workers > 1:
                production_config["limit_max_requests"] = (
                    100000  # Restart workers after N requests
                )

            uvicorn_config.update(production_config)

            # Try to use uvloop for better performance
            if importlib.util.find_spec("uvloop") is not None:
                uvicorn_config["loop"] = "uvloop"
                if self.agent.dev_mode:
                    log_info("Using uvloop for enhanced performance")
            elif self.agent.dev_mode:
                log_warn("uvloop not available, using default asyncio loop")

        # Environment-based log level adjustment
        env_log_level = os.getenv("UVICORN_LOG_LEVEL", log_level).lower()
        if env_log_level in ["critical", "error", "warning", "info", "debug", "trace"]:
            uvicorn_config["log_level"] = env_log_level

        # Disable access log in production if not explicitly enabled
        if not dev and "access_log" not in kwargs:
            uvicorn_config["access_log"] = False

        if self.agent.dev_mode:
            log_debug("Uvicorn configuration:")
            config_display = {
                k: v
                for k, v in uvicorn_config.items()
                if k not in ["ssl_keyfile", "ssl_certfile"]
            }
            for key, value in config_display.items():
                log_debug(f"  {key}: {value}")

        try:
            # Start FastAPI server with production-ready configuration
            uvicorn.run(self.agent, **uvicorn_config)
        except OSError as e:
            if "Address already in use" in str(e):
                log_error(
                    f"Port {port} is already in use. Choose a different port or stop the conflicting service."
                )
                if self.agent.dev_mode:
                    log_info(
                        "Try using auto_port=True or set a different port explicitly"
                    )
            else:
                log_error(f"Failed to start server: {e}")
            raise
        except KeyboardInterrupt:
            if self.agent.dev_mode:
                log_info("Server stopped by user (Ctrl+C)")
        except Exception as e:
            log_error(f"Unexpected server error: {e}")
            raise
        finally:
            # Phase 5: Graceful shutdown - stop heartbeat
            if self.agent.dev_mode:
                log_info("Agent shutdown initiated...")

            # Stop heartbeat worker
            self.agent.agentfield_handler.stop_heartbeat()

            if self.agent.dev_mode:
                log_success("Agent shutdown complete")
