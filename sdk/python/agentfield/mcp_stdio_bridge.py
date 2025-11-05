import asyncio
import json
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Dict, Optional

import uvicorn
from fastapi import FastAPI, HTTPException

from .logger import get_logger

logger = get_logger(__name__)


@dataclass
class PendingRequest:
    """Represents a pending request waiting for response"""

    future: asyncio.Future
    timestamp: float


class StdioMCPBridge:
    """
    Bridge that converts stdio-based MCP servers to HTTP endpoints.

    This bridge starts a stdio MCP server process and provides HTTP endpoints
    that translate HTTP requests to JSON-RPC over stdio and back.
    """

    def __init__(self, server_config: dict, port: int, dev_mode: bool = False):
        self.server_config = server_config
        self.port = port
        self.dev_mode = dev_mode

        # Process management
        self.process: Optional[asyncio.subprocess.Process] = None
        self.stdin_writer: Optional[asyncio.StreamWriter] = None
        self.stdout_reader: Optional[asyncio.StreamReader] = None
        self.stderr_reader: Optional[asyncio.StreamReader] = None

        # Request correlation
        self.pending_requests: Dict[str, PendingRequest] = {}
        self.request_timeout = 30.0  # seconds

        # Server state
        self.initialized = False
        self.running = False
        self.app: Optional[FastAPI] = None
        self.server_task: Optional[asyncio.Task] = None
        self.stdio_reader_task: Optional[asyncio.Task] = None

        # Request ID counter for JSON-RPC
        self._request_id_counter = 0

    def _get_next_request_id(self) -> int:
        """Get next request ID for JSON-RPC"""
        self._request_id_counter += 1
        return self._request_id_counter

    async def start(self) -> bool:
        """Start the stdio MCP server and HTTP bridge"""
        try:
            if self.dev_mode:
                logger.debug(
                    f"Starting stdio MCP bridge for {self.server_config.get('alias', 'unknown')} "
                    f"on port {self.port}"
                )

            # Start the stdio MCP server process
            if not await self._start_stdio_process():
                return False

            # Start stdio response reader BEFORE initializing MCP session
            self.running = True
            self.stdio_reader_task = asyncio.create_task(self._read_stdio_responses())

            # Give the reader task a moment to start
            await asyncio.sleep(0.1)

            # Initialize MCP session
            if not await self._initialize_mcp_session():
                await self.stop()
                return False

            # Setup HTTP server
            self._setup_http_server()

            # Start HTTP server
            if self.app is None:
                raise RuntimeError("HTTP server not properly initialized")

            config = uvicorn.Config(
                app=self.app,
                host="localhost",
                port=self.port,
                log_level="error" if not self.dev_mode else "info",
                access_log=self.dev_mode,
            )

            server = uvicorn.Server(config)
            self.server_task = asyncio.create_task(server.serve())

            if self.dev_mode:
                logger.debug(
                    f"Stdio MCP bridge started successfully on port {self.port}"
                )

            return True

        except Exception as e:
            logger.error(f"Failed to start stdio MCP bridge: {e}")
            await self.stop()
            return False

    async def stop(self) -> None:
        """Stop the bridge and cleanup resources"""
        if self.dev_mode:
            logger.debug("Stopping stdio MCP bridge...")

        self.running = False

        # Cancel pending requests
        for request_id, pending in self.pending_requests.items():
            if not pending.future.done():
                pending.future.set_exception(Exception("Bridge shutting down"))
        self.pending_requests.clear()

        # Stop HTTP server
        if self.server_task and not self.server_task.done():
            self.server_task.cancel()
            try:
                await self.server_task
            except asyncio.CancelledError:
                pass

        # Stop stdio reader
        if self.stdio_reader_task and not self.stdio_reader_task.done():
            self.stdio_reader_task.cancel()
            try:
                await self.stdio_reader_task
            except asyncio.CancelledError:
                pass

        # Close stdio streams
        if self.stdin_writer:
            self.stdin_writer.close()
            await self.stdin_writer.wait_closed()

        # Terminate process
        if self.process:
            try:
                self.process.terminate()
                try:
                    await asyncio.wait_for(
                        asyncio.create_task(self._wait_for_process()), timeout=5.0
                    )
                except asyncio.TimeoutError:
                    self.process.kill()
                    await asyncio.create_task(self._wait_for_process())
            except Exception as e:
                logger.error(f"Error stopping process: {e}")

        if self.dev_mode:
            logger.debug("Stdio MCP bridge stopped")

    async def _wait_for_process(self):
        """Wait for process to terminate"""
        if self.process:
            await self.process.wait()

    async def health_check(self) -> bool:
        """Check if bridge and stdio process are healthy"""
        if not self.running or not self.process:
            return False

        # Check if process is still running
        if self.process.returncode is not None:
            return False

        # Try a simple tools/list request to verify communication
        try:
            await asyncio.wait_for(
                self._send_stdio_request("tools/list", {}), timeout=5.0
            )
            return True
        except Exception:
            return False

    async def _start_stdio_process(self) -> bool:
        """Start the stdio MCP server process"""
        try:
            run_command = self.server_config.get("run", "")
            if not run_command:
                raise ValueError("No run command specified in server config")

            working_dir = self.server_config.get("working_dir", ".")
            env = os.environ.copy()
            env.update(self.server_config.get("environment", {}))

            if self.dev_mode:
                logger.debug(f"Starting process: {run_command}")
                logger.debug(f"Working directory: {working_dir}")

            # Start process
            self.process = await asyncio.create_subprocess_shell(
                run_command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=working_dir,
                env=env,
            )

            if (
                self.process.stdin is None
                or self.process.stdout is None
                or self.process.stderr is None
            ):
                raise RuntimeError("Failed to create stdio pipes for process")

            self.stdin_writer = self.process.stdin
            self.stdout_reader = self.process.stdout
            self.stderr_reader = self.process.stderr

            # Give process time to start
            await asyncio.sleep(1.0)

            # Check if process started successfully
            if self.process.returncode is not None:
                stderr_output = ""
                if self.stderr_reader:
                    try:
                        stderr_data = await asyncio.wait_for(
                            self.stderr_reader.read(1024), timeout=1.0
                        )
                        stderr_output = stderr_data.decode("utf-8", errors="ignore")
                    except asyncio.TimeoutError:
                        pass

                raise RuntimeError(
                    f"Process failed to start. Exit code: {self.process.returncode}. Stderr: {stderr_output}"
                )

            return True

        except Exception as e:
            logger.error(f"Failed to start stdio process: {e}")
            return False

    async def _initialize_mcp_session(self) -> bool:
        """Initialize MCP session with handshake"""
        try:
            if self.dev_mode:
                logger.debug("Initializing MCP session...")

            # Send initialize request
            init_params = {
                "protocolVersion": "2024-11-05",
                "capabilities": {"roots": {"listChanged": True}},
                "clientInfo": {"name": "agentfield-stdio-bridge", "version": "1.0.0"},
            }

            response = await self._send_stdio_request("initialize", init_params)

            if "error" in response:
                raise RuntimeError(f"Initialize failed: {response['error']}")

            # Send initialized notification (no response expected)
            await self._send_stdio_notification("notifications/initialized", {})

            self.initialized = True

            if self.dev_mode:
                logger.debug("MCP session initialized successfully")

            return True

        except Exception as e:
            logger.error(f"Failed to initialize MCP session: {e}")
            return False

    def _setup_http_server(self) -> None:
        """Setup FastAPI HTTP server with MCP endpoints"""

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # Startup
            yield
            # Shutdown
            await self.stop()

        self.app = FastAPI(
            title="MCP Stdio Bridge",
            description="HTTP bridge for stdio-based MCP servers",
            lifespan=lifespan,
        )

        @self.app.get("/health")
        async def health_endpoint():
            """Health check endpoint"""
            is_healthy = await self.health_check()
            if is_healthy:
                return {"status": "healthy", "bridge": "running", "process": "running"}
            else:
                raise HTTPException(
                    status_code=503, detail="Bridge or process not healthy"
                )

        @self.app.post("/mcp/tools/list")
        async def list_tools_endpoint():
            """List available tools from stdio MCP server"""
            try:
                response = await self._handle_list_tools({})
                return response
            except Exception as e:
                logger.error(f"Error listing tools: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/mcp/tools/call")
        async def call_tool_endpoint(request: dict):
            """Call a specific tool on stdio MCP server"""
            try:
                response = await self._handle_call_tool(request)
                return response
            except Exception as e:
                logger.error(f"Error calling tool: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        # Also support the standard MCP v1 endpoint format
        @self.app.post("/mcp/v1")
        async def mcp_v1_endpoint(request: dict):
            """Standard MCP v1 JSON-RPC endpoint"""
            try:
                method = request.get("method", "")
                params = request.get("params", {})

                if method == "tools/list":
                    result = await self._handle_list_tools(params)
                    return {
                        "jsonrpc": "2.0",
                        "id": request.get("id", 1),
                        "result": result,
                    }
                elif method == "tools/call":
                    result = await self._handle_call_tool(params)
                    return {
                        "jsonrpc": "2.0",
                        "id": request.get("id", 1),
                        "result": result,
                    }
                else:
                    raise HTTPException(
                        status_code=400, detail=f"Unsupported method: {method}"
                    )

            except Exception as e:
                logger.error(f"Error in MCP v1 endpoint: {e}")
                return {
                    "jsonrpc": "2.0",
                    "id": request.get("id", 1),
                    "error": {"code": -32603, "message": str(e)},
                }

    async def _handle_list_tools(self, request: dict) -> dict:
        """Handle tools/list request"""
        try:
            response = await self._send_stdio_request("tools/list", {})

            if "error" in response:
                raise RuntimeError(f"Tools list failed: {response['error']}")

            result = response.get("result", {})
            tools = result.get("tools", [])

            return {"tools": tools}

        except Exception as e:
            logger.error(f"Failed to list tools: {e}")
            raise

    async def _handle_call_tool(self, request: dict) -> dict:
        """Handle tools/call request"""
        try:
            tool_name = request.get("name")
            arguments = request.get("arguments", {})

            if not tool_name:
                raise ValueError("Tool name is required")

            params = {"name": tool_name, "arguments": arguments}

            response = await self._send_stdio_request("tools/call", params)

            if "error" in response:
                raise RuntimeError(f"Tool call failed: {response['error']}")

            return response.get("result", {})

        except Exception as e:
            logger.error(f"Failed to call tool: {e}")
            raise

    async def _send_stdio_request(self, method: str, params: dict) -> dict:
        """Send JSON-RPC request to stdio process and wait for response"""
        if not self.stdin_writer:
            raise RuntimeError("Stdio process not initialized")

        request_id = self._get_next_request_id()

        request = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params,
        }

        # Create future for response
        future = asyncio.Future()
        self.pending_requests[str(request_id)] = PendingRequest(
            future=future, timestamp=asyncio.get_event_loop().time()
        )

        try:
            # Send request
            request_json = json.dumps(request) + "\n"
            self.stdin_writer.write(request_json.encode("utf-8"))
            await self.stdin_writer.drain()

            if self.dev_mode:
                logger.debug(f"Sent request: {method} (id: {request_id})")

            # Wait for response with timeout
            response = await asyncio.wait_for(future, timeout=self.request_timeout)
            return response

        except asyncio.TimeoutError:
            # Clean up pending request
            self.pending_requests.pop(str(request_id), None)
            raise RuntimeError(f"Request timeout for {method}")
        except Exception as e:
            # Clean up pending request
            self.pending_requests.pop(str(request_id), None)
            raise RuntimeError(f"Request failed for {method}: {e}")

    async def _send_stdio_notification(self, method: str, params: dict) -> None:
        """Send JSON-RPC notification to stdio process (no response expected)"""
        if not self.stdin_writer:
            raise RuntimeError("Stdio process not initialized")

        notification = {"jsonrpc": "2.0", "method": method, "params": params}

        notification_json = json.dumps(notification) + "\n"
        self.stdin_writer.write(notification_json.encode("utf-8"))
        await self.stdin_writer.drain()

        if self.dev_mode:
            logger.debug(f"Sent notification: {method}")

    async def _read_stdio_responses(self) -> None:
        """Read responses from stdio process and correlate with pending requests"""
        if not self.stdout_reader:
            return

        try:
            while self.running:
                try:
                    # Read line from stdout
                    line = await asyncio.wait_for(
                        self.stdout_reader.readline(), timeout=1.0
                    )

                    if not line:
                        # EOF reached
                        break

                    line_str = line.decode("utf-8").strip()
                    if not line_str:
                        continue

                    # Parse JSON response
                    try:
                        response = json.loads(line_str)
                    except json.JSONDecodeError:
                        if self.dev_mode:
                            logger.warning(
                                f"Failed to parse JSON response: {line_str[:100]}..."
                            )
                        continue

                    # Handle response
                    await self._handle_stdio_response(response)

                except asyncio.TimeoutError:
                    # Check for expired requests
                    await self._cleanup_expired_requests()
                    continue
                except Exception as e:
                    if self.running:
                        logger.error(f"Error reading stdio response: {e}")
                    break

        except Exception as e:
            if self.running:
                logger.error(f"Stdio reader task failed: {e}")
        finally:
            # Cancel all pending requests
            for pending in self.pending_requests.values():
                if not pending.future.done():
                    pending.future.set_exception(Exception("Stdio reader stopped"))
            self.pending_requests.clear()

    async def _handle_stdio_response(self, response: dict) -> None:
        """Handle a response from stdio process"""
        response_id = response.get("id")

        if response_id is None:
            # This might be a notification, ignore
            return

        request_id = str(response_id)
        pending = self.pending_requests.pop(request_id, None)

        if pending and not pending.future.done():
            pending.future.set_result(response)

            if self.dev_mode:
                logger.debug(f"Received response for request {request_id}")
        elif self.dev_mode:
            logger.warning(f"Received response for unknown request {request_id}")

    async def _cleanup_expired_requests(self) -> None:
        """Clean up expired pending requests"""
        current_time = asyncio.get_event_loop().time()
        expired_ids = []

        for request_id, pending in self.pending_requests.items():
            if current_time - pending.timestamp > self.request_timeout:
                expired_ids.append(request_id)
                if not pending.future.done():
                    pending.future.set_exception(
                        asyncio.TimeoutError("Request expired")
                    )

        for request_id in expired_ids:
            self.pending_requests.pop(request_id, None)

        if expired_ids and self.dev_mode:
            logger.warning(f"Cleaned up {len(expired_ids)} expired requests")
