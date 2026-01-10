"""
Async Execution Manager for the AgentField SDK.

This module provides the central orchestrator for managing hundreds of concurrent
async executions with intelligent polling, resource management, and comprehensive
monitoring capabilities.
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urljoin

import aiohttp

from .async_config import AsyncConfig
from .execution_state import ExecutionPriority, ExecutionState, ExecutionStatus
from .http_connection_manager import ConnectionManager
from .logger import get_logger
from .result_cache import ResultCache
from .status import normalize_status
from .types import WebhookConfig

logger = get_logger(__name__)


class LazyAsyncLock:
    """Deferred asyncio.Lock that instantiates once the event loop is running."""

    def __init__(self):
        self._lock: Optional[asyncio.Lock] = None

    def _lock_obj(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def __aenter__(self):
        return await self._lock_obj().__aenter__()

    async def __aexit__(self, exc_type, exc, tb):
        return await self._lock_obj().__aexit__(exc_type, exc, tb)


class LazySemaphore:
    """Deferred asyncio.Semaphore that instantiates within the active loop."""

    def __init__(self, size_factory):
        self._size_factory = size_factory
        self._sem: Optional[asyncio.Semaphore] = None

    def _sem_obj(self) -> asyncio.Semaphore:
        if self._sem is None:
            self._sem = asyncio.Semaphore(max(1, int(self._size_factory())))
        return self._sem

    async def acquire(self):
        return await self._sem_obj().acquire()

    def release(self):
        self._sem_obj().release()

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self.release()


@dataclass
class PollingMetrics:
    """Metrics for polling performance monitoring."""

    total_polls: int = 0
    successful_polls: int = 0
    failed_polls: int = 0
    timeout_polls: int = 0
    batch_polls: int = 0
    average_poll_duration: float = 0.0
    last_poll_time: float = field(default_factory=time.time)

    @property
    def success_rate(self) -> float:
        """Calculate polling success rate as a percentage."""
        if self.total_polls == 0:
            return 0.0
        return (self.successful_polls / self.total_polls) * 100

    def record_poll(
        self, success: bool, duration: float, timeout: bool = False
    ) -> None:
        """Record a polling operation."""
        self.total_polls += 1
        self.last_poll_time = time.time()

        if success:
            self.successful_polls += 1
        else:
            self.failed_polls += 1
            if timeout:
                self.timeout_polls += 1

        # Update average duration using exponential moving average
        alpha = 0.1  # Smoothing factor
        self.average_poll_duration = (
            alpha * duration + (1 - alpha) * self.average_poll_duration
        )


@dataclass
class ExecutionManagerMetrics:
    """Comprehensive metrics for the execution manager."""

    # Execution counts
    total_executions: int = 0
    active_executions: int = 0
    completed_executions: int = 0
    failed_executions: int = 0
    cancelled_executions: int = 0
    timeout_executions: int = 0

    # Performance metrics
    average_execution_time: float = 0.0
    average_queue_time: float = 0.0
    peak_concurrent_executions: int = 0

    # Resource metrics
    memory_usage_mb: float = 0.0
    cleanup_operations: int = 0

    # Polling metrics
    polling_metrics: PollingMetrics = field(default_factory=PollingMetrics)

    # Timestamps
    created_at: float = field(default_factory=time.time)
    last_cleanup: float = field(default_factory=time.time)

    @property
    def uptime(self) -> float:
        """Get manager uptime in seconds."""
        return time.time() - self.created_at

    @property
    def success_rate(self) -> float:
        """Calculate execution success rate as a percentage."""
        total_completed = (
            self.completed_executions
            + self.failed_executions
            + self.cancelled_executions
            + self.timeout_executions
        )
        if total_completed == 0:
            return 0.0
        return (self.completed_executions / total_completed) * 100


class AsyncExecutionManager:
    """
    Central orchestrator for managing hundreds of concurrent async executions.

    This class provides:
    - Concurrent execution tracking with ExecutionState objects
    - Intelligent polling with adaptive intervals based on execution age
    - Resource management with cleanup of completed executions
    - Background polling task coordination using asyncio
    - Thread-safe operations for concurrent access
    - Comprehensive metrics and monitoring
    - Integration with ConnectionManager and ResultCache
    """

    def __init__(
        self,
        base_url: str,
        config: Optional[AsyncConfig] = None,
        connection_manager: Optional[ConnectionManager] = None,
        result_cache: Optional[ResultCache] = None,
    ):
        """
        Initialize the async execution manager.

        Args:
            base_url: Base URL for the af server
            config: AsyncConfig instance for configuration parameters
            connection_manager: Optional ConnectionManager instance
            result_cache: Optional ResultCache instance
        """
        self.base_url = base_url.rstrip("/")
        self.config = config or AsyncConfig()

        # Validate configuration
        self.config.validate()

        # Initialize components
        self.connection_manager = connection_manager or ConnectionManager(self.config)
        self.result_cache = result_cache or ResultCache(self.config)

        # Execution tracking
        self._executions: Dict[str, ExecutionState] = {}
        self._execution_lock = LazyAsyncLock()
        self._capacity_semaphore = LazySemaphore(
            lambda: self.config.max_concurrent_executions
        )

        # Event stream configuration
        self._event_stream_headers: Dict[str, str] = {}

        # Polling coordination
        self._polling_task: Optional[asyncio.Task] = None
        self._polling_semaphore = LazySemaphore(
            lambda: self.config.max_active_polls
        )
        self._shutdown_event: Optional[asyncio.Event] = None

        # Metrics and monitoring
        self.metrics = ExecutionManagerMetrics()

        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._metrics_task: Optional[asyncio.Task] = None
        self._event_stream_task: Optional[asyncio.Task] = None

        # Circuit breaker state
        self._circuit_breaker_failures = 0
        self._circuit_breaker_last_failure = 0.0
        self._circuit_breaker_open = False

        logger.debug(f"AsyncExecutionManager initialized with base_url={base_url}")

    def set_event_stream_headers(self, headers: Optional[Dict[str, str]]) -> None:
        """Configure headers forwarded to the SSE event stream."""

        if headers is None:
            self._event_stream_headers = {}
            return

        self._event_stream_headers = {
            key: value for key, value in headers.items() if value is not None
        }

    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()

    async def start(self) -> None:
        """
        Start the execution manager and all background tasks.

        Raises:
            RuntimeError: If manager is already started
        """
        if self._polling_task is not None:
            raise RuntimeError("AsyncExecutionManager is already started")

        # Start components
        await self.connection_manager.start()
        await self.result_cache.start()

        if self._shutdown_event is None:
            self._shutdown_event = asyncio.Event()
        self._shutdown_event.clear()

        # Start background tasks
        self._polling_task = asyncio.create_task(self._polling_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

        if self.config.enable_performance_logging:
            self._metrics_task = asyncio.create_task(self._metrics_loop())

        if self.config.enable_event_stream:
            self._event_stream_task = asyncio.create_task(self._event_stream_loop())

        logger.info(
            f"AsyncExecutionManager started with max_concurrent={self.config.max_concurrent_executions}"
        )

    async def stop(self) -> None:
        """
        Stop the execution manager and cleanup all resources.
        """
        logger.info("Stopping AsyncExecutionManager...")

        # Signal shutdown
        if self._shutdown_event is None:
            self._shutdown_event = asyncio.Event()
        self._shutdown_event.set()

        # Cancel background tasks
        tasks_to_cancel = [
            self._polling_task,
            self._cleanup_task,
            self._metrics_task,
            self._event_stream_task,
        ]

        for task in tasks_to_cancel:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        self._polling_task = None
        self._cleanup_task = None
        self._metrics_task = None
        self._event_stream_task = None

        # Cancel all active executions
        async with self._execution_lock:
            for execution in self._executions.values():
                if execution.is_active:
                    execution.cancel("Manager shutdown")
                    self._release_capacity_for_execution(execution)

        # Stop components
        await self.connection_manager.close()
        await self.result_cache.stop()

        logger.info("AsyncExecutionManager stopped")

    async def submit_execution(
        self,
        target: str,
        input_data: Dict[str, Any],
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
        priority: ExecutionPriority = ExecutionPriority.NORMAL,
        webhook: Optional[Union[WebhookConfig, Dict[str, Any]]] = None,
    ) -> str:
        """
        Submit an async execution and return execution_id.

        Args:
            target: Target endpoint for execution
            input_data: Input data for the execution
            headers: Optional HTTP headers
            timeout: Optional execution timeout (uses config default if None)
            priority: Execution priority for queue management

        Returns:
            str: Execution ID for tracking the execution

        Raises:
            RuntimeError: If manager is not started or at capacity
            aiohttp.ClientError: For HTTP-related errors
        """
        if self._polling_task is None:
            raise RuntimeError("AsyncExecutionManager is not started")

        # Check circuit breaker
        if self._is_circuit_breaker_open():
            raise RuntimeError("Circuit breaker is open - too many recent failures")

        # Reserve capacity slot; released once terminal
        await self._capacity_semaphore.acquire()

        # Prepare request
        url = urljoin(self.base_url, f"/api/v1/execute/async/{target}")
        request_headers = {"Content-Type": "application/json", **(headers or {})}
        payload: Dict[str, Any] = {
            "input": input_data,
        }

        if webhook:
            if isinstance(webhook, WebhookConfig):
                payload["webhook"] = webhook.to_payload()
            elif isinstance(webhook, dict):
                payload["webhook"] = webhook
            else:
                raise TypeError("webhook must be a WebhookConfig or dict")

        # Set timeout
        execution_timeout = timeout or self.config.default_execution_timeout

        try:
            # Submit execution
            start_time = time.time()
            async with self.connection_manager.get_session() as session:
                response = await session.post(
                    url,
                    json=payload,
                    headers=request_headers,
                    timeout=self.config.polling_timeout,
                )
                response.raise_for_status()
                result = await response.json()

            execution_id = result.get("execution_id")
            if not execution_id:
                raise ValueError("Server did not return execution_id")

            workflow_id = result.get("workflow_id") or result.get("run_id")
            status = self._map_execution_status(result.get("status"))
            created_at = self._parse_timestamp(result.get("created_at"))
            webhook_registered = bool(result.get("webhook_registered"))
            webhook_error = result.get("webhook_error")

            if webhook and not webhook_registered and webhook_error:
                logger.warning(
                    "Webhook registration rejected for %s: %s",
                    target,
                    webhook_error,
                )

            # Create execution state
            execution_state = ExecutionState(
                execution_id=execution_id,
                target=target,
                input_data=input_data,
                status=status,
                priority=priority,
                timeout=execution_timeout,
                workflow_id=workflow_id,
                created_at=created_at or datetime.now(timezone.utc),
                updated_at=created_at or datetime.now(timezone.utc),
                webhook_registered=webhook_registered,
                webhook_error=webhook_error,
            )

            # Store execution
            async with self._execution_lock:
                self._executions[execution_id] = execution_state
                self.metrics.total_executions += 1
                self.metrics.active_executions += 1

                # Update peak concurrent executions
                if (
                    self.metrics.active_executions
                    > self.metrics.peak_concurrent_executions
                ):
                    self.metrics.peak_concurrent_executions = (
                        self.metrics.active_executions
                    )

            # Reset circuit breaker on success
            self._circuit_breaker_failures = 0

            duration = time.time() - start_time
            logger.debug(
                f"Submitted execution {execution_id[:8]}... for target {target} in {duration:.3f}s"
            )

            return execution_id

        except Exception as e:
            self._capacity_semaphore.release()
            self._record_circuit_breaker_failure()
            logger.error(f"Failed to submit execution for target {target}: {e}")
            raise

    def _map_execution_status(self, status: Optional[str]) -> ExecutionStatus:
        if not status:
            return ExecutionStatus.QUEUED
        normalized = status.lower()
        if normalized in ExecutionStatus._value2member_map_:
            return ExecutionStatus._value2member_map_[normalized]
        return ExecutionStatus.QUEUED

    @staticmethod
    def _parse_timestamp(value: Optional[str]) -> Optional[datetime]:
        if not value:
            return None
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None

    async def wait_for_result(
        self, execution_id: str, timeout: Optional[float] = None
    ) -> Any:
        """
        Wait for execution result with intelligent polling.

        Args:
            execution_id: Execution ID to wait for
            timeout: Optional timeout override

        Returns:
            Any: Execution result

        Raises:
            KeyError: If execution_id is not found
            TimeoutError: If execution times out
            RuntimeError: If execution fails or is cancelled
        """
        # Check cache first
        cached_result = self.result_cache.get_execution_result(execution_id)
        if cached_result is not None:
            logger.debug(f"Retrieved cached result for execution {execution_id[:8]}...")
            return cached_result

        # Get execution state
        async with self._execution_lock:
            execution = self._executions.get(execution_id)
            if execution is None:
                raise KeyError(f"Execution {execution_id} not found")

        # Set timeout
        wait_timeout = (
            timeout or execution.timeout or self.config.default_execution_timeout
        )
        start_time = time.time()

        # Wait for completion
        while time.time() - start_time < wait_timeout:
            async with self._execution_lock:
                execution = self._executions.get(execution_id)
                if execution is None:
                    raise KeyError(f"Execution {execution_id} was removed")

                if execution.is_terminal:
                    if execution.is_successful:
                        # Cache successful result
                        if execution.result is not None:
                            self.result_cache.set_execution_result(
                                execution_id, execution.result
                            )
                        return execution.result
                    elif execution.status == ExecutionStatus.FAILED:
                        raise RuntimeError(
                            f"Execution failed: {execution.error_message}"
                        )
                    elif execution.status == ExecutionStatus.CANCELLED:
                        raise RuntimeError(
                            f"Execution was cancelled: {execution._cancellation_reason}"
                        )
                    elif execution.status == ExecutionStatus.TIMEOUT:
                        raise TimeoutError(
                            f"Execution timed out after {execution.timeout} seconds"
                        )

            # Wait before next check
            await asyncio.sleep(0.1)

        # Timeout reached
        async with self._execution_lock:
            execution = self._executions.get(execution_id)
            if execution and execution.is_active:
                execution.timeout_execution()
                self.metrics.timeout_executions += 1

        raise TimeoutError(f"Wait timeout reached after {wait_timeout} seconds")

    async def cancel_execution(
        self, execution_id: str, reason: Optional[str] = None
    ) -> bool:
        """
        Cancel an active execution.

        Args:
            execution_id: Execution ID to cancel
            reason: Optional cancellation reason

        Returns:
            bool: True if execution was cancelled, False if not found or already terminal
        """
        async with self._execution_lock:
            execution = self._executions.get(execution_id)
            if execution is None or execution.is_terminal:
                return False

            execution.cancel(reason)
            self.metrics.cancelled_executions += 1
            self.metrics.active_executions -= 1

            logger.debug(
                f"Cancelled execution {execution_id[:8]}... - {reason or 'No reason provided'}"
            )
            return True

    async def get_execution_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """
        Get current status of an execution.

        Args:
            execution_id: Execution ID to check

        Returns:
            Optional[Dict]: Execution status dictionary or None if not found
        """
        async with self._execution_lock:
            execution = self._executions.get(execution_id)
            if execution is None:
                return None

            return execution.to_dict()

    async def list_executions(
        self,
        status_filter: Optional[ExecutionStatus] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        List executions with optional filtering.

        Args:
            status_filter: Optional status to filter by
            limit: Optional limit on number of results

        Returns:
            List[Dict]: List of execution status dictionaries
        """
        async with self._execution_lock:
            executions = list(self._executions.values())

            # Apply status filter
            if status_filter:
                executions = [e for e in executions if e.status == status_filter]

            # Sort by creation time (newest first)
            executions.sort(key=lambda e: e.created_at, reverse=True)

            # Apply limit
            if limit:
                executions = executions[:limit]

            return [execution.to_dict() for execution in executions]

    async def cleanup_completed_executions(self) -> int:
        """
        Clean up completed executions to manage memory.

        Returns:
            int: Number of executions cleaned up
        """
        cleanup_count = 0
        current_time = time.time()

        async with self._execution_lock:
            # Collect terminal executions for retention analysis
            completed_executions = {
                exec_id: execution
                for exec_id, execution in self._executions.items()
                if execution.is_terminal
            }

            if not completed_executions:
                return 0

            removal_candidates = set()

            # Time-based pruning to keep memory bounded during long-running sessions
            retention_seconds = self.config.completed_execution_retention_seconds
            if retention_seconds > 0:
                for exec_id, execution in completed_executions.items():
                    end_time = (
                        execution.metrics.end_time or execution.metrics.submit_time
                    )
                    if end_time and (current_time - end_time) > retention_seconds:
                        removal_candidates.add(exec_id)

            # Enforce cap on stored completions after time-based pruning
            remaining = [
                (exec_id, execution)
                for exec_id, execution in completed_executions.items()
                if exec_id not in removal_candidates
            ]

            if len(remaining) > self.config.max_completed_executions:
                # Remove the oldest executions first
                remaining.sort(key=lambda item: item[1].metrics.end_time or 0)
                overflow = len(remaining) - self.config.max_completed_executions
                for i in range(overflow):
                    removal_candidates.add(remaining[i][0])

            # Apply removals and cache results where applicable
            for exec_id in removal_candidates:
                execution = completed_executions.get(exec_id)
                if execution is None:
                    continue

                if execution.is_successful and execution.result is not None:
                    self.result_cache.set_execution_result(exec_id, execution.result)

                self._release_capacity_for_execution(execution)
                self._executions.pop(exec_id, None)
                cleanup_count += 1

        if cleanup_count > 0:
            self.metrics.cleanup_operations += 1
            self.metrics.last_cleanup = current_time
            logger.debug(f"Cleaned up {cleanup_count} completed executions")

        return cleanup_count

    async def _event_stream_loop(self) -> None:
        """Listen for execution events over SSE and nudge polling."""
        logger.debug("Starting event stream loop")

        url = urljoin(self.base_url, self.config.event_stream_path)
        backoff = max(self.config.event_stream_retry_backoff, 0.5)

        while not self._shutdown_event.is_set():
            try:
                request_headers = {"Accept": "text/event-stream"}
                if self._event_stream_headers:
                    request_headers.update(self._event_stream_headers)

                async with self.connection_manager.get_session() as session:
                    timeout = aiohttp.ClientTimeout(total=None, sock_read=None)
                    async with session.get(
                        url, headers=request_headers, timeout=timeout
                    ) as response:
                        if response.status != 200:
                            body = await response.text()
                            logger.warn(
                                f"Event stream returned {response.status} for {url}: {body[:256]}"
                            )
                            await asyncio.sleep(backoff)
                            continue

                        buffer = ""
                        async for chunk in response.content.iter_chunked(1024):
                            if self._shutdown_event.is_set():
                                break
                            if not chunk:
                                continue
                            try:
                                decoded = chunk.decode("utf-8", errors="ignore")
                            except Exception:
                                continue

                            buffer += decoded

                            # Prevent unbounded buffer growth (1MB limit)
                            if len(buffer) > 1024 * 1024:
                                logger.warn(
                                    "SSE buffer exceeded 1MB limit, clearing to prevent memory leak"
                                )
                                buffer = ""
                                continue

                            while "\n\n" in buffer:
                                raw_event, buffer = buffer.split("\n\n", 1)
                                data_lines = []
                                for line in raw_event.splitlines():
                                    if line.startswith(":"):
                                        continue
                                    if line.startswith("data:"):
                                        data_lines.append(line[5:].lstrip())

                                if not data_lines:
                                    continue

                                payload_str = "\n".join(data_lines).strip()
                                if not payload_str:
                                    continue

                                try:
                                    payload = json.loads(payload_str)
                                except json.JSONDecodeError:
                                    logger.debug(
                                        f"Failed to decode SSE payload: {payload_str[:120]}"
                                    )
                                    continue

                                await self._handle_event_stream_payload(payload)

            except asyncio.CancelledError:
                break
            except Exception as e:
                if self._shutdown_event.is_set():
                    break
                logger.warn(f"Event stream error: {e}")
                await asyncio.sleep(backoff)

        logger.debug("Event stream loop stopped")

    async def _handle_event_stream_payload(self, payload: Dict[str, Any]) -> None:
        """Process a single SSE payload."""
        execution_id = payload.get("execution_id") or payload.get("executionId")
        if not execution_id:
            return

        schedule_poll = False
        status_hint = normalize_status(payload.get("status"))
        event_type = str(payload.get("type", "")).lower()

        async with self._execution_lock:
            execution = self._executions.get(execution_id)
            if execution is None:
                return

            if event_type == "execution_started" or status_hint == "running":
                execution.update_status(ExecutionStatus.RUNNING)
            elif status_hint == "queued":
                execution.update_status(ExecutionStatus.QUEUED)
            elif status_hint == "pending":
                execution.update_status(ExecutionStatus.PENDING)
            elif status_hint in {
                "succeeded",
                "failed",
                "cancelled",
                "timeout",
            } or event_type in {"execution_completed", "execution_failed"}:
                if status_hint == "failed":
                    execution.update_status(ExecutionStatus.FAILED)
                elif status_hint == "cancelled":
                    execution.update_status(ExecutionStatus.CANCELLED)
                elif status_hint == "timeout":
                    execution.update_status(ExecutionStatus.TIMEOUT)
                else:
                    execution.update_status(ExecutionStatus.SUCCEEDED)
                schedule_poll = True

        if schedule_poll:
            asyncio.create_task(self._poll_execution_immediate(execution_id))

    async def _poll_execution_immediate(self, execution_id: str) -> None:
        """Trigger an immediate poll for the provided execution."""
        async with self._execution_lock:
            execution = self._executions.get(execution_id)

        if execution is None:
            return

        if execution.is_terminal and execution.result is not None:
            return

        try:
            await self._poll_single_execution(execution)
        except Exception as exc:
            logger.debug(f"Immediate poll for {execution_id[:8]}... failed: {exc}")

    async def start_polling_task(self) -> None:
        """
        Start the background polling task.

        Note: This is automatically called by start() and should not be called manually.
        """
        if self._polling_task is None or self._polling_task.done():
            self._polling_task = asyncio.create_task(self._polling_loop())
            logger.debug("Background polling task started")

    async def stop_polling_task(self) -> None:
        """
        Stop the background polling task.

        Note: This is automatically called by stop() and should not be called manually.
        """
        if self._polling_task:
            self._polling_task.cancel()
            try:
                await self._polling_task
            except asyncio.CancelledError:
                pass
            self._polling_task = None
            logger.debug("Background polling task stopped")

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive execution manager metrics.

        Returns:
            Dict[str, Any]: Metrics dictionary
        """

        # Update current metrics
        async def _update_metrics():
            async with self._execution_lock:
                active_count = sum(1 for e in self._executions.values() if e.is_active)
                self.metrics.active_executions = active_count

        # Run the update if we're in an async context
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(_update_metrics())
        except RuntimeError:
            pass  # Not in async context

        return {
            "total_executions": self.metrics.total_executions,
            "active_executions": self.metrics.active_executions,
            "completed_executions": self.metrics.completed_executions,
            "failed_executions": self.metrics.failed_executions,
            "cancelled_executions": self.metrics.cancelled_executions,
            "timeout_executions": self.metrics.timeout_executions,
            "success_rate": self.metrics.success_rate,
            "average_execution_time": self.metrics.average_execution_time,
            "average_queue_time": self.metrics.average_queue_time,
            "peak_concurrent_executions": self.metrics.peak_concurrent_executions,
            "memory_usage_mb": self.metrics.memory_usage_mb,
            "cleanup_operations": self.metrics.cleanup_operations,
            "uptime": self.metrics.uptime,
            "polling_metrics": {
                "total_polls": self.metrics.polling_metrics.total_polls,
                "successful_polls": self.metrics.polling_metrics.successful_polls,
                "failed_polls": self.metrics.polling_metrics.failed_polls,
                "success_rate": self.metrics.polling_metrics.success_rate,
                "average_poll_duration": self.metrics.polling_metrics.average_poll_duration,
                "batch_polls": self.metrics.polling_metrics.batch_polls,
            },
            "circuit_breaker": {
                "failures": self._circuit_breaker_failures,
                "is_open": self._circuit_breaker_open,
                "last_failure": self._circuit_breaker_last_failure,
            },
            "connection_manager": self.connection_manager.get_metrics().__dict__,
            "result_cache": self.result_cache.get_stats(),
        }

    async def _polling_loop(self) -> None:
        """Background task for intelligent polling of active executions."""
        logger.debug("Starting polling loop")

        while not self._shutdown_event.is_set():
            try:
                await self._poll_active_executions()
                await asyncio.sleep(self.config.batch_poll_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Polling loop error: {e}")
                await asyncio.sleep(1.0)  # Brief pause on error

        logger.debug("Polling loop stopped")

    async def _poll_active_executions(self) -> None:
        """Poll all active executions that are ready for polling."""
        # Get executions ready for polling
        executions_to_poll = []

        async with self._execution_lock:
            for execution in self._executions.values():
                if execution.should_poll:
                    # Check for timeout
                    if execution.is_overdue:
                        execution.timeout_execution()
                        self.metrics.timeout_executions += 1
                        self.metrics.active_executions -= 1
                        continue

                    executions_to_poll.append(execution)

        if not executions_to_poll:
            return

        # Use batch polling if enabled and beneficial
        if (
            self.config.enable_batch_polling and len(executions_to_poll) >= 3
        ):  # Batch threshold
            await self._batch_poll_executions(executions_to_poll)
        else:
            await self._individual_poll_executions(executions_to_poll)

    async def _batch_poll_executions(self, executions: List[ExecutionState]) -> None:
        """Poll multiple executions in batches for efficiency."""
        # Split into batches
        batch_size = min(self.config.batch_size, len(executions))

        for i in range(0, len(executions), batch_size):
            batch = executions[i : i + batch_size]

            # Create batch requests
            requests = []
            for execution in batch:
                requests.append(
                    {
                        "method": "GET",
                        "url": self._execution_status_url(execution.execution_id),
                        "timeout": self.config.polling_timeout,
                    }
                )

            # Execute batch
            start_time = time.time()
            try:
                responses = await self.connection_manager.batch_request(requests)
                duration = time.time() - start_time

                # Process responses
                for execution, response in zip(batch, responses):
                    await self._process_poll_response(
                        execution, response, duration / len(batch)
                    )

                self.metrics.polling_metrics.batch_polls += 1

            except Exception as e:
                logger.error(f"Batch polling failed: {e}")
                # Fall back to individual polling
                await self._individual_poll_executions(batch)

    async def _individual_poll_executions(
        self, executions: List[ExecutionState]
    ) -> None:
        """Poll executions individually with concurrency control."""

        # Use semaphore to limit concurrent polls
        async def poll_single(execution: ExecutionState):
            async with self._polling_semaphore:
                await self._poll_single_execution(execution)

        # Create tasks for concurrent polling
        tasks = [poll_single(execution) for execution in executions]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _poll_single_execution(self, execution: ExecutionState) -> None:
        """Poll a single execution for status updates."""
        url = self._execution_status_url(execution.execution_id)

        start_time = time.time()
        try:
            response = await self.connection_manager.request(
                "GET", url, timeout=self.config.polling_timeout
            )
            duration = time.time() - start_time

            await self._process_poll_response(execution, response, duration)

        except Exception as e:
            duration = time.time() - start_time
            await self._process_poll_response(execution, e, duration)

    async def _process_poll_response(
        self, execution: ExecutionState, response: Any, duration: float
    ) -> None:
        """Process the response from a polling operation."""
        success = False
        timeout_occurred = False

        try:
            if isinstance(response, Exception):
                # Handle error response
                if isinstance(response, asyncio.TimeoutError):
                    timeout_occurred = True

                execution.record_poll_attempt(False, duration)

                # Update poll interval based on failure
                new_interval = min(
                    execution.current_poll_interval * 1.5, self.config.max_poll_interval
                )
                execution.update_poll_interval(new_interval)

                logger.debug(
                    f"Poll failed for execution {execution.execution_id[:8]}...: {response}"
                )

            else:
                # Handle successful response
                response.raise_for_status()
                status_data = await response.json()

                # Update execution state
                await self._update_execution_from_status(execution, status_data)

                execution.record_poll_attempt(True, duration)
                success = True

                # Update poll interval based on execution age
                new_interval = self.config.get_poll_interval_for_age(execution.age)
                execution.update_poll_interval(new_interval)

        except Exception as e:
            execution.record_poll_attempt(False, duration)
            logger.error(
                f"Error processing poll response for {execution.execution_id[:8]}...: {e}"
            )

        finally:
            # Record metrics
            self.metrics.polling_metrics.record_poll(
                success, duration, timeout_occurred
            )

    def _execution_status_url(self, execution_id: str) -> str:
        """Return the canonical status endpoint for an execution."""
        return urljoin(self.base_url, f"/api/v1/executions/{execution_id}")

    async def _update_execution_from_status(
        self, execution: ExecutionState, status_data: Dict[str, Any]
    ) -> None:
        """Update execution state from status response."""
        raw_status = status_data.get("status")
        normalized = normalize_status(raw_status)

        try:
            new_status = ExecutionStatus(normalized)
        except ValueError:
            logger.warning(
                "Unknown status '%s' for execution %s",
                normalized,
                execution.execution_id[:8],
            )
            return

        old_status = execution.status

        # Update status
        if new_status != old_status:
            if new_status == ExecutionStatus.SUCCEEDED:
                result = status_data.get("result")
                execution.set_result(result)

                async with self._execution_lock:
                    self.metrics.completed_executions += 1
                    self.metrics.active_executions -= 1
                self._release_capacity_for_execution(execution)

            elif new_status == ExecutionStatus.FAILED:
                error_msg = status_data.get("error", "Execution failed")
                error_details = status_data.get("error_details")
                execution.set_error(error_msg, error_details)

                async with self._execution_lock:
                    self.metrics.failed_executions += 1
                    self.metrics.active_executions -= 1
                self._release_capacity_for_execution(execution)
            elif new_status == ExecutionStatus.CANCELLED:
                execution.update_status(new_status)

                async with self._execution_lock:
                    self.metrics.cancelled_executions += 1
                    self.metrics.active_executions -= 1
                self._release_capacity_for_execution(execution)

            elif new_status == ExecutionStatus.TIMEOUT:
                execution.update_status(new_status)

                async with self._execution_lock:
                    self.metrics.timeout_executions += 1
                    self.metrics.active_executions -= 1
                self._release_capacity_for_execution(execution)

            else:
                execution.update_status(new_status)

            old_repr = getattr(old_status, "value", old_status)
            new_repr = getattr(new_status, "value", new_status)
            logger.debug(
                f"Execution {execution.execution_id[:8]}... status: {old_repr} -> {new_repr}"
            )

    def _release_capacity_for_execution(self, execution: ExecutionState) -> None:
        if getattr(execution, "_capacity_released", False):
            return
        execution._capacity_released = True
        try:
            self._capacity_semaphore.release()
        except ValueError:
            # Semaphore already fully released (can occur during shutdown cleanup)
            pass

    async def _cleanup_loop(self) -> None:
        """Background task for periodic cleanup of completed executions."""
        logger.debug("Starting cleanup loop")

        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(self.config.cleanup_interval)
                await self.cleanup_completed_executions()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")

        logger.debug("Cleanup loop stopped")

    async def _metrics_loop(self) -> None:
        """Background task for periodic metrics logging."""
        logger.debug("Starting metrics loop")

        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(60.0)  # Log metrics every minute

                metrics = self.get_metrics()
                logger.debug(
                    f"Execution metrics: "
                    f"active={metrics['active_executions']}, "
                    f"total={metrics['total_executions']}, "
                    f"success_rate={metrics['success_rate']:.1f}%, "
                    f"poll_success_rate={metrics['polling_metrics']['success_rate']:.1f}%"
                )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Metrics loop error: {e}")

        logger.debug("Metrics loop stopped")

    def _is_circuit_breaker_open(self) -> bool:
        """Check if circuit breaker is open."""
        if not self._circuit_breaker_open:
            return False

        # Check if recovery timeout has passed
        if (
            time.time() - self._circuit_breaker_last_failure
            > self.config.circuit_breaker_recovery_timeout
        ):
            self._circuit_breaker_open = False
            self._circuit_breaker_failures = 0
            logger.info("Circuit breaker closed - attempting recovery")
            return False

        return True

    def _record_circuit_breaker_failure(self) -> None:
        """Record a failure for circuit breaker logic."""
        self._circuit_breaker_failures += 1
        self._circuit_breaker_last_failure = time.time()

        if (
            self._circuit_breaker_failures
            >= self.config.circuit_breaker_failure_threshold
        ):
            self._circuit_breaker_open = True
            logger.warn(
                f"Circuit breaker opened after {self._circuit_breaker_failures} failures"
            )

    def __repr__(self) -> str:
        """String representation of the execution manager."""
        return (
            f"AsyncExecutionManager("
            f"base_url='{self.base_url}', "
            f"active_executions={self.metrics.active_executions}, "
            f"total_executions={self.metrics.total_executions}, "
            f"success_rate={self.metrics.success_rate:.1f}%"
            f")"
        )
