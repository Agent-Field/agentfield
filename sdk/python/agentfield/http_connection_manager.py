"""
HTTP Connection Manager for async execution.

This module provides aiohttp session pooling with configurable connection limits,
connection reuse, proper cleanup, timeout handling, and connection health monitoring.
Supports both single requests and batch operations for the AgentField SDK async execution.
"""

import asyncio
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import aiohttp

from .async_config import AsyncConfig
from .logger import get_logger

logger = get_logger(__name__)


@dataclass
class ConnectionMetrics:
    """Metrics for connection pool monitoring."""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    timeout_requests: int = 0
    active_connections: int = 0
    pool_size: int = 0
    created_at: float = field(default_factory=time.time)

    @property
    def success_rate(self) -> float:
        """Calculate success rate as a percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100

    @property
    def uptime(self) -> float:
        """Get uptime in seconds."""
        return time.time() - self.created_at

    def record_request(self, success: bool, timeout: bool = False) -> None:
        """Record a request attempt."""
        self.total_requests += 1
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
            if timeout:
                self.timeout_requests += 1


@dataclass
class ConnectionHealth:
    """Health status of connection pool."""

    is_healthy: bool = True
    last_check: float = field(default_factory=time.time)
    consecutive_failures: int = 0
    last_error: Optional[str] = None

    def mark_healthy(self) -> None:
        """Mark connection as healthy."""
        self.is_healthy = True
        self.consecutive_failures = 0
        self.last_error = None
        self.last_check = time.time()

    def mark_unhealthy(self, error: str) -> None:
        """Mark connection as unhealthy."""
        self.is_healthy = False
        self.consecutive_failures += 1
        self.last_error = error
        self.last_check = time.time()


class ConnectionManager:
    """
    HTTP Connection Manager with aiohttp session pooling.

    Provides efficient HTTP connection management for async execution with:
    - Configurable connection limits and timeouts
    - Connection reuse and proper cleanup
    - Health monitoring and metrics
    - Support for single requests and batch operations
    - Thread-safe operations for concurrent access
    """

    def __init__(self, config: Optional[AsyncConfig] = None):
        """
        Initialize the connection manager.

        Args:
            config: AsyncConfig instance for configuration parameters
        """
        self.config = config or AsyncConfig()
        self._session: Optional[aiohttp.ClientSession] = None
        self._connector: Optional[aiohttp.TCPConnector] = None
        self._lock = asyncio.Lock()
        self._closed = False

        # Metrics and health monitoring
        self.metrics = ConnectionMetrics()
        self.health = ConnectionHealth()

        # Background tasks
        self._health_check_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None

        logger.debug(f"ConnectionManager initialized with config: {self.config}")

    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    async def start(self) -> None:
        """
        Start the connection manager and initialize session.

        Raises:
            RuntimeError: If manager is already started or closed
        """
        async with self._lock:
            if self._session is not None:
                raise RuntimeError("ConnectionManager is already started")

            if self._closed:
                raise RuntimeError(
                    "ConnectionManager is closed and cannot be restarted"
                )

            # Create TCP connector with configuration
            self._connector = aiohttp.TCPConnector(
                limit=self.config.connection_pool_size,
                limit_per_host=self.config.connection_pool_per_host,
                ttl_dns_cache=300,  # 5 minutes DNS cache
                use_dns_cache=True,
                keepalive_timeout=30,
                enable_cleanup_closed=True,
                force_close=False,
            )

            # Create session with timeout configuration
            timeout = aiohttp.ClientTimeout(
                total=self.config.max_execution_timeout,
                connect=self.config.polling_timeout,
                sock_read=self.config.polling_timeout,
            )

            self._session = aiohttp.ClientSession(
                connector=self._connector,
                timeout=timeout,
                headers={
                    "User-Agent": "AgentField-SDK-AsyncClient/1.0",
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                },
            )

            # Update metrics
            self.metrics.pool_size = self.config.connection_pool_size

            # Start background tasks if enabled
            if self.config.enable_performance_logging:
                self._health_check_task = asyncio.create_task(self._health_check_loop())
                self._cleanup_task = asyncio.create_task(self._cleanup_loop())

            logger.info(
                f"ConnectionManager started with pool size {self.config.connection_pool_size}"
            )

    async def close(self) -> None:
        """
        Close the connection manager and cleanup resources.
        """
        async with self._lock:
            if self._closed:
                return

            self._closed = True

            # Cancel background tasks
            if self._health_check_task:
                self._health_check_task.cancel()
                try:
                    await self._health_check_task
                except asyncio.CancelledError:
                    pass

            if self._cleanup_task:
                self._cleanup_task.cancel()
                try:
                    await self._cleanup_task
                except asyncio.CancelledError:
                    pass

            # Close session and connector
            if self._session:
                await self._session.close()
                self._session = None

            if self._connector:
                await self._connector.close()
                self._connector = None

            logger.info("ConnectionManager closed")

    @asynccontextmanager
    async def get_session(self):
        """
        Get an aiohttp session for making requests.

        Yields:
            aiohttp.ClientSession: Active session for making requests

        Raises:
            RuntimeError: If manager is not started or is closed
        """
        if self._session is None:
            raise RuntimeError("ConnectionManager is not started. Call start() first.")

        if self._closed:
            raise RuntimeError("ConnectionManager is closed")

        try:
            yield self._session
        except Exception as e:
            self.health.mark_unhealthy(str(e))
            raise

    async def request(self, method: str, url: str, **kwargs) -> aiohttp.ClientResponse:
        """
        Make a single HTTP request.

        Args:
            method: HTTP method (GET, POST, etc.)
            url: Request URL
            **kwargs: Additional arguments for aiohttp request

        Returns:
            aiohttp.ClientResponse: Response object

        Raises:
            aiohttp.ClientError: For HTTP-related errors
            asyncio.TimeoutError: For timeout errors
        """
        start_time = time.time()
        success = False
        timeout_occurred = False

        try:
            async with self.get_session() as session:
                response = await session.request(method, url, **kwargs)
                success = True
                self.health.mark_healthy()
                return response

        except asyncio.TimeoutError:
            timeout_occurred = True
            logger.warn(f"Request timeout for {method} {url}")
            raise
        except Exception as e:
            self.health.mark_unhealthy(str(e))
            logger.error(f"Request failed for {method} {url}: {e}")
            raise
        finally:
            # Record metrics
            self.metrics.record_request(success, timeout_occurred)

            # Log slow requests
            duration = time.time() - start_time
            if (
                self.config.log_slow_executions
                and duration > self.config.slow_execution_threshold
            ):
                logger.warn(f"Slow request: {method} {url} took {duration:.2f}s")

    async def batch_request(
        self, requests: List[Dict[str, Any]]
    ) -> List[Union[aiohttp.ClientResponse, Exception]]:
        """
        Make multiple HTTP requests concurrently.

        Args:
            requests: List of request dictionaries with 'method', 'url', and optional kwargs

        Returns:
            List of responses or exceptions for each request
        """
        if not requests:
            return []

        # Limit concurrent requests
        semaphore = asyncio.Semaphore(self.config.max_active_polls)

        async def make_request(
            req_data: Dict[str, Any],
        ) -> Union[aiohttp.ClientResponse, Exception]:
            async with semaphore:
                try:
                    method = req_data.pop("method")
                    url = req_data.pop("url")
                    return await self.request(method, url, **req_data)
                except Exception as e:
                    return e

        # Execute all requests concurrently
        tasks = [make_request(req.copy()) for req in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        logger.debug(f"Batch request completed: {len(requests)} requests")
        return results

    async def health_check(self) -> bool:
        """
        Perform a health check on the connection pool.

        Returns:
            bool: True if healthy, False otherwise
        """
        try:
            # Simple health check - try to create a request
            if self._session is None or self._session.closed:
                self.health.mark_unhealthy("Session is closed")
                return False

            # Check connector health
            if self._connector is None or self._connector.closed:
                self.health.mark_unhealthy("Connector is closed")
                return False

            self.health.mark_healthy()
            return True

        except Exception as e:
            self.health.mark_unhealthy(str(e))
            return False

    async def _health_check_loop(self) -> None:
        """Background task for periodic health checks."""
        while not self._closed:
            try:
                await asyncio.sleep(60)  # Check every minute
                await self.health_check()

                # Log health status if unhealthy
                if not self.health.is_healthy:
                    logger.warn(f"Connection pool unhealthy: {self.health.last_error}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check loop error: {e}")

    async def _cleanup_loop(self) -> None:
        """Background task for periodic cleanup."""
        while not self._closed:
            try:
                await asyncio.sleep(self.config.cleanup_interval)

                # Update active connections metric
                if self._connector:
                    self.metrics.active_connections = len(self._connector._conns)

                # Log metrics if performance logging is enabled
                if self.config.enable_performance_logging:
                    logger.debug(
                        f"Connection metrics: {self.metrics.total_requests} total, "
                        f"{self.metrics.success_rate:.1f}% success, "
                        f"{self.metrics.active_connections} active"
                    )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")

    def get_metrics(self) -> ConnectionMetrics:
        """
        Get current connection metrics.

        Returns:
            ConnectionMetrics: Current metrics snapshot
        """
        # Update active connections if connector is available
        if self._connector:
            self.metrics.active_connections = len(self._connector._conns)

        return self.metrics

    def get_health(self) -> ConnectionHealth:
        """
        Get current health status.

        Returns:
            ConnectionHealth: Current health status
        """
        return self.health

    @property
    def is_healthy(self) -> bool:
        """Check if connection manager is healthy."""
        return self.health.is_healthy and not self._closed

    @property
    def is_closed(self) -> bool:
        """Check if connection manager is closed."""
        return self._closed

    def __repr__(self) -> str:
        """String representation of the connection manager."""
        return (
            f"ConnectionManager("
            f"pool_size={self.config.connection_pool_size}, "
            f"healthy={self.is_healthy}, "
            f"closed={self.is_closed}, "
            f"requests={self.metrics.total_requests}"
            f")"
        )
