"""
Result Cache for async execution results.

This module provides in-memory caching of completed execution results with TTL
(time-to-live) support, cache size limits, LRU eviction, thread-safe operations
for concurrent access, and cache hit/miss metrics.
"""

import asyncio
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .async_config import AsyncConfig
from .execution_state import ExecutionState
from .logger import get_logger

logger = get_logger(__name__)


@dataclass
class CacheEntry:
    """Individual cache entry with metadata."""

    value: Any
    created_at: float = field(default_factory=time.time)
    accessed_at: float = field(default_factory=time.time)
    access_count: int = 0
    ttl: Optional[float] = None

    @property
    def age(self) -> float:
        """Get age of the entry in seconds."""
        return time.time() - self.created_at

    @property
    def time_since_access(self) -> float:
        """Get time since last access in seconds."""
        return time.time() - self.accessed_at

    @property
    def is_expired(self) -> bool:
        """Check if entry has expired based on TTL."""
        if self.ttl is None:
            return False
        return self.age > self.ttl

    def touch(self) -> None:
        """Update access time and increment access count."""
        self.accessed_at = time.time()
        self.access_count += 1


@dataclass
class CacheMetrics:
    """Metrics for cache performance monitoring."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    expirations: int = 0
    size: int = 0
    max_size: int = 0
    created_at: float = field(default_factory=time.time)

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate as a percentage."""
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return (self.hits / total) * 100

    @property
    def uptime(self) -> float:
        """Get cache uptime in seconds."""
        return time.time() - self.created_at

    def record_hit(self) -> None:
        """Record a cache hit."""
        self.hits += 1

    def record_miss(self) -> None:
        """Record a cache miss."""
        self.misses += 1

    def record_eviction(self) -> None:
        """Record a cache eviction."""
        self.evictions += 1

    def record_expiration(self) -> None:
        """Record a cache expiration."""
        self.expirations += 1


class ResultCache:
    """
    Thread-safe in-memory cache for execution results.

    Provides efficient caching with:
    - TTL (time-to-live) support for automatic expiration
    - LRU (Least Recently Used) eviction when size limits are reached
    - Thread-safe operations for concurrent access
    - Comprehensive metrics for cache performance monitoring
    - Configurable size limits and cleanup intervals
    """

    def __init__(self, config: Optional[AsyncConfig] = None):
        """
        Initialize the result cache.

        Args:
            config: AsyncConfig instance for configuration parameters
        """
        self.config = config or AsyncConfig()

        # Thread-safe storage using OrderedDict for LRU behavior
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()  # Reentrant lock for nested operations

        # Metrics and monitoring
        self.metrics = CacheMetrics()
        self.metrics.max_size = self.config.result_cache_max_size

        # Background cleanup (event lazily allocated to avoid loop requirements during import)
        self._cleanup_task: Optional[asyncio.Task] = None
        self._cleanup_interval = self.config.cleanup_interval
        self._shutdown_event: Optional[asyncio.Event] = None

        logger.debug(
            f"ResultCache initialized with max_size={self.config.result_cache_max_size}, ttl={self.config.result_cache_ttl}"
        )

    def __len__(self) -> int:
        """Get current cache size."""
        with self._lock:
            return len(self._cache)

    def __contains__(self, key: str) -> bool:
        """Check if key exists in cache (without affecting LRU order)."""
        with self._lock:
            return key in self._cache and not self._cache[key].is_expired

    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()

    async def start(self) -> None:
        """Start the cache and background cleanup task."""
        if self.config.enable_result_caching:
            self._shutdown_event = asyncio.Event()
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            logger.info("ResultCache started with background cleanup")
        else:
            logger.info("ResultCache started (caching disabled)")

    async def stop(self) -> None:
        """Stop the cache and cleanup background tasks."""
        if self._shutdown_event is not None:
            self._shutdown_event.set()

        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        self._shutdown_event = None

        with self._lock:
            self._cache.clear()
            self.metrics.size = 0

        logger.info("ResultCache stopped")

    def get(self, key: str) -> Optional[Any]:
        """
        Get a value from the cache.

        Args:
            key: Cache key to retrieve

        Returns:
            Cached value if found and not expired, None otherwise
        """
        if not self.config.enable_result_caching:
            self.metrics.record_miss()
            return None

        with self._lock:
            if key not in self._cache:
                self.metrics.record_miss()
                return None

            entry = self._cache[key]

            # Check if expired
            if entry.is_expired:
                self._remove_entry(key)
                self.metrics.record_miss()
                self.metrics.record_expiration()
                return None

            # Update access info and move to end (most recently used)
            entry.touch()
            self._cache.move_to_end(key)

            self.metrics.record_hit()
            logger.debug(f"Cache hit for key: {key[:20]}...")
            return entry.value

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """
        Set a value in the cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Optional TTL override (uses config default if None)
        """
        if not self.config.enable_result_caching:
            return

        # Use config TTL if not specified
        if ttl is None:
            ttl = self.config.result_cache_ttl

        with self._lock:
            # Remove existing entry if present
            if key in self._cache:
                del self._cache[key]

            # Create new entry
            entry = CacheEntry(value=value, ttl=ttl)
            self._cache[key] = entry

            # Move to end (most recently used)
            self._cache.move_to_end(key)

            # Enforce size limit with LRU eviction
            self._enforce_size_limit()

            # Update metrics
            self.metrics.size = len(self._cache)

            logger.debug(f"Cache set for key: {key[:20]}... (ttl={ttl}s)")

    def delete(self, key: str) -> bool:
        """
        Delete a key from the cache.

        Args:
            key: Cache key to delete

        Returns:
            True if key was found and deleted, False otherwise
        """
        with self._lock:
            if key in self._cache:
                self._remove_entry(key)
                return True
            return False

    def clear(self) -> None:
        """Clear all entries from the cache."""
        with self._lock:
            self._cache.clear()
            self.metrics.size = 0
            logger.debug("Cache cleared")

    def get_execution_result(self, execution_id: str) -> Optional[Any]:
        """
        Get cached result for an execution.

        Args:
            execution_id: Execution ID to retrieve result for

        Returns:
            Cached execution result if available
        """
        return self.get(f"exec:{execution_id}")

    def set_execution_result(
        self, execution_id: str, result: Any, ttl: Optional[float] = None
    ) -> None:
        """
        Cache result for an execution.

        Args:
            execution_id: Execution ID
            result: Execution result to cache
            ttl: Optional TTL override
        """
        self.set(f"exec:{execution_id}", result, ttl)

    def cache_execution_state(self, execution_state: ExecutionState) -> None:
        """
        Cache a completed execution state.

        Args:
            execution_state: ExecutionState to cache
        """
        if execution_state.is_successful and execution_state.result is not None:
            self.set_execution_result(
                execution_state.execution_id, execution_state.result
            )

    def get_keys(self, pattern: Optional[str] = None) -> List[str]:
        """
        Get all cache keys, optionally filtered by pattern.

        Args:
            pattern: Optional pattern to filter keys (simple string matching)

        Returns:
            List of cache keys
        """
        with self._lock:
            keys = list(self._cache.keys())

            if pattern:
                keys = [k for k in keys if pattern in k]

            return keys

    def get_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        with self._lock:
            # Calculate additional stats
            total_entries = len(self._cache)
            expired_count = sum(1 for entry in self._cache.values() if entry.is_expired)
            avg_age = 0.0
            avg_access_count = 0.0

            if total_entries > 0:
                avg_age = (
                    sum(entry.age for entry in self._cache.values()) / total_entries
                )
                avg_access_count = (
                    sum(entry.access_count for entry in self._cache.values())
                    / total_entries
                )

            return {
                "size": total_entries,
                "max_size": self.metrics.max_size,
                "hits": self.metrics.hits,
                "misses": self.metrics.misses,
                "hit_rate": self.metrics.hit_rate,
                "evictions": self.metrics.evictions,
                "expirations": self.metrics.expirations,
                "expired_entries": expired_count,
                "average_age": avg_age,
                "average_access_count": avg_access_count,
                "uptime": self.metrics.uptime,
                "enabled": self.config.enable_result_caching,
            }

    def _remove_entry(self, key: str) -> None:
        """Remove an entry from cache (must be called with lock held)."""
        if key in self._cache:
            del self._cache[key]
            self.metrics.size = len(self._cache)

    def _enforce_size_limit(self) -> None:
        """Enforce cache size limit using LRU eviction (must be called with lock held)."""
        while len(self._cache) > self.config.result_cache_max_size:
            # Remove least recently used (first item in OrderedDict)
            oldest_key = next(iter(self._cache))
            self._remove_entry(oldest_key)
            self.metrics.record_eviction()
            logger.debug(f"Evicted LRU entry: {oldest_key[:20]}...")

    def _cleanup_expired(self) -> int:
        """Remove expired entries from cache (must be called with lock held)."""
        expired_keys = []

        for key, entry in self._cache.items():
            if entry.is_expired:
                expired_keys.append(key)

        for key in expired_keys:
            self._remove_entry(key)
            self.metrics.record_expiration()

        return len(expired_keys)

    async def _cleanup_loop(self) -> None:
        """Background task for periodic cache cleanup."""
        shutdown_event = self._shutdown_event
        if shutdown_event is None:
            shutdown_event = asyncio.Event()
            shutdown_event.set()
        while not shutdown_event.is_set():
            try:
                await asyncio.sleep(self._cleanup_interval)

                with self._lock:
                    expired_count = self._cleanup_expired()

                    if expired_count > 0:
                        logger.debug(
                            f"Cleaned up {expired_count} expired cache entries"
                        )

                    # Log cache stats if performance logging is enabled
                    if self.config.enable_performance_logging:
                        stats = self.get_stats()
                        logger.debug(
                            f"Cache stats: {stats['size']}/{stats['max_size']} entries, "
                            f"{stats['hit_rate']:.1f}% hit rate, "
                            f"{stats['evictions']} evictions"
                        )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")

    def __repr__(self) -> str:
        """String representation of the cache."""
        with self._lock:
            return (
                f"ResultCache("
                f"size={len(self._cache)}/{self.config.result_cache_max_size}, "
                f"hit_rate={self.metrics.hit_rate:.1f}%, "
                f"enabled={self.config.enable_result_caching}"
                f")"
            )
