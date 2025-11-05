"""
Async execution configuration for the AgentField SDK.

This module provides configuration classes for managing async execution behavior,
polling strategies, resource limits, and performance tuning parameters.
"""

from dataclasses import dataclass
import os


@dataclass
class AsyncConfig:
    """
    Configuration class for async execution behavior.

    This class defines all the parameters needed for efficient async execution
    including polling intervals, resource limits, timeouts, and performance tuning.
    """

    # Polling Strategy Configuration
    initial_poll_interval: float = 0.03  # 30ms - aggressive initial polling
    fast_poll_interval: float = 0.08  # 80ms - for short executions (0-10s)
    medium_poll_interval: float = 0.4  # 400ms - for medium executions (10s-60s)
    slow_poll_interval: float = 1.5  # 1.5s - for long executions (60s+)
    max_poll_interval: float = 4.0  # 4s - maximum polling interval

    # Execution Duration Thresholds (in seconds)
    fast_execution_threshold: float = 10.0  # Switch to medium polling after 10s
    medium_execution_threshold: float = 60.0  # Switch to slow polling after 60s

    # Timeout Configuration
    max_execution_timeout: float = 21600.0  # 6 hours maximum execution time
    default_execution_timeout: float = 7200.0  # 2 hours default timeout
    polling_timeout: float = 20.0  # 20s timeout for individual poll requests

    # Resource Limits
    max_concurrent_executions: int = 4096  # Maximum concurrent executions to track
    max_active_polls: int = 512  # Maximum concurrent polling operations
    connection_pool_size: int = 64  # HTTP connection pool size
    connection_pool_per_host: int = 32  # Connections per host

    # Batch Processing
    batch_size: int = 100  # Maximum executions to check in single batch
    batch_poll_interval: float = 0.1  # 100ms - interval for batch polling

    # Caching Configuration
    result_cache_ttl: float = 600.0  # 10 minutes - cache completed results
    result_cache_max_size: int = 20000  # Maximum cached results

    # Memory Management
    cleanup_interval: float = 30.0  # 30 seconds - cleanup completed executions
    max_completed_executions: int = 4000  # Keep max 4000 completed executions
    completed_execution_retention_seconds: float = (
        600.0  # Retain completed executions for 10 minutes
    )

    # Retry and Backoff Configuration
    max_retry_attempts: int = 3  # Maximum retry attempts for failed polls
    retry_backoff_base: float = 1.0  # Base backoff time (seconds)
    retry_backoff_multiplier: float = 2.0  # Exponential backoff multiplier
    retry_backoff_max: float = 30.0  # Maximum backoff time

    # Circuit Breaker Configuration
    circuit_breaker_failure_threshold: int = 5  # Failures before opening circuit
    circuit_breaker_recovery_timeout: float = 60.0  # Time before attempting recovery
    circuit_breaker_success_threshold: int = 3  # Successes needed to close circuit

    # Logging and Monitoring
    enable_performance_logging: bool = False  # Enable detailed performance logs
    enable_polling_metrics: bool = False  # Enable polling metrics collection
    log_slow_executions: bool = True  # Log executions exceeding threshold
    slow_execution_threshold: float = 30.0  # Threshold for slow execution logging

    # Feature Flags
    enable_async_execution: bool = True  # Master switch for async execution
    enable_batch_polling: bool = True  # Enable batch status checking
    enable_result_caching: bool = True  # Enable result caching
    enable_connection_pooling: bool = True  # Enable HTTP connection pooling
    fallback_to_sync: bool = True  # Fallback to sync if async fails

    # Event streaming (SSE) configuration
    enable_event_stream: bool = False  # Subscribe to SSE updates when available
    event_stream_path: str = "/api/ui/v1/executions/events"
    event_stream_retry_backoff: float = (
        3.0  # Seconds before reconnect after stream errors
    )

    @classmethod
    def from_environment(cls) -> "AsyncConfig":
        """
        Create AsyncConfig from environment variables.

        Environment variables use the prefix AGENTFIELD_ASYNC_ followed by the
        uppercase parameter name. For example:
        - AGENTFIELD_ASYNC_MAX_EXECUTION_TIMEOUT=1800
        - AGENTFIELD_ASYNC_BATCH_SIZE=50

        Returns:
            AsyncConfig instance with values from environment variables
        """
        config = cls()

        # Helper function to get env var with type conversion
        def get_env_var(name: str, default_value, converter=None):
            env_name = f"AGENTFIELD_ASYNC_{name.upper()}"
            value = os.getenv(env_name)
            if value is None:
                return default_value

            if converter:
                try:
                    return converter(value)
                except (ValueError, TypeError):
                    return default_value
            return value

        # Polling Configuration
        config.initial_poll_interval = get_env_var(
            "initial_poll_interval", config.initial_poll_interval, float
        )
        config.fast_poll_interval = get_env_var(
            "fast_poll_interval", config.fast_poll_interval, float
        )
        config.medium_poll_interval = get_env_var(
            "medium_poll_interval", config.medium_poll_interval, float
        )
        config.slow_poll_interval = get_env_var(
            "slow_poll_interval", config.slow_poll_interval, float
        )
        config.max_poll_interval = get_env_var(
            "max_poll_interval", config.max_poll_interval, float
        )

        # Timeout Configuration
        config.max_execution_timeout = get_env_var(
            "max_execution_timeout", config.max_execution_timeout, float
        )
        config.default_execution_timeout = get_env_var(
            "default_execution_timeout", config.default_execution_timeout, float
        )
        config.polling_timeout = get_env_var(
            "polling_timeout", config.polling_timeout, float
        )

        # Resource Limits
        config.max_concurrent_executions = get_env_var(
            "max_concurrent_executions", config.max_concurrent_executions, int
        )
        config.max_active_polls = get_env_var(
            "max_active_polls", config.max_active_polls, int
        )
        config.connection_pool_size = get_env_var(
            "connection_pool_size", config.connection_pool_size, int
        )
        config.batch_size = get_env_var("batch_size", config.batch_size, int)

        # Feature Flags
        config.enable_async_execution = get_env_var(
            "enable_async_execution",
            config.enable_async_execution,
            lambda x: x.lower() == "true",
        )
        config.enable_batch_polling = get_env_var(
            "enable_batch_polling",
            config.enable_batch_polling,
            lambda x: x.lower() == "true",
        )
        config.enable_result_caching = get_env_var(
            "enable_result_caching",
            config.enable_result_caching,
            lambda x: x.lower() == "true",
        )
        config.fallback_to_sync = get_env_var(
            "fallback_to_sync", config.fallback_to_sync, lambda x: x.lower() == "true"
        )
        config.enable_event_stream = get_env_var(
            "enable_event_stream",
            config.enable_event_stream,
            lambda x: x.lower() == "true",
        )
        config.event_stream_path = get_env_var(
            "event_stream_path", config.event_stream_path
        )
        config.event_stream_retry_backoff = get_env_var(
            "event_stream_retry_backoff", config.event_stream_retry_backoff, float
        )

        config.completed_execution_retention_seconds = get_env_var(
            "completed_execution_retention_seconds",
            config.completed_execution_retention_seconds,
            float,
        )

        return config

    def validate(self) -> None:
        """
        Validate configuration parameters.

        Raises:
            ValueError: If any configuration parameter is invalid
        """
        if self.initial_poll_interval <= 0:
            raise ValueError("initial_poll_interval must be positive")

        if self.max_execution_timeout <= 0:
            raise ValueError("max_execution_timeout must be positive")

        if self.default_execution_timeout <= 0:
            raise ValueError("default_execution_timeout must be positive")

        if self.default_execution_timeout > self.max_execution_timeout:
            raise ValueError(
                "default_execution_timeout cannot exceed max_execution_timeout"
            )

        if self.max_concurrent_executions <= 0:
            raise ValueError("max_concurrent_executions must be positive")

        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

        if self.connection_pool_size <= 0:
            raise ValueError("connection_pool_size must be positive")

        # Ensure polling intervals are in logical order
        if not (
            self.initial_poll_interval
            <= self.fast_poll_interval
            <= self.medium_poll_interval
            <= self.slow_poll_interval
            <= self.max_poll_interval
        ):
            raise ValueError("Polling intervals must be in ascending order")

        # Ensure thresholds are logical
        if self.fast_execution_threshold >= self.medium_execution_threshold:
            raise ValueError(
                "fast_execution_threshold must be less than medium_execution_threshold"
            )

        if self.completed_execution_retention_seconds < 0:
            raise ValueError("completed_execution_retention_seconds cannot be negative")

    def get_poll_interval_for_age(self, execution_age: float) -> float:
        """
        Get the appropriate polling interval based on execution age.

        Args:
            execution_age: Age of the execution in seconds

        Returns:
            Appropriate polling interval in seconds
        """
        if execution_age < self.fast_execution_threshold:
            return self.fast_poll_interval
        elif execution_age < self.medium_execution_threshold:
            return self.medium_poll_interval
        else:
            return self.slow_poll_interval

    def __str__(self) -> str:
        """String representation of the configuration."""
        return (
            f"AsyncConfig("
            f"polling={self.initial_poll_interval}->{self.max_poll_interval}s, "
            f"timeout={self.max_execution_timeout}s, "
            f"max_concurrent={self.max_concurrent_executions}, "
            f"batch_size={self.batch_size}, "
            f"async_enabled={self.enable_async_execution}, "
            f"event_stream={self.enable_event_stream}"
            f")"
        )


# Global default configuration instance
DEFAULT_ASYNC_CONFIG = AsyncConfig()
