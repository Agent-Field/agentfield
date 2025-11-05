import asyncio
import hashlib
import os
import random
import time
from typing import Any, Optional
from agentfield.logger import log_debug


class RateLimitError(Exception):
    """Custom exception for rate limit errors"""

    def __init__(self, message: str, retry_after: Optional[float] = None):
        super().__init__(message)
        self.retry_after = retry_after


class StatelessRateLimiter:
    """
    Stateless rate limiter with adaptive exponential backoff.

    Designed to work across hundreds of containers without coordination.
    Uses container-specific jitter to naturally distribute load.
    """

    def __init__(
        self,
        max_retries: int = 20,
        base_delay: float = 1.0,
        max_delay: float = 300.0,
        jitter_factor: float = 0.25,
        circuit_breaker_threshold: int = 10,
        circuit_breaker_timeout: int = 300,
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.jitter_factor = jitter_factor
        self.circuit_breaker_threshold = circuit_breaker_threshold
        self.circuit_breaker_timeout = circuit_breaker_timeout

        # Container-specific seed for consistent but distributed jitter
        self._container_seed = self._get_container_seed()

        # Circuit breaker state (per-instance)
        self._consecutive_failures = 0
        self._circuit_open_time = None

    def _get_container_seed(self) -> int:
        """Generate a container-specific seed for consistent jitter distribution"""
        # Use hostname, process ID, and other container-specific identifiers
        identifier = f"{os.getenv('HOSTNAME', 'localhost')}-{os.getpid()}"
        return int(hashlib.md5(identifier.encode()).hexdigest()[:8], 16)

    def _is_rate_limit_error(self, error: Exception) -> bool:
        """
        Universal rate limit error detection for any LiteLLM provider.

        Args:
            error: Exception to check

        Returns:
            bool: True if this is a rate limit error
        """
        # Check for specific LiteLLM rate limit error
        if hasattr(error, "__class__") and "RateLimitError" in str(error.__class__):
            return True

        # Check HTTP status codes
        if hasattr(error, "response"):
            if hasattr(error.response, "status_code"):
                if error.response.status_code in [429, 503]:
                    return True

        # Check for HTTP status in error attributes
        if hasattr(error, "status_code"):
            if error.status_code in [429, 503]:
                return True

        # Check error message for rate limit keywords
        error_message = str(error).lower()
        rate_limit_keywords = [
            "rate limit",
            "rate-limit",
            "rate_limit",
            "too many requests",
            "quota exceeded",
            "temporarily rate-limited",
            "rate limited",
            "requests per",
            "rpm exceeded",
            "tpm exceeded",
            "usage limit",
            "throttled",
            "throttling",
        ]

        return any(keyword in error_message for keyword in rate_limit_keywords)

    def _extract_retry_after(self, error: Exception) -> Optional[float]:
        """
        Extract retry-after value from error if available.

        Args:
            error: Exception that may contain retry-after information

        Returns:
            Optional[float]: Retry-after seconds if found
        """
        # Check for Retry-After header in HTTP response
        if hasattr(error, "response") and hasattr(error.response, "headers"):
            retry_after = error.response.headers.get("Retry-After")
            if retry_after:
                try:
                    return float(retry_after)
                except ValueError:
                    pass

        # Check for retry_after in error attributes
        if hasattr(error, "retry_after"):
            try:
                return float(error.retry_after)
            except (ValueError, TypeError):
                pass

        return None

    def _calculate_backoff_delay(
        self, attempt: int, retry_after: Optional[float] = None
    ) -> float:
        """
        Calculate backoff delay with exponential backoff and jitter.

        Args:
            attempt: Current attempt number (0-based)
            retry_after: Server-suggested retry delay

        Returns:
            float: Delay in seconds
        """
        # Use server-suggested delay if available and reasonable
        if retry_after and retry_after <= self.max_delay:
            base_delay = retry_after
        else:
            # Exponential backoff: base_delay * (2 ^ attempt)
            base_delay = min(self.base_delay * (2**attempt), self.max_delay)

        # Add container-specific jitter to distribute load
        # Use container seed to ensure consistent but distributed jitter
        random.seed(self._container_seed + attempt)
        jitter_range = base_delay * self.jitter_factor
        jitter = random.uniform(-jitter_range, jitter_range)

        # Ensure minimum delay and apply jitter
        delay = max(0.1, base_delay + jitter)

        log_debug(
            f"Rate limit backoff: attempt={attempt}, base_delay={base_delay:.2f}s, jitter={jitter:.2f}s, total_delay={delay:.2f}s"
        )

        return delay

    def _check_circuit_breaker(self) -> bool:
        """
        Check if circuit breaker is open.

        Returns:
            bool: True if circuit is open (should not retry)
        """
        if self._circuit_open_time is None:
            return False

        # Check if circuit breaker timeout has passed
        if time.time() - self._circuit_open_time > self.circuit_breaker_timeout:
            log_debug("Circuit breaker timeout passed, attempting to close circuit")
            self._circuit_open_time = None
            self._consecutive_failures = 0
            return False

        return True

    def _update_circuit_breaker(self, success: bool):
        """
        Update circuit breaker state based on operation result.

        Args:
            success: Whether the operation succeeded
        """
        if success:
            # Reset on success
            self._consecutive_failures = 0
            if self._circuit_open_time:
                log_debug("Circuit breaker closed after successful request")
                self._circuit_open_time = None
        else:
            # Increment failures
            self._consecutive_failures += 1

            # Open circuit if threshold reached
            if (
                self._consecutive_failures >= self.circuit_breaker_threshold
                and self._circuit_open_time is None
            ):
                self._circuit_open_time = time.time()
                log_debug(
                    f"Circuit breaker opened after {self._consecutive_failures} consecutive failures"
                )

    async def execute_with_retry(self, func, *args, **kwargs) -> Any:
        """
        Execute a function with rate limit retry logic.

        Args:
            func: Async function to execute
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func

        Returns:
            Any: Result of successful function execution

        Raises:
            RateLimitError: If max retries exceeded or circuit breaker is open
            Exception: Original exception if not rate limit related
        """
        # Check circuit breaker
        if self._check_circuit_breaker():
            raise RateLimitError(
                f"Circuit breaker is open. Too many consecutive rate limit failures. "
                f"Will retry after {self.circuit_breaker_timeout} seconds."
            )

        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                # Execute the function
                result = await func(*args, **kwargs)

                # Success - update circuit breaker and return
                self._update_circuit_breaker(success=True)

                if attempt > 0:
                    log_debug(f"Rate limit retry succeeded on attempt {attempt + 1}")

                return result

            except Exception as error:
                last_error = error

                # Check if this is a rate limit error
                if not self._is_rate_limit_error(error):
                    # Not a rate limit error - re-raise immediately
                    raise error

                # Update circuit breaker for rate limit failure
                self._update_circuit_breaker(success=False)

                # Check if we've exceeded max retries
                if attempt >= self.max_retries:
                    log_debug(f"Rate limit max retries ({self.max_retries}) exceeded")
                    break

                # Extract retry-after if available
                retry_after = self._extract_retry_after(error)

                # Calculate backoff delay
                delay = self._calculate_backoff_delay(attempt, retry_after)

                log_debug(
                    f"Rate limit detected on attempt {attempt + 1}, retrying in {delay:.2f}s. Error: {str(error)[:100]}"
                )

                # Wait before retry
                await asyncio.sleep(delay)

        # All retries exhausted
        raise RateLimitError(
            f"Rate limit retries exhausted after {self.max_retries} attempts. "
            f"Last error: {str(last_error)}"
        )
