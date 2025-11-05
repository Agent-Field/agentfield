"""
Execution state management for async executions.

This module provides dataclasses and enums for tracking the state of individual
async executions throughout their lifecycle.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional, List
import time


class ExecutionStatus(Enum):
    """Enumeration of possible execution statuses."""

    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


class ExecutionPriority(Enum):
    """Enumeration of execution priorities for queue management."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


@dataclass
class ExecutionMetrics:
    """Metrics and performance data for an execution."""

    # Timing metrics
    submit_time: float = field(default_factory=time.time)
    start_time: Optional[float] = None
    end_time: Optional[float] = None

    # Polling metrics
    poll_count: int = 0
    total_poll_time: float = 0.0
    last_poll_time: Optional[float] = None

    # Network metrics
    network_requests: int = 0
    network_errors: int = 0
    retry_count: int = 0

    # Resource metrics
    result_size_bytes: Optional[int] = None
    memory_usage_mb: Optional[float] = None

    @property
    def total_duration(self) -> Optional[float]:
        """Total execution duration in seconds."""
        if self.submit_time and self.end_time:
            return self.end_time - self.submit_time
        return None

    @property
    def execution_duration(self) -> Optional[float]:
        """Actual execution duration (excluding queue time)."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None

    @property
    def queue_duration(self) -> Optional[float]:
        """Time spent in queue before execution started."""
        if self.submit_time and self.start_time:
            return self.start_time - self.submit_time
        return None

    @property
    def average_poll_interval(self) -> Optional[float]:
        """Average time between polls."""
        if self.poll_count > 1 and self.total_poll_time > 0:
            return self.total_poll_time / (self.poll_count - 1)
        return None

    def add_poll(self, poll_duration: float) -> None:
        """Record a polling operation."""
        self.poll_count += 1
        self.total_poll_time += poll_duration
        self.last_poll_time = time.time()
        self.network_requests += 1

    def add_network_error(self) -> None:
        """Record a network error."""
        self.network_errors += 1

    def add_retry(self) -> None:
        """Record a retry attempt."""
        self.retry_count += 1


@dataclass
class ExecutionState:
    """
    Complete state information for an async execution.

    This class tracks all aspects of an execution from submission to completion,
    including status, results, errors, metrics, and polling information.
    """

    # Core identification
    execution_id: str
    target: str
    input_data: Dict[str, Any]

    # Status and lifecycle
    status: ExecutionStatus = ExecutionStatus.QUEUED
    priority: ExecutionPriority = ExecutionPriority.NORMAL
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Results and errors
    result: Optional[Any] = None
    error_message: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None

    # Execution context
    workflow_id: Optional[str] = None
    parent_execution_id: Optional[str] = None
    session_id: Optional[str] = None
    actor_id: Optional[str] = None

    # Webhook metadata
    webhook_registered: bool = False
    webhook_error: Optional[str] = None

    # Configuration
    timeout: Optional[float] = None
    max_retries: int = 3

    # Polling state
    next_poll_time: float = field(default_factory=time.time)
    current_poll_interval: float = 0.05  # Start with 50ms
    consecutive_failures: int = 0

    # Metrics and monitoring
    metrics: ExecutionMetrics = field(default_factory=ExecutionMetrics)

    # Internal state
    _is_cancelled: bool = field(default=False, init=False)
    _cancellation_reason: Optional[str] = field(default=None, init=False)
    _capacity_released: bool = field(default=False, init=False, repr=False)

    def __post_init__(self):
        """Post-initialization setup."""
        # Ensure metrics are initialized
        if not hasattr(self, "metrics") or self.metrics is None:
            self.metrics = ExecutionMetrics()

        # Set initial poll time
        if self.next_poll_time == 0:
            self.next_poll_time = time.time() + self.current_poll_interval

    @property
    def age(self) -> float:
        """Age of the execution in seconds since creation."""
        return time.time() - self.metrics.submit_time

    @property
    def is_terminal(self) -> bool:
        """Whether the execution is in a terminal state."""
        return self.status in {
            ExecutionStatus.SUCCEEDED,
            ExecutionStatus.FAILED,
            ExecutionStatus.CANCELLED,
            ExecutionStatus.TIMEOUT,
        }

    @property
    def is_active(self) -> bool:
        """Whether the execution is actively running or queued."""
        return self.status in {
            ExecutionStatus.PENDING,
            ExecutionStatus.QUEUED,
            ExecutionStatus.RUNNING,
        }

    @property
    def is_successful(self) -> bool:
        """Whether the execution completed successfully."""
        return self.status == ExecutionStatus.SUCCEEDED and self.result is not None

    @property
    def is_cancelled(self) -> bool:
        """Whether the execution has been cancelled."""
        return self._is_cancelled or self.status == ExecutionStatus.CANCELLED

    @property
    def should_poll(self) -> bool:
        """Whether this execution should be polled now."""
        return (
            self.is_active
            and not self.is_cancelled
            and time.time() >= self.next_poll_time
        )

    @property
    def is_overdue(self) -> bool:
        """Whether this execution has exceeded its timeout."""
        if self.timeout is None:
            return False
        return self.age > self.timeout

    def update_status(
        self, status: ExecutionStatus, error_message: Optional[str] = None
    ) -> None:
        """
        Update the execution status and timestamp.

        Args:
            status: New execution status
            error_message: Optional error message for failed executions
        """
        old_status = self.status
        self.status = status
        self.updated_at = datetime.now(timezone.utc)

        # Update metrics based on status change
        current_time = time.time()

        if old_status == ExecutionStatus.QUEUED and status == ExecutionStatus.RUNNING:
            self.metrics.start_time = current_time
        elif status in {
            ExecutionStatus.SUCCEEDED,
            ExecutionStatus.FAILED,
            ExecutionStatus.CANCELLED,
            ExecutionStatus.TIMEOUT,
        }:
            self.metrics.end_time = current_time

        # Handle error cases
        if status == ExecutionStatus.FAILED and error_message:
            self.error_message = error_message

    def set_result(self, result: Any) -> None:
        """
        Set the execution result and mark as completed.

        Args:
            result: The execution result
        """
        self.result = result
        self.update_status(ExecutionStatus.SUCCEEDED)

        # Calculate result size if possible
        try:
            import sys

            self.metrics.result_size_bytes = sys.getsizeof(result)
        except Exception:
            pass  # Size calculation is optional

    def set_error(
        self, error_message: str, error_details: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Set execution error and mark as failed.

        Args:
            error_message: Human-readable error message
            error_details: Optional detailed error information
        """
        self.error_message = error_message
        self.error_details = error_details
        self.update_status(ExecutionStatus.FAILED)

    def cancel(self, reason: Optional[str] = None) -> None:
        """
        Cancel the execution.

        Args:
            reason: Optional cancellation reason
        """
        self._is_cancelled = True
        self._cancellation_reason = reason
        self.update_status(ExecutionStatus.CANCELLED)

    def timeout_execution(self) -> None:
        """Mark the execution as timed out."""
        self.update_status(
            ExecutionStatus.TIMEOUT, f"Execution timed out after {self.timeout} seconds"
        )

    def update_poll_interval(self, new_interval: float) -> None:
        """
        Update the polling interval and next poll time.

        Args:
            new_interval: New polling interval in seconds
        """
        self.current_poll_interval = new_interval
        self.next_poll_time = time.time() + new_interval

    def record_poll_attempt(self, success: bool, duration: float = 0.0) -> None:
        """
        Record a polling attempt.

        Args:
            success: Whether the poll was successful
            duration: Duration of the poll request
        """
        self.metrics.add_poll(duration)

        if success:
            self.consecutive_failures = 0
        else:
            self.consecutive_failures += 1
            self.metrics.add_network_error()

    def record_retry(self) -> None:
        """Record a retry attempt."""
        self.metrics.add_retry()

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert execution state to dictionary representation.

        Returns:
            Dictionary representation of the execution state
        """
        return {
            "execution_id": self.execution_id,
            "target": self.target,
            "status": self.status.value,
            "priority": self.priority.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "age": self.age,
            "result": self.result,
            "error_message": self.error_message,
            "error_details": self.error_details,
            "workflow_id": self.workflow_id,
            "parent_execution_id": self.parent_execution_id,
            "session_id": self.session_id,
            "actor_id": self.actor_id,
            "timeout": self.timeout,
            "is_terminal": self.is_terminal,
            "is_active": self.is_active,
            "is_successful": self.is_successful,
            "is_cancelled": self.is_cancelled,
            "metrics": {
                "total_duration": self.metrics.total_duration,
                "execution_duration": self.metrics.execution_duration,
                "queue_duration": self.metrics.queue_duration,
                "poll_count": self.metrics.poll_count,
                "network_requests": self.metrics.network_requests,
                "network_errors": self.metrics.network_errors,
                "retry_count": self.metrics.retry_count,
                "result_size_bytes": self.metrics.result_size_bytes,
                "average_poll_interval": self.metrics.average_poll_interval,
            },
        }

    def __str__(self) -> str:
        """String representation of the execution state."""
        return (
            f"ExecutionState(id={self.execution_id[:8]}..., "
            f"target={self.target}, status={self.status.value}, "
            f"age={self.age:.1f}s, polls={self.metrics.poll_count})"
        )

    def __repr__(self) -> str:
        """Detailed string representation."""
        return (
            f"ExecutionState("
            f"execution_id='{self.execution_id}', "
            f"target='{self.target}', "
            f"status={self.status}, "
            f"age={self.age:.2f}, "
            f"polls={self.metrics.poll_count}, "
            f"interval={self.current_poll_interval}"
            f")"
        )


@dataclass
class ExecutionBatch:
    """
    Represents a batch of executions for efficient batch processing.
    """

    executions: List[ExecutionState] = field(default_factory=list)
    batch_id: str = field(default_factory=lambda: f"batch_{int(time.time() * 1000)}")
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def size(self) -> int:
        """Number of executions in the batch."""
        return len(self.executions)

    @property
    def execution_ids(self) -> List[str]:
        """List of execution IDs in the batch."""
        return [exec_state.execution_id for exec_state in self.executions]

    @property
    def active_executions(self) -> List[ExecutionState]:
        """List of active (non-terminal) executions in the batch."""
        return [exec_state for exec_state in self.executions if exec_state.is_active]

    @property
    def completed_executions(self) -> List[ExecutionState]:
        """List of completed executions in the batch."""
        return [exec_state for exec_state in self.executions if exec_state.is_terminal]

    def add_execution(self, execution: ExecutionState) -> None:
        """Add an execution to the batch."""
        if execution not in self.executions:
            self.executions.append(execution)

    def remove_execution(self, execution_id: str) -> Optional[ExecutionState]:
        """Remove and return an execution from the batch."""
        for i, execution in enumerate(self.executions):
            if execution.execution_id == execution_id:
                return self.executions.pop(i)
        return None

    def get_execution(self, execution_id: str) -> Optional[ExecutionState]:
        """Get an execution by ID."""
        for execution in self.executions:
            if execution.execution_id == execution_id:
                return execution
        return None

    def clear_completed(self) -> List[ExecutionState]:
        """Remove and return all completed executions."""
        completed = self.completed_executions
        self.executions = self.active_executions
        return completed

    def __len__(self) -> int:
        """Number of executions in the batch."""
        return len(self.executions)

    def __iter__(self):
        """Iterate over executions in the batch."""
        return iter(self.executions)

    def __str__(self) -> str:
        """String representation of the batch."""
        return f"ExecutionBatch(id={self.batch_id}, size={self.size}, active={len(self.active_executions)})"
