"""Domain-specific exceptions for the AgentField Python SDK."""

from __future__ import annotations


class AgentFieldError(Exception):
    """Base exception for all AgentField SDK errors."""
    def __init__(self, message:str):
        super().__init__(message)

class AgentFieldClientError(AgentFieldError):
    """Error communicating with the AgentField control plane."""

    pass


class ExecutionFailedError(AgentFieldClientError):
    """The remote reasoner ran and explicitly returned a failed status.

    Distinct from a transport / submission / network failure (plain
    ``AgentFieldClientError``): the call reached the reasoner, the work
    ran, and the reasoner returned an error. Retrying via the sync
    fallback path would re-run the same reasoner with the same input,
    burn the same budget, and produce the same failure — so the SDK's
    ``Agent.call`` skips the sync fallback when this is raised.

    Inherits from ``AgentFieldClientError`` for backward compatibility:
    callers that catch ``AgentFieldClientError`` still see this. New
    callers that want to distinguish "the work ran and failed" from
    "the call never reached the reasoner" should catch this directly.
    """

    pass


class ReasonerFailed(AgentFieldError):
    """Raised *inside* a reasoner to report that the work ran but failed.

    Returning a value from a reasoner — even one whose payload says
    ``success: False`` — makes the async execution handler record the
    execution as ``succeeded`` (it only distinguishes "returned" from
    "raised", never inspecting the result). A build that completes zero
    issues and merges nothing therefore surfaces as green, which is easy to
    act on incorrectly.

    Raise this when the reasoner has determined its own work failed but you
    still want the structured ``result`` preserved on the execution record.
    The handler posts ``status="failed"`` to the control plane while also
    sending ``result`` (the control plane stores the result payload
    regardless of status), so the rich outcome — debt, DAG state, any PR
    that was opened — is not lost behind a bare error string.

    Args:
        message: Human-readable failure summary (becomes the execution error).
        result: Optional structured result to preserve on the execution
            record (e.g. ``BuildResult.model_dump()``). JSON-encoded by the
            handler before it is posted.
        error_details: Optional structured error metadata, mirrored onto the
            status payload's ``error_details`` field.
    """

    def __init__(
        self,
        message: str,
        *,
        result: object | None = None,
        error_details: object | None = None,
    ) -> None:
        super().__init__(message)
        self.result = result
        self.error_details = error_details


class ExecutionTimeoutError(AgentFieldError):
    """Execution timed out waiting for completion."""

    pass


class ExecutionCancelledError(AgentFieldError):
    """The awaited execution was cancelled (typically by user action via the
    control plane's cancel-tree endpoint).

    Distinct from ``ExecutionFailedError`` (the reasoner ran and failed) and
    from a transport / submission failure (plain ``AgentFieldClientError``):
    cancellation expresses *explicit user intent* to stop the work. The SDK's
    ``Agent.call`` must not silently re-issue a cancelled call via the sync
    fallback path — that would re-run work the user explicitly told the
    system to abandon.

    Intentionally NOT a subclass of ``AgentFieldClientError`` (the
    retry-eligible bucket): cancellation is never retry-eligible, regardless
    of ``async_config.fallback_to_sync``.
    """

    pass


class MemoryAccessError(AgentFieldError):
    """Error accessing agent memory storage."""

    pass


class RegistrationError(AgentFieldError):
    """Error registering agent with control plane."""

    pass


class ValidationError(AgentFieldError):
    """Input validation error."""

    pass


__all__ = [
    "AgentFieldError",
    "AgentFieldClientError",
    "ExecutionFailedError",
    "ReasonerFailed",
    "ExecutionTimeoutError",
    "ExecutionCancelledError",
    "MemoryAccessError",
    "RegistrationError",
    "ValidationError",
]
