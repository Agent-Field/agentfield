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


class ExecutionTimeoutError(AgentFieldError):
    """Execution timed out waiting for completion."""

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
    "ExecutionTimeoutError",
    "MemoryAccessError",
    "RegistrationError",
    "ValidationError",
]
