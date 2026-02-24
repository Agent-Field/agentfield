"""Domain-specific exceptions for the AgentField Python SDK."""

from __future__ import annotations


class AgentFieldError(Exception):
    """Base exception for all AgentField SDK errors."""

    pass


class AgentFieldClientError(AgentFieldError):
    """Error communicating with the AgentField control plane."""

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
    "ExecutionTimeoutError",
    "MemoryAccessError",
    "RegistrationError",
    "ValidationError",
]
