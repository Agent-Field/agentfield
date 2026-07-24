"""Session provider/transport capability validation.

The SDK deliberately does not infer a transport from provider, caller type, or
runtime environment. Session authors choose both knobs explicitly; this module
only validates that the combination is supported.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet


SUPPORTED_SESSION_TRANSPORTS: Dict[str, FrozenSet[str]] = {
    "openai": frozenset({"webrtc", "websocket"}),
    "openrouter": frozenset({"audio_turns"}),
}


class SessionTransportError(ValueError):
    """Raised when a session provider/transport pair is unsupported."""

    def __init__(self, provider: str, transport: str, supported: FrozenSet[str]) -> None:
        self.provider = provider
        self.transport = transport
        self.supported = supported
        supported_display = ", ".join(sorted(supported)) or "none"
        super().__init__(
            f"Unsupported session transport '{transport}' for provider '{provider}'. "
            f"Supported transports: {supported_display}. AgentField does not infer "
            "or switch providers; set provider and transport explicitly."
        )


@dataclass(frozen=True)
class SessionTransportCapability:
    provider: str
    transport: str


def normalize_session_transport_value(value: str) -> str:
    return value.strip().lower().replace("-", "_")


def validate_session_transport(provider: str, transport: str) -> SessionTransportCapability:
    """Validate an explicit session provider/transport pair.

    This is intended to run in SDK registration paths and again in the control
    plane session-start path. It does not infer defaults.
    """

    normalized_provider = normalize_session_transport_value(provider)
    normalized_transport = normalize_session_transport_value(transport)

    if not normalized_provider:
        raise ValueError("Session provider is required; AgentField does not infer providers.")
    if not normalized_transport:
        raise ValueError("Session transport is required; AgentField does not infer transports.")

    supported = SUPPORTED_SESSION_TRANSPORTS.get(normalized_provider)
    if supported is None:
        known = ", ".join(sorted(SUPPORTED_SESSION_TRANSPORTS))
        raise ValueError(
            f"Unknown session provider '{provider}'. Known providers: {known}. "
            "Register provider capabilities before using a custom session provider."
        )

    if normalized_transport not in supported:
        raise SessionTransportError(normalized_provider, normalized_transport, supported)

    return SessionTransportCapability(
        provider=normalized_provider,
        transport=normalized_transport,
    )
