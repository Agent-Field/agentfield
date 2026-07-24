"""Realtime/session DX primitives for AgentField agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional

from .session_transport import validate_session_transport


SessionHandler = Callable[["RealtimeSession"], Awaitable[Any]]


@dataclass(frozen=True)
class SessionDefinition:
    name: str
    provider: str
    transport: str
    model: Optional[str] = None
    modalities: List[str] = field(default_factory=lambda: ["audio", "text"])
    voice: Optional[str] = None
    tools: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    proposed_tags: List[str] = field(default_factory=list)
    approved_tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "provider": self.provider,
            "transport": self.transport,
            "model": self.model,
            "modalities": list(self.modalities),
            "voice": self.voice,
            "tools": list(self.tools),
            "tags": list(self.tags),
            "proposed_tags": list(self.proposed_tags or self.tags),
            "approved_tags": list(self.approved_tags),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class SessionTurn:
    text: Optional[str] = None
    transcript: Optional[str] = None
    audio: Optional[Any] = None
    audio_format: Optional[str] = None
    channel: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class RealtimeSession:
    """Session handler context.

    The control plane owns media transport. This object is the app-facing
    context passed to a session handler and keeps agent work on app.call.
    """

    def __init__(self, app: Any, session_id: str, definition: SessionDefinition):
        self.app = app
        self.session_id = session_id
        self.definition = definition
        self._outbox: List[Dict[str, Any]] = []

    async def input(self) -> SessionTurn:
        raise RuntimeError(
            "session.input() is populated by the AgentField control plane transport adapter"
        )

    async def call(self, target: str, *args: Any, **kwargs: Any) -> Any:
        return await self.app.call(target, *args, **kwargs)

    async def say(self, text: str, **metadata: Any) -> Dict[str, Any]:
        event = {"type": "speech", "text": text, "metadata": metadata}
        self._outbox.append(event)
        return event

    @property
    def outbox(self) -> List[Dict[str, Any]]:
        return list(self._outbox)


def build_session_definition(
    name: str,
    *,
    provider: str,
    transport: str,
    model: Optional[str] = None,
    modalities: Optional[List[str]] = None,
    voice: Optional[str] = None,
    tools: Optional[List[str]] = None,
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> SessionDefinition:
    capability = validate_session_transport(provider, transport)
    return SessionDefinition(
        name=name,
        provider=capability.provider,
        transport=capability.transport,
        model=model,
        modalities=list(modalities or ["audio", "text"]),
        voice=voice,
        tools=list(tools or []),
        tags=list(tags or []),
        proposed_tags=list(tags or []),
        metadata=dict(metadata or {}),
    )
