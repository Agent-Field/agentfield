"""Shared pytest fixtures for the copilot-agent example."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pytest

# Make the example package importable as flat modules (`copilot_session`,
# `reasoners`) because the example ships as a script-style project rather
# than an installable package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class _FakeScopedMemory:
    def __init__(self, store: dict[str, Any]):
        self._store = store

    async def get(self, key: str, default: Any = None) -> Any:
        return self._store.get(key, default)

    async def set(self, key: str, value: Any) -> None:
        self._store[key] = value


class _FakeMemory:
    def __init__(self) -> None:
        self.sessions: dict[str, dict[str, Any]] = {}

    def session(self, session_id: str) -> _FakeScopedMemory:
        return _FakeScopedMemory(self.sessions.setdefault(session_id, {}))


class _FakeApp:
    """Minimal AgentField Agent stub for reasoner tests."""

    def __init__(self, node_id: str = "copilot-test") -> None:
        self.node_id = node_id
        self.memory = _FakeMemory()
        self._reasoners: dict[str, Callable[..., Any]] = {}

    def reasoner(self) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def wrap(fn: Callable[..., Any]) -> Callable[..., Any]:
            self._reasoners[fn.__name__] = fn
            return fn
        return wrap


@pytest.fixture
def fake_app() -> _FakeApp:
    return _FakeApp()


@pytest.fixture
def stub_copilot_client(monkeypatch: pytest.MonkeyPatch):
    """Replace ``CopilotClient`` with an async-context stub that fires a
    scripted sequence of :class:`SessionEvent`-shaped objects through the
    caller's ``on_event`` callback.

    Returns a helper that the test configures with ``events`` and
    ``final_event`` before invoking the reasoner/wrapper.
    """

    state: dict[str, Any] = {
        "events": [],
        "final_event": None,
        "raise_timeout": False,
        "captured_session_kwargs": None,
        "captured_prompt": None,
    }

    class _FakeSession:
        def __init__(self, on_event: Callable[[Any], None]) -> None:
            self._on_event = on_event

        async def __aenter__(self) -> "_FakeSession":
            return self

        async def __aexit__(self, *_a: Any) -> None:
            return None

        async def send_and_wait(
            self,
            prompt: str,
            *,
            attachments: Any = None,
            mode: Any = None,
            timeout: float = 60.0,
        ) -> Any:
            state["captured_prompt"] = prompt
            if state["raise_timeout"]:
                raise TimeoutError("fake timeout")
            for ev in state["events"]:
                self._on_event(ev)
            return state["final_event"]

    class _FakeClient:
        def __init__(self, *a: Any, **kw: Any) -> None:
            pass

        async def __aenter__(self) -> "_FakeClient":
            return self

        async def __aexit__(self, *_a: Any) -> None:
            return None

        async def create_session(self, **kwargs: Any) -> _FakeSession:
            state["captured_session_kwargs"] = kwargs
            return _FakeSession(kwargs["on_event"])

    import copilot_session as cs
    monkeypatch.setattr(cs, "CopilotClient", _FakeClient)
    return state


def make_event(event_type: Any, **data_fields: Any) -> SimpleNamespace:
    """Build a minimal duck-typed SessionEvent for the wrapper tests."""
    return SimpleNamespace(
        type=event_type,
        data=SimpleNamespace(**data_fields),
        id="e1",
        timestamp="now",
        ephemeral=False,
        parent_id=None,
    )
