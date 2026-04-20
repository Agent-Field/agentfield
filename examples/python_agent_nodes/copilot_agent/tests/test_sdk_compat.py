"""Compatibility smoke test for ``github-copilot-sdk==0.2.2``.

Asserts that every SDK symbol this example depends on exists and has the
expected shape. If the PyPI package drifts — e.g. renames
``send_and_wait`` or removes a ``SessionEventType`` value — this test
fails loudly BEFORE users hit an opaque ``AttributeError`` at runtime.

Per rubber-duck finding #6 ("SDK preview risk").
"""

from __future__ import annotations

import inspect

import pytest

pytest.importorskip("copilot")


def test_copilot_client_symbols() -> None:
    from copilot import CopilotClient, SubprocessConfig  # noqa: F401

    params = inspect.signature(CopilotClient.create_session).parameters
    for expected in (
        "on_permission_request",
        "model",
        "session_id",
        "working_directory",
        "config_dir",
        "skill_directories",
        "available_tools",
        "excluded_tools",
        "on_event",
        "system_message",
    ):
        assert expected in params, f"CopilotClient.create_session missing {expected}"


def test_session_send_and_wait_signature() -> None:
    from copilot.session import CopilotSession

    params = inspect.signature(CopilotSession.send_and_wait).parameters
    assert "prompt" in params
    assert "timeout" in params


def test_permission_handler_present() -> None:
    from copilot.session import PermissionHandler, PermissionRequestResult

    assert callable(PermissionHandler.approve_all)
    # PermissionRequestResult must accept the kind= we use for denial.
    denied = PermissionRequestResult(kind="denied", message="m")
    assert denied.kind == "denied"


def test_session_event_type_values() -> None:
    from copilot.generated.session_events import SessionEventType

    required = {
        "SESSION_IDLE",
        "SESSION_ERROR",
        "ASSISTANT_MESSAGE",
        "ASSISTANT_TURN_END",
        "ASSISTANT_USAGE",
        "TOOL_EXECUTION_START",
        "TOOL_EXECUTION_COMPLETE",
        "PERMISSION_REQUESTED",
        "SESSION_SHUTDOWN",
    }
    have = {e.name for e in SessionEventType}
    missing = required - have
    assert not missing, f"SessionEventType missing values: {missing}"


def test_session_event_shape() -> None:
    import dataclasses

    from copilot.generated.session_events import SessionEvent

    fields = {f.name for f in dataclasses.fields(SessionEvent)}
    for expected in ("data", "id", "timestamp", "type"):
        assert expected in fields, f"SessionEvent missing field {expected}"
