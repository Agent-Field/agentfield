"""Thin wrapper around github-copilot-sdk 0.2.2 that turns a Copilot session
into an AgentField-shaped structured response.

Design decisions (see research/docs/2026-04-20-copilot-cli-support.md and the
rubber-duck critique in the plan):

* By default, do NOT set ``config_dir`` on the session — Copilot uses the
  user's real ``~/.copilot/`` so skills installed by ``af skill install``
  and auth from ``copilot --login`` are visible.
* Opt-in isolation via ``isolate=True`` or ``AGENTFIELD_COPILOT_ISOLATE=1``
  uses a per-node directory, not per-run, so concurrent reasoner calls on
  the same agent share the same sandbox rather than each getting a fresh
  (auth-less, skill-less) one.
* The AgentField session id is mapped to a Copilot session id via
  session-scoped memory, keeping the two namespaces separate.
* Events are discriminated by ``SessionEventType`` (stable enum), never by
  isinstance on ``event.data`` — the Data dataclass is a union in v0.2.2
  and subclass names are not exported.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Optional
from uuid import uuid4

from copilot import CopilotClient, SubprocessConfig
from copilot.session import (
    PermissionHandler,
    PermissionRequestResult,
)
from copilot.generated.session_events import (
    SessionEvent,
    SessionEventType,
)


def deny_all_handler(request: Any, invocation: dict[str, str]) -> PermissionRequestResult:
    """Permission handler that denies every tool invocation.

    Used by reasoners that must not execute anything (``ask``, ``plan``).
    Also used as the default when no explicit policy is requested.
    """
    return PermissionRequestResult(
        kind="denied",
        message="tool execution not permitted in this reasoner",
    )


@dataclass
class CopilotRunResult:
    """Structured output returned by :func:`run_copilot`.

    Stable across SDK preview churn — the Copilot SDK's own event-data
    classes are not re-exported here.
    """

    af_session_id: Optional[str]
    copilot_session_id: str
    model: Optional[str]
    answer: str
    transcript: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    finished_reason: str = "idle"  # "idle" | "error" | "timeout" | "aborted"
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "af_session_id": self.af_session_id,
            "copilot_session_id": self.copilot_session_id,
            "model": self.model,
            "answer": self.answer,
            "transcript": self.transcript,
            "tool_calls": self.tool_calls,
            "usage": self.usage,
            "finished_reason": self.finished_reason,
            "error": self.error,
        }


def _build_subprocess_config() -> Optional[SubprocessConfig]:
    """Forward a Copilot GitHub token from the agent env if present.

    Auth precedence mirrors the SDK:
        1. ``COPILOT_GITHUB_TOKEN``
        2. ``GH_TOKEN``
        3. ``GITHUB_TOKEN``
        4. logged-in ``copilot`` user (no env needed — SDK picks it up)
    """
    for var in ("COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN"):
        token = os.getenv(var, "").strip()
        if token:
            return SubprocessConfig(github_token=token)
    return None


def _resolve_config_dir(isolate: bool, node_id: str) -> Optional[str]:
    """Decide which Copilot ``config_dir`` to use.

    Returns ``None`` when the caller should reuse the user's real
    ``~/.copilot/``. Returns a per-node sandbox path when ``isolate=True``
    or ``AGENTFIELD_COPILOT_ISOLATE=1``.
    """
    env_flag = os.getenv("AGENTFIELD_COPILOT_ISOLATE", "").strip().lower()
    if not isolate and env_flag not in ("1", "true", "yes", "on"):
        return None
    home = os.getenv("AGENTFIELD_HOME") or os.path.expanduser("~/.agentfield")
    path = os.path.join(home, "copilot-home", node_id)
    os.makedirs(path, exist_ok=True)
    return path


async def _resolve_copilot_session_id(
    app: Any,
    af_session_id: Optional[str],
    continue_session: bool,
) -> str:
    """Map the AgentField session to a Copilot session id.

    Stored in session-scoped memory under
    ``copilot_session_id:<af_session_id>``. Fresh uuid when no mapping
    exists, or when ``continue_session`` is False.
    """
    if not af_session_id or not continue_session or app is None or app.memory is None:
        return str(uuid4())
    try:
        scoped = app.memory.session(af_session_id)
        existing = await scoped.get("copilot_session_id")
        if isinstance(existing, str) and existing:
            return existing
        new_id = str(uuid4())
        await scoped.set("copilot_session_id", new_id)
        return new_id
    except Exception:
        # Memory is best-effort for session mapping; fall back to a fresh id.
        return str(uuid4())


def _assistant_text(event: SessionEvent) -> Optional[str]:
    """Extract the final assistant message text from an event, if any."""
    data = event.data
    for attr in ("content", "message", "transformed_content", "summary_content"):
        val = getattr(data, attr, None)
        if isinstance(val, str) and val.strip():
            return val
    return None


async def run_copilot(
    *,
    app: Any,
    prompt: str,
    node_id: str,
    af_session_id: Optional[str] = None,
    cwd: Optional[str] = None,
    model: Optional[str] = None,
    isolate: bool = False,
    continue_session: bool = False,
    available_tools: Optional[list[str]] = None,
    excluded_tools: Optional[list[str]] = None,
    system_message: Optional[str] = None,
    permission_handler: Any = deny_all_handler,
    timeout: float = 120.0,
) -> CopilotRunResult:
    """Run a single Copilot turn and return a structured result.

    ``available_tools=[]`` forces Copilot into a read-only / pure-reasoning
    mode; combined with ``permission_handler=_deny_all_handler`` this makes
    the reasoner safe by default. Callers that want full agent mode pass
    ``permission_handler=PermissionHandler.approve_all`` and either leave
    ``available_tools=None`` (all first-party tools) or provide an allow
    list.
    """
    copilot_session_id = await _resolve_copilot_session_id(
        app, af_session_id, continue_session
    )
    config_dir = _resolve_config_dir(isolate, node_id)
    subprocess_config = _build_subprocess_config()

    client_kwargs: dict[str, Any] = {}
    if subprocess_config is not None:
        client_kwargs["subprocess_config"] = subprocess_config

    transcript: list[dict[str, Any]] = []
    tool_calls: list[dict[str, Any]] = []
    usage: dict[str, int] = {}
    error_msg: Optional[str] = None
    final_event: Optional[SessionEvent] = None

    def _on_event(event: SessionEvent) -> None:
        nonlocal error_msg
        et = event.type
        data = event.data
        if et == SessionEventType.ASSISTANT_MESSAGE:
            text = _assistant_text(event)
            if text:
                transcript.append({"role": "assistant", "content": text})
        elif et == SessionEventType.TOOL_EXECUTION_START:
            tool_calls.append(
                {
                    "tool_call_id": getattr(data, "tool_call_id", None),
                    "tool_name": getattr(data, "tool_name", None),
                    "status": "started",
                }
            )
            transcript.append(
                {
                    "role": "tool_call",
                    "tool_name": getattr(data, "tool_name", None),
                    "tool_call_id": getattr(data, "tool_call_id", None),
                }
            )
        elif et == SessionEventType.TOOL_EXECUTION_COMPLETE:
            tool_call_id = getattr(data, "tool_call_id", None)
            for tc in tool_calls:
                if tc.get("tool_call_id") == tool_call_id:
                    tc["status"] = "complete"
                    break
        elif et == SessionEventType.ASSISTANT_USAGE:
            for k in ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens"):
                v = getattr(data, k, None)
                if isinstance(v, int):
                    usage[k] = usage.get(k, 0) + v
        elif et == SessionEventType.SESSION_ERROR:
            error_msg = getattr(data, "message", None) or "unknown session error"

    async with CopilotClient(**client_kwargs) as client:
        session_kwargs: dict[str, Any] = {
            "on_permission_request": permission_handler,
            "session_id": copilot_session_id,
            "on_event": _on_event,
        }
        if model:
            session_kwargs["model"] = model
        if cwd:
            session_kwargs["working_directory"] = cwd
        if config_dir:
            session_kwargs["config_dir"] = config_dir
        if available_tools is not None:
            session_kwargs["available_tools"] = available_tools
        if excluded_tools is not None:
            session_kwargs["excluded_tools"] = excluded_tools
        if system_message:
            session_kwargs["system_message"] = system_message

        async with await client.create_session(**session_kwargs) as session:
            try:
                final_event = await session.send_and_wait(prompt, timeout=timeout)
            except TimeoutError as exc:
                return CopilotRunResult(
                    af_session_id=af_session_id,
                    copilot_session_id=copilot_session_id,
                    model=model,
                    answer="",
                    transcript=transcript,
                    tool_calls=tool_calls,
                    usage=usage,
                    finished_reason="timeout",
                    error=str(exc),
                )

    answer = ""
    for entry in reversed(transcript):
        if entry.get("role") == "assistant" and isinstance(entry.get("content"), str):
            answer = entry["content"]
            break
    if not answer and final_event is not None:
        answer = _assistant_text(final_event) or ""

    finished_reason = "idle"
    if error_msg:
        finished_reason = "error"
    elif final_event is not None and final_event.type == SessionEventType.SESSION_ERROR:
        finished_reason = "error"
        error_msg = error_msg or _assistant_text(final_event) or "session error"

    return CopilotRunResult(
        af_session_id=af_session_id,
        copilot_session_id=copilot_session_id,
        model=model,
        answer=answer,
        transcript=transcript,
        tool_calls=tool_calls,
        usage=usage,
        finished_reason=finished_reason,
        error=error_msg,
    )
