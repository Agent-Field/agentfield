"""AgentField reasoners that expose GitHub Copilot CLI as a first-class node.

Permission posture per rubber-duck finding #3:

* ``ask`` / ``plan`` — no tools available, deny-by-default permission handler.
* ``review`` — narrow read-only allow list.
* ``run_task`` — opt-in ``allow_tools=True`` required; defaults to safe.

Every reasoner returns :class:`CopilotRunResult` serialized to a dict so the
structured fields survive the AgentField reasoner boundary even as the
underlying Copilot SDK (Public Preview) evolves.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from copilot.session import PermissionHandler

from copilot_session import CopilotRunResult, deny_all_handler, run_copilot


_DEFAULT_MODEL = os.getenv("COPILOT_MODEL", "gpt-5")


def register(app: Any) -> None:
    """Register all reasoners on the given AgentField :class:`Agent`."""

    node_id = getattr(app, "node_id", None) or os.getenv("AGENT_NODE_ID", "copilot")

    async def _run(**kwargs: Any) -> dict[str, Any]:
        ctx = kwargs.pop("_ctx", None)
        af_session_id = getattr(ctx, "session_id", None) if ctx is not None else None
        result: CopilotRunResult = await run_copilot(
            app=app,
            af_session_id=af_session_id,
            node_id=node_id,
            **kwargs,
        )
        return result.to_dict()

    @app.reasoner()
    async def ask(
        prompt: str,
        model: Optional[str] = None,
        cwd: Optional[str] = None,
        isolate: bool = False,
        continue_session: bool = False,
        timeout: float = 60.0,
        execution_context: Any = None,
    ) -> dict[str, Any]:
        """One-shot Q&A. No tool execution, no filesystem access.

        Use this when you want Copilot to reason over a prompt and respond
        with text only.
        """
        return await _run(
            prompt=prompt,
            model=model or _DEFAULT_MODEL,
            cwd=cwd,
            isolate=isolate,
            continue_session=continue_session,
            available_tools=[],
            permission_handler=deny_all_handler,
            timeout=timeout,
            _ctx=execution_context,
        )

    @app.reasoner()
    async def plan(
        task: str,
        cwd: Optional[str] = None,
        model: Optional[str] = None,
        isolate: bool = False,
        timeout: float = 120.0,
        execution_context: Any = None,
    ) -> dict[str, Any]:
        """Ask Copilot to produce a step-by-step plan without executing tools."""
        prompt = (
            "Produce a detailed step-by-step plan for the following task. "
            "Do NOT execute anything — output plan text only.\n\n"
            f"Task:\n{task}"
        )
        return await _run(
            prompt=prompt,
            model=model or _DEFAULT_MODEL,
            cwd=cwd,
            isolate=isolate,
            available_tools=[],
            permission_handler=deny_all_handler,
            timeout=timeout,
            _ctx=execution_context,
        )

    @app.reasoner()
    async def review(
        diff: Optional[str] = None,
        files: Optional[list[str]] = None,
        cwd: Optional[str] = None,
        model: Optional[str] = None,
        isolate: bool = False,
        timeout: float = 180.0,
        execution_context: Any = None,
    ) -> dict[str, Any]:
        """Review a diff or a set of files. Read-only tools allowed."""
        if diff:
            prompt = (
                "Review the following diff for bugs, security issues, and "
                "logic errors. Return a numbered list of findings.\n\n"
                f"```diff\n{diff}\n```"
            )
            # With an inline diff we do not need any tools.
            return await _run(
                prompt=prompt,
                model=model or _DEFAULT_MODEL,
                cwd=cwd,
                isolate=isolate,
                available_tools=[],
                permission_handler=deny_all_handler,
                timeout=timeout,
                _ctx=execution_context,
            )

        if not files:
            return {
                "error": "review requires either `diff` or `files`",
                "af_session_id": None,
                "copilot_session_id": "",
                "model": model or _DEFAULT_MODEL,
                "answer": "",
                "transcript": [],
                "tool_calls": [],
                "usage": {},
                "finished_reason": "error",
            }

        file_list = "\n".join(f"- {p}" for p in files)
        prompt = (
            "Review the following files in the working tree. Read them, then "
            "produce a numbered list of findings (bugs, security issues, "
            "logic errors). Do not modify any files.\n\n"
            f"Files to review:\n{file_list}"
        )
        return await _run(
            prompt=prompt,
            model=model or _DEFAULT_MODEL,
            cwd=cwd,
            isolate=isolate,
            available_tools=["read_file", "list_directory", "grep", "git_diff"],
            permission_handler=PermissionHandler.approve_all,
            timeout=timeout,
            _ctx=execution_context,
        )

    @app.reasoner()
    async def run_task(
        task: str,
        cwd: Optional[str] = None,
        model: Optional[str] = None,
        allow_tools: bool = False,
        allow_list: Optional[list[str]] = None,
        deny_list: Optional[list[str]] = None,
        isolate: bool = False,
        continue_session: bool = False,
        timeout: float = 600.0,
        execution_context: Any = None,
    ) -> dict[str, Any]:
        """Full agent mode — Copilot plans and executes.

        Dangerous by default, so ``allow_tools`` must be set explicitly.
        When ``allow_list`` is provided it takes precedence over
        ``deny_list`` (allow-lists are strictly safer).
        """
        if not allow_tools:
            return {
                "error": "run_task requires allow_tools=True; tool execution refused.",
                "af_session_id": getattr(execution_context, "session_id", None),
                "copilot_session_id": "",
                "model": model or _DEFAULT_MODEL,
                "answer": "",
                "transcript": [],
                "tool_calls": [],
                "usage": {},
                "finished_reason": "error",
            }

        available = allow_list if allow_list else None
        excluded = None if allow_list else deny_list

        return await _run(
            prompt=task,
            model=model or _DEFAULT_MODEL,
            cwd=cwd,
            isolate=isolate,
            continue_session=continue_session,
            available_tools=available,
            excluded_tools=excluded,
            permission_handler=PermissionHandler.approve_all,
            timeout=timeout,
            _ctx=execution_context,
        )

    # Expose reasoners on the registration result so tests can reach them
    # directly without going through the HTTP layer.
    register.ask = ask
    register.plan = plan
    register.review = review
    register.run_task = run_task
