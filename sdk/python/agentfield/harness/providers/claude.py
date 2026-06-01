"""Claude Code provider using claude_agent_sdk (native Python SDK).

Uses lazy import - claude_agent_sdk is an optional dependency that's only
loaded when the claude-code provider is actually used.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from agentfield.harness._result import Metrics, RawResult


def _get_claude_sdk() -> Any:
    """Lazy import of claude_agent_sdk."""
    try:
        import claude_agent_sdk  # pyright: ignore[reportMissingImports]

        return claude_agent_sdk
    except ImportError as exc:
        raise ImportError(
            "claude_agent_sdk is required for the 'claude-code' provider. "
            "Install it with: pip install claude-agent-sdk"
        ) from exc


_PERMISSION_MAP = {
    "auto": "bypassPermissions",
    "plan": "plan",
}


class ClaudeCodeProvider:
    """Claude Code provider using the native claude_agent_sdk."""

    async def execute(self, prompt: str, options: dict[str, object]) -> RawResult:
        """Execute a prompt via Claude Code SDK."""
        sdk = _get_claude_sdk()

        agent_options: dict[str, object] = {}
        if options.get("model") is not None:
            agent_options["model"] = options["model"]
        if options.get("cwd") is not None:
            agent_options["cwd"] = options["cwd"]
        if options.get("max_turns") is not None:
            agent_options["max_turns"] = options["max_turns"]
        if options.get("tools") is not None:
            agent_options["allowed_tools"] = options["tools"]
        if options.get("system_prompt") is not None:
            agent_options["system_prompt"] = options["system_prompt"]
        if options.get("max_budget_usd") is not None:
            agent_options["max_budget_usd"] = options["max_budget_usd"]
        if options.get("permission_mode") is not None:
            raw_mode = str(options["permission_mode"])
            agent_options["permission_mode"] = _PERMISSION_MAP.get(raw_mode, raw_mode)
        if options.get("env") is not None:
            agent_options["env"] = options["env"]

        resume_sid = options.get("resume_session_id")
        if resume_sid:
            agent_options["resume"] = str(resume_sid)

        messages: List[Dict[str, Any]] = []
        result_text: Optional[str] = None
        total_cost: Optional[float] = None
        num_turns = 0
        session_id = ""
        start_api = time.monotonic()

        try:
            # Capture stderr so CLI failures can be diagnosed
            stderr_lines: list[str] = []

            def _on_stderr(line: str) -> None:
                stderr_lines.append(line)

            opts = (
                sdk.ClaudeAgentOptions(**agent_options)
                if hasattr(sdk, "ClaudeAgentOptions")
                else agent_options
            )
            # Set stderr callback after construction to avoid polluting
            # agent_options dict (which tests may assert on).
            if hasattr(opts, "stderr"):
                opts.stderr = _on_stderr

            msg_count = 0
            async for msg in sdk.query(prompt=prompt, options=opts):
                msg_count += 1
                if isinstance(msg, dict):
                    msg_dict = msg
                elif hasattr(msg, "__dict__"):
                    msg_dict = dict(msg.__dict__)
                else:
                    msg_dict = {"raw": str(msg)}

                messages.append(msg_dict)

                msg_type = str(msg_dict.get("type", ""))
                msg_subtype = str(msg_dict.get("subtype", ""))
                if msg_type == "result" or msg_subtype == "success":
                    raw_result = msg_dict.get("result", msg_dict.get("text", ""))
                    result_text = (
                        raw_result if isinstance(raw_result, str) else str(raw_result)
                    )
                    sid = msg_dict.get("session_id", "")
                    session_id = sid if isinstance(sid, str) else str(sid)
                    cost_info = msg_dict.get("cost_usd") or msg_dict.get(
                        "total_cost_usd"
                    )
                    if cost_info is not None:
                        total_cost = float(cost_info)
                    turns = msg_dict.get("num_turns")
                    num_turns = (
                        int(turns) if isinstance(turns, (int, float)) else len(messages)
                    )
                elif msg_type == "assistant" and result_text is None:
                    content = msg_dict.get("content")
                    message_obj = msg_dict.get("message")
                    if content is None and isinstance(message_obj, dict):
                        content = message_obj.get("content")

                    if isinstance(content, str):
                        result_text = content
                    elif isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                text = block.get("text")
                                if isinstance(text, str):
                                    result_text = text

            api_ms = int((time.monotonic() - start_api) * 1000)

            return RawResult(
                result=result_text,
                messages=messages,
                metrics=Metrics(
                    duration_ms=0,
                    duration_api_ms=api_ms,
                    num_turns=num_turns,
                    total_cost_usd=total_cost,
                    session_id=session_id,
                ),
                is_error=False,
            )
        except Exception as exc:
            import logging as _logging

            stderr_output = "\n".join(stderr_lines).strip()
            _logging.getLogger("agentfield.harness.claude").error(
                "ClaudeCodeProvider error: %s\nStderr output:\n%s",
                exc,
                stderr_output or "(no stderr captured)",
            )
            api_ms = int((time.monotonic() - start_api) * 1000)
            error_detail = str(exc)
            if stderr_output:
                error_detail = f"{error_detail}\nStderr output:\n{stderr_output}"
            return RawResult(
                result=None,
                messages=messages,
                metrics=Metrics(duration_api_ms=api_ms, session_id=session_id),
                is_error=True,
                error_message=error_detail,
            )
